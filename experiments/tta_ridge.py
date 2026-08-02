"""Test-time augmentation for the conv-ridge family, priced in bytes.

Same question as `tta_cnn.py`, asked where the answer is least obvious: below a
kilobyte, where two-thirds of the artifact is `predict.py` and forty bytes of
extra aggregation is a real fraction of the budget.

The ridge head is linear, so T views cost one matmul rather than T: the views
are summed in feature space and the head is fit against the same sum. That is
why flip TTA was ~12 bytes here and not a second scoring pass, and it is why
shifts are also cheap — the only new source is the shift list itself.

One detail that is free and matters, and that a first version of this module got
wrong. Summing T views scales the features by about T, which weakens the
effective ridge penalty by T^2 — so a fixed `lambda` is not a fixed amount of
regularization across schedules, and comparing schedules at one lambda compares
two things at once. The first emitted artifact scored 2.8 points *below* its own
incumbent on test for exactly that reason. The Gram matrix is therefore
accumulated once per schedule and solved at every lambda in a grid, with the
best picked on validation, so each schedule is seen at its own best setting.
At T=2 and lambda=1e2 this reproduces `golf.py` exactly, which is the check that
the incumbent is inside the grid rather than outside it.

Selection is on `tinycifar.data.load_dev()`: heads are fit on the 45,000-image
fit split and scored on the 5,000-image validation split. `--final` refits on
the full training set — matching how the incumbent `g-*` artifacts were built —
and scores test once.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from experiments.baselines import emit  # noqa: E402
from experiments.conv_features import build_feats, geometry, make_feats_src  # noqa: E402
from experiments.golf import scale_expr, unpack_expr  # noqa: E402
from experiments.quant_sweep import lloyd_max  # noqa: E402
from experiments.tta_cnn import SCHEDULES, mcnemar, shift_repr  # noqa: E402
from tinycifar import artifact as A  # noqa: E402
from tinycifar import pack as P  # noqa: E402
from tinycifar.data import load, load_dev  # noqa: E402
from tinycifar.evaluate import evaluate, summarize  # noqa: E402

HEAD = '''import numpy as np
B=open(__file__[:-10]+"w","rb").read()
C=np.frombuffer(B,np.float16,{K},1).astype(np.float32)
W=C[{UNPACK}].reshape({D},10)
g=np.random.default_rng({SEED})
R=g.standard_normal(({NF},{PP}),dtype=np.float32)*{S}
t=g.standard_normal({NF},dtype=np.float32)*{TS}
def f(x):
 c=np.lib.stride_tricks.sliding_window_view(x,({P},{P}),(1,2))[:,::{ST},::{ST}].transpose(0,1,2,4,5,3).reshape(len(x),{G}*{G},{PP})
 c=(c-c.mean(2,keepdims=True))/np.sqrt(c.var(2,keepdims=True)+.01)
 h=np.maximum(c@R.T-t,0).reshape(len(x),{PG},{PS},{PG},{PS},{NF})
 return h.mean((2,4)).reshape(len(x),-1)
'''

TAIL_FLIP = '''def predict(x):
 return np.concatenate([np.argmax((f(z)+f(z[:,:,::-1]))@W[:-1]+W[-1],1)for z in np.array_split(x.astype(np.float32)/255,-(-len(x)//500))])
'''

TAIL_PLAIN = '''def predict(x):
 return np.concatenate([np.argmax(f(z)@W[:-1]+W[-1],1)for z in np.array_split(x.astype(np.float32)/255,-(-len(x)//500))])
'''

TAIL_ROLL = '''def T(z):
 return {AGG}
def predict(x):
 return np.concatenate([np.argmax(sum(T(np.roll(z,d,(1,2)))for d in {S})@W[:-1]+W[-1],1)for z in np.array_split(x.astype(np.float32)/255,-(-len(x)//500))])
'''

TAIL_EDGE = '''def T(z):
 return {AGG}
def predict(x):
 return np.concatenate([np.argmax(sum(T(p[:,{R}+i:{R}+32+i,{R}+j:{R}+32+j])for i,j in {S})@W[:-1]+W[-1],1)for p in[np.pad(z,((0,0),({R},{R}),({R},{R}),(0,0)),"edge")for z in np.array_split(x.astype(np.float32)/255,-(-len(x)//500))]])
'''

AGG_FLIP = "f(z)+f(z[:,:,::-1])"
AGG_PLAIN = "f(z)"

CHUNK = 2000


def make_tail(shifts, flip, mode):
    if list(shifts) == [(0, 0)]:
        return TAIL_FLIP if flip else TAIL_PLAIN
    agg = AGG_FLIP if flip else AGG_PLAIN
    if mode == "wrap":
        return TAIL_ROLL.format(AGG=agg, S=shift_repr(shifts))
    r = max(max(abs(i), abs(j)) for i, j in shifts)
    return TAIL_EDGE.format(AGG=agg, S=shift_repr(shifts), R=r)


def shift(z, s, mode):
    if s == (0, 0):
        return z
    if mode == "wrap":
        return np.roll(z, s, (1, 2))
    r = max(abs(s[0]), abs(s[1]))
    p = np.pad(z, ((0, 0), (r, r), (r, r), (0, 0)), "edge")
    return p[:, r + s[0]:r + 32 + s[0], r + s[1]:r + 32 + s[1]]


def sum_feats(feats, z, shifts, flip, mode):
    """Exactly the quantity the decoder sums — no rescaling anywhere."""
    acc = None
    for s in shifts:
        v = shift(z, s, mode)
        for w in ((v, v[:, :, ::-1]) if flip else (v,)):
            g = feats(w)
            acc = g if acc is None else acc + g
    return acc


def design(feats, x, shifts, flip, mode, lo=0, hi=None):
    f = sum_feats(feats, x[lo:hi].astype(np.float32) / 255, shifts, flip, mode)
    return np.hstack([f, np.ones((len(f), 1), np.float32)]).astype(np.float64)


def fit_gram(feats, x, y, dim, shifts, flip, mode):
    """Accumulate the normal equations once; lambda is applied at solve time."""
    G = np.zeros((dim + 1, dim + 1))
    Hy = np.zeros((dim + 1, 10))
    for i in range(0, len(x), CHUNK):
        f = design(feats, x, shifts, flip, mode, i, i + CHUNK)
        Y = np.zeros((len(f), 10))
        Y[np.arange(len(f)), y[i:i + CHUNK]] = 1.0
        G += f.T @ f
        Hy += f.T @ Y
    return G, Hy


def solve(G, Hy, lam):
    Gl = G.copy()
    Gl.flat[::G.shape[0] + 1] += lam
    return np.linalg.solve(Gl, Hy)


def pack(W, bits, k, patch, stride, pool, seed, shifts, flip, mode, tscale=0.1):
    g, pg = geometry(patch, stride, pool)
    dim = k * pg * pg
    c, idx = lloyd_max(W, bits)                        # global codebook
    blob = struct.pack("<B", bits) + c.tobytes() + P.bitpack(idx, bits)
    off = 1 + 2 * (1 << bits)
    src = HEAD.format(
        K=1 << bits, UNPACK=unpack_expr(bits, (dim + 1) * 10, off),
        D=dim + 1, SEED=seed, NF=k, PP=patch * patch * 3,
        S=scale_expr(patch), TS=tscale, P=patch, ST=stride, G=g, PG=pg, PS=pool,
    ) + make_tail(shifts, flip, mode)
    return {"predict.py": src.encode(), "w": blob}, c[idx].reshape(W.shape)


def make_feats(k, patch, stride, pool, seed, tscale=0.1):
    scale = 1.0 / np.sqrt(patch * patch * 3)
    src, _ = make_feats_src(k, patch, stride, pool, seed, scale, tscale)
    return build_feats(src)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--patch", type=int, default=4)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--pool", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--lams", nargs="*", type=float,
                    default=[25.0, 1e2, 4e2, 1.6e3, 6.4e3])
    ap.add_argument("--scheds", nargs="*", default=list(SCHEDULES))
    ap.add_argument("--final", default="",
                    help="schedule to refit on full train and score on TEST")
    ap.add_argument("--name", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)

    xfit, yfit, xva, yva = load_dev()
    feats = make_feats(a.k, a.patch, a.stride, a.pool, a.seed)
    dim = a.k * geometry(a.patch, a.stride, a.pool)[1] ** 2
    args = (a.k, a.patch, a.stride, a.pool, a.seed)

    rows, ref_ok, base = [], None, None
    for name in a.scheds:
        shifts, flip, mode = SCHEDULES[name]
        t0 = time.perf_counter()
        G, Hy = fit_gram(feats, xfit, yfit, dim, shifts, flip, mode)
        Fva = np.concatenate([design(feats, xva, shifts, flip, mode, i, i + 500)
                              for i in range(0, len(xva), 500)])
        best = None
        for lam in a.lams:
            W = solve(G, Hy, lam)
            files, Wq = pack(W, a.bits, *args, shifts, flip, mode)
            ok = np.argmax(Fva @ Wq, 1) == yva
            n = A.measure(files).description_length
            if best is None or ok.mean() > best["val"]:
                best = dict(name=name, val=float(ok.mean()), bytes=n, ok=ok,
                            lam=lam, nviews=len(shifts) * (2 if flip else 1))
        best["train_seconds"] = round(time.perf_counter() - t0, 1)
        if name == "flip":
            ref_ok, base = best["ok"], best["bytes"]
        rows.append(best)
        print(f"  {name:<13} {best['nviews']:2d} views  "
              f"val {best['val'] * 100:6.2f}%  {best['bytes']:>7,} B  "
              f"(lam {best['lam']:g}, {best['train_seconds']:.0f}s)", flush=True)

    print(f"\nk={a.k} bits={a.bits}, against the flip incumbent ({base:,} B):")
    ref = next(x for x in rows if x["name"] == "flip")
    for r in rows:
        lo_, wi_, p = mcnemar(ref_ok, r["ok"])
        r.update(win=wi_, lose=lo_, p=p, delta_bytes=r["bytes"] - base)
        print(f"  {r['name']:<13} {(r['val'] - ref['val']) * 100:+6.2f} pp  "
              f"{r['delta_bytes']:+5d} B  paired +{r['win']}/-{r['lose']} "
              f"p={r['p']:.3g}")

    if a.final:
        shifts, flip, mode = SCHEDULES[a.final]
        v = next(r for r in rows if r["name"] == a.final)
        xtr, ytr, _, _ = load()
        t0 = time.perf_counter()
        G, Hy = fit_gram(feats, xtr, ytr, dim, shifts, flip, mode)
        files, _ = pack(solve(G, Hy, v["lam"]), a.bits, *args, shifts, flip, mode)
        ts = time.perf_counter() - t0
        nm = a.name or (f"g-k{a.k}-p{a.patch}s{a.stride}-{a.bits}b-{a.final}"
                        .replace("+", ""))
        d = emit(nm, files)
        rec = evaluate(
            d, name=nm,
            method=f"golfed: random {a.patch}x{a.patch} conv (k={a.k}) + ridge "
                   f"head, {a.bits}-bit global codebook, {a.final} TTA "
                   f"({v['nviews']} views)",
            notes=f"TTA schedule and lambda={v['lam']:g} selected on val "
                  f"({v['val'] * 100:.2f}%); head refit on the full "
                  f"50,000-image train set",
            train_seconds=ts)
        print(f"\nTEST: {summarize(rec)}")

    if a.out:
        Path(a.out).write_text(json.dumps(
            [{k: v for k, v in r.items() if k != "ok"} for r in rows], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
