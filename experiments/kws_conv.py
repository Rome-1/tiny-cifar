"""Random convolutional features over the mel map -- the repo's own sub-4 KB family.

`kws_ridge.py` puts a linear head straight on the log-mel map and tops out
around 50% in float. That understates what this repo can do, because its whole
sub-4 KB CIFAR frontier is not a linear model: it is random convolutional
filters drawn from a four-byte seed, rectified and pooled, with a ridge head on
top. The filters cost a seed however many there are (lever C), so the only real
weight is the head.

This file is that family with the image swapped for a log-mel spectrogram. The
front end is `kws_ridge.feat_src` verbatim; everything above it is
`experiments/conv_features.py`'s shape, re-emitted for a [frames x bands] map
and twelve classes.

Selection is the same as `kws_ridge.py`: `(k, pool, bits)` is declared and
scored on test, only the ridge penalty is chosen and it is chosen on validation
through the quantized artifact. The seed is fixed at 0 and never searched --
ranking seeds on anything would be a best-of-N this measurement does not need.

Run:
    python3 experiments/kws_conv.py --out /tmp/kws_conv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from experiments.golf import unpack_expr  # noqa: E402
from experiments.kws_ridge import (  # noqa: E402
    K, check_roundtrip, compile_feat, feat_src, features, fit, pack_indices,
    quantize,
)
from tinycifar import artifact as A  # noqa: E402
from tinycifar.evaluate_nclass import evaluate_arrays, summarize  # noqa: E402

TEMPLATE = '''import numpy as np
B=open(__file__[:-10]+"w","rb").read()
C=np.frombuffer(B,np.float16,{NCB}).astype(np.float32)
W=C[{UNPACK}].reshape({D},{K})
g=np.random.default_rng(0)
R=g.standard_normal(({NF},{PP}),dtype=np.float32)*{PP}**-.5
t=g.standard_normal({NF},dtype=np.float32)*.5
def predict(x):
 {FEAT}
 c=np.lib.stride_tricks.sliding_window_view(v.reshape(len(x),{F},{M}),({P},{P}),(1,2)).reshape(len(x),{GF}*{GM},{PP})
 c=(c-c.mean(2,keepdims=True))/np.sqrt(c.var(2,keepdims=True)+.01)
 h=np.maximum(c@R.T-t,0).reshape(len(x),{PF},{SF},{PM},{SM},{NF}).mean((2,4)).reshape(len(x),-1)
 return np.argmax(h@W+C[{NC}:],1)
'''


def geometry(frames: int, bands: int, patch: int, pool_f: int, pool_m: int):
    """(#windows along each axis, pooled grid, stride) or None if it does not tile."""
    gf, gm = frames - patch + 1, bands - patch + 1
    if gf % pool_f or gm % pool_m:
        return None
    return gf, gm, gf // pool_f, gm // pool_m


def conv_feats(V, frames, bands, patch, pool_f, pool_m, nf, seed=0, chunk=4000):
    """Trainer-side mirror of the emitted expression, same rng call sequence."""
    g = np.random.default_rng(seed)
    pp = patch * patch
    R = g.standard_normal((nf, pp), dtype=np.float32) * pp ** -0.5
    t = g.standard_normal(nf, dtype=np.float32) * 0.5
    gf, gm, sf, sm = geometry(frames, bands, patch, pool_f, pool_m)
    out = []
    for i in range(0, len(V), chunk):
        m = V[i:i + chunk].reshape(-1, frames, bands)
        c = np.lib.stride_tricks.sliding_window_view(
            m, (patch, patch), (1, 2)).reshape(len(m), gf * gm, pp)
        c = (c - c.mean(2, keepdims=True)) / np.sqrt(c.var(2, keepdims=True) + .01)
        h = np.maximum(c @ R.T - t, 0)
        out.append(h.reshape(len(m), pool_f, sf, pool_m, sm, nf)
                   .mean((2, 4)).reshape(len(m), -1).astype(np.float32))
    return np.concatenate(out)


def make_files(src, frames, bands, patch, pool_f, pool_m, nf, W, bits):
    gf, gm, sf, sm = geometry(frames, bands, patch, pool_f, pool_m)
    Wq, b = W[:-1], W[-1]
    c, idx = quantize(Wq, bits)
    nc = len(c)
    payload = (c.astype(np.float16).tobytes() + b.astype(np.float16).tobytes()
               + pack_indices(idx, bits))
    check_roundtrip(payload, idx, bits, 2 * (nc + K))
    code = TEMPLATE.format(
        NC=nc, NCB=nc + K, UNPACK=unpack_expr(bits, Wq.size, 2 * (nc + K)),
        D=Wq.shape[0], K=K, FEAT=src, F=frames, M=bands, P=patch, PP=patch * patch,
        GF=gf, GM=gm, PF=pool_f, SF=sf, PM=pool_m, SM=sm, NF=nf)
    return {"predict.py": code.encode(), "w": payload}, nc


def predict_local(files, H, D1, bits, nc):
    B = files["w"]
    cb = np.frombuffer(B, np.float16, nc + K).astype(np.float32)
    ns = {"np": np, "B": B}
    i = eval(unpack_expr(bits, (D1 - 1) * K, 2 * (nc + K)), ns)   # noqa: S307
    W = cb[:nc][np.asarray(i)].reshape(D1 - 1, K)
    return np.argmax(H @ W + cb[nc:], 1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(REPO / "data" / "sc12.npz"))
    ap.add_argument("--out", default="/tmp/kws_conv")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--bands", type=int, default=16)
    ap.add_argument("--patch", type=int, default=3)
    ap.add_argument("--pool", nargs="*", type=int, default=[2, 7])
    ap.add_argument("--nf", nargs="*", type=int, default=[4, 8, 16])
    ap.add_argument("--bits", nargs="*", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--lam", nargs="*", type=float,
                    default=[1e0, 1e1, 1e2, 1e3, 1e4])
    a = ap.parse_args(argv)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(a.data)
    xtr, ytr, xva, yva, xte, yte = (z["xtrain"], z["ytrain"], z["xval"],
                                    z["yval"], z["xtest"], z["ytest"])

    src = feat_src("mel", a.frames, a.bands)
    f = compile_feat(src)
    Vtr, Vva = features(f, xtr), features(f, xva)
    print(f"mel F={a.frames} M={a.bands}; train {len(xtr):,} val {len(xva):,} "
          f"test {len(xte):,}; chance {100 / K:.2f}%")

    for pool in a.pool:
        if geometry(a.frames, a.bands, a.patch, pool, pool) is None:
            print(f"pool {pool}: does not tile, skipped")
            continue
        for nf in a.nf:
            t0 = time.perf_counter()
            Htr = conv_feats(Vtr, a.frames, a.bands, a.patch, pool, pool, nf)
            Hva = conv_feats(Vva, a.frames, a.bands, a.patch, pool, pool, nf)
            D = Htr.shape[1]
            print(f"\npool {pool}x{pool}, {nf} filters: D={D} "
                  f"({time.perf_counter() - t0:.0f}s)")
            heads = {lam: fit(Htr, ytr, lam) for lam in a.lam}
            for bits in a.bits:
                best = None
                for lam, W in heads.items():
                    files, nc = make_files(src, a.frames, a.bands, a.patch,
                                           pool, pool, nf, W, bits)
                    acc = float((predict_local(files, Hva, W.shape[0], bits, nc)
                                 == yva).mean())
                    if best is None or acc > best[0]:
                        best = (acc, lam, files)
                vacc, lam, files = best
                name = f"kwsc-F{a.frames}-M{a.bands}-p{pool}-k{nf}-{bits}b"
                d = out / name
                d.mkdir(exist_ok=True)
                A.write_dir(files, d)
                rec = evaluate_arrays(
                    d, xte, yte, K, name=name, split="test",
                    method=f"mel -> {nf} random {a.patch}x{a.patch} conv filters "
                           f"from a seed, {pool}x{pool} pool, {bits}-bit head",
                    notes=f"lambda={lam:g} on val; val {vacc * 100:.2f}%")
                rec.update(kind="conv", frames=a.frames, bands=a.bands, nf=nf,
                           pool=pool, bits=bits, dim=D, lam=lam,
                           val_accuracy=vacc, src_bytes=len(files["predict.py"]),
                           coef_bytes=len(files["w"]))
                (out / f"{name}.json").write_text(json.dumps(rec, indent=2))
                print(f"  {bits}b lam={lam:<7g} val {vacc * 100:5.2f}%  "
                      f"{summarize(rec)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
