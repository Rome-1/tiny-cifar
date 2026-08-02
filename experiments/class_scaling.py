"""How does description length scale with the number of classes?

CIFAR-10 cannot ask this question: it has one class count. A linear head costs
`classes x features x bits`, so the arithmetic says the sub-kilobyte region is
closed at 1000 classes and the whole frontier shifts right — but arithmetic is
not measurement, and this repo has twice watched an architecture that surveys
recommended die on contact with the harness.

The measurement holds everything fixed except the class count: the same images,
the same random-convolutional-feature family the sub-4 KB frontier is built on,
the same ridge solve, the same quantizer, the same harness. Only K varies.

Two things differ from `experiments/conv_features.py`, and both are forced:

* The head is K-wide, not 10-wide, so the decoder is emitted with K baked in.
* The codebook is global, not per-column. Per-column costs `K * 2^b` float16
  centroids — 160 B at ten classes and 32 KB at a thousand, which would make
  the quantizer, not the head, the thing the curve measures.

Configuration is chosen on a validation split carved out of train. Test is
scored once, at the end, on the configuration val picked.

Run:
    python3 experiments/class_scaling.py --data data/in32.npz --classes 10 100
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

from experiments.conv_features import (  # noqa: E402
    build_feats, geometry, make_feats_src,
)
from tinycifar import pack as P  # noqa: E402
from tinycifar.evaluate_nclass import evaluate_arrays, summarize  # noqa: E402

ARTIFACTS = REPO / "artifacts" / "class-scaling"
CHUNK = 2000

# Global Lloyd-max codebook, K baked in. `conv_features.GLOBAL_DECODE` is the
# same code with 10 hardcoded; it is re-stated rather than imported because
# parameterizing the CIFAR path's decoder would change bytes on the board.
GLOBAL_DECODE = '''n=B[0];k=1<<n;D={D};K={K}
C=np.frombuffer(B,np.float16,k,1).astype(np.float32)
c=np.unpackbits(np.frombuffer(B[1+2*k:],np.uint8),bitorder="little")[:D*K*n].reshape(-1,n)
W=C[(c.astype(np.uint32)<<np.arange(n,dtype=np.uint32)).sum(1)].reshape(D,K)
'''

TEMPLATE = '''import numpy as np,pathlib
B=(pathlib.Path(__file__).parent/"w").read_bytes()
{DECODE}{FEATS}def predict(x):
 x=x.astype(np.float32)/255
 o=np.empty(len(x),np.int64)
 for i in range(0,len(x),500):
  z=x[i:i+500];f=feats(z)+feats(z[:,:,::-1])
  o[i:i+500]=np.argmax(f@W[:-1]+W[-1],1)
 return o
'''


def lloyd_max_big(w: np.ndarray, bits: int, iters: int = 30,
                  sample: int = 400_000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Lloyd-max on a subsample, then assign every weight.

    `quant_sweep.lloyd_max` runs on the full vector; at 1000 classes that vector
    is ~15M long and the (n, 2^b) distance matrix is the peak memory of the
    whole script. Fitting the centroids on a 400k subsample and assigning in
    chunks is the same estimator to well inside any effect this measures.
    """
    w = w.astype(np.float64).reshape(-1)
    g = np.random.default_rng(seed)
    s = w if len(w) <= sample else w[g.choice(len(w), sample, replace=False)]
    k = 1 << bits
    qs = np.linspace(0, 100, k + 2)[1:-1]
    c = np.percentile(s, qs)
    for _ in range(iters):
        idx = np.abs(s[:, None] - c[None, :]).argmin(1)
        for j in range(k):
            m = idx == j
            if m.any():
                c[j] = s[m].mean()
        c.sort()
    c = c.astype(np.float16).astype(np.float64)     # centroids ship as float16
    out = np.empty(len(w), np.int64)
    for i in range(0, len(w), 1_000_000):
        out[i:i + 1_000_000] = np.abs(
            w[i:i + 1_000_000, None] - c[None, :]).argmin(1)
    return c.astype(np.float16), out


def build_global_blob(W: np.ndarray, bits: int) -> bytes:
    c, idx = lloyd_max_big(W, bits)
    return struct.pack("<B", bits) + c.tobytes() + P.bitpack(idx, bits)


def fit_head(feats, x, y, dim: int, n_classes: int, lams) -> dict:
    """Ridge on K one-hot targets; one Gram pass serves every lambda.

    The Gram matrix does not depend on lambda, so the whole regularization
    sweep costs one featurization pass plus one (dim+1)^2 solve per lambda.
    That is the entire reason this family is affordable at 1000 classes on a
    CPU: cost is O(N*dim + dim^2*K), independent of how many lambdas we try.
    """
    G = np.zeros((dim + 1, dim + 1))
    Hy = np.zeros((dim + 1, n_classes))
    for i in range(0, len(x), CHUNK):
        z = x[i:i + CHUNK].astype(np.float32) / 255
        yy = y[i:i + CHUNK]
        f = feats(z) + feats(z[:, :, ::-1])          # matches artifact's TTA
        f = np.hstack([f, np.ones((len(f), 1), np.float32)]).astype(np.float64)
        Y = np.zeros((len(f), n_classes))
        Y[np.arange(len(f)), yy] = 1.0
        G += f.T @ f
        Hy += f.T @ Y
    out = {}
    for lam in lams:
        A = G.copy()
        A.flat[::dim + 2] += lam
        out[lam] = np.linalg.solve(A, Hy)
    return out


def predict_float(feats, W, x) -> np.ndarray:
    """Unquantized accuracy — used only for validation-set selection."""
    o = np.empty(len(x), np.int64)
    for i in range(0, len(x), CHUNK):
        z = x[i:i + CHUNK].astype(np.float32) / 255
        f = feats(z) + feats(z[:, :, ::-1])
        o[i:i + CHUNK] = np.argmax(f @ W[:-1] + W[-1], 1)
    return o


def emit(name: str, files: dict[str, bytes]) -> Path:
    d = ARTIFACTS / name
    d.mkdir(parents=True, exist_ok=True)
    for k, v in files.items():
        (d / k).write_bytes(v)
    return d


def build_artifact(name, src, W, bits, dim, n_classes) -> Path:
    return emit(name, {
        "predict.py": TEMPLATE.format(
            DECODE=GLOBAL_DECODE.format(D=dim + 1, K=n_classes),
            FEATS=src).encode(),
        "w": build_global_blob(W, bits),
    })


def subset(x, y, n_classes: int):
    """Classes [0, n_classes) — nested, so 10 is a subset of 100 is a subset
    of 1000, and the images at every K are literally the same images."""
    m = y < n_classes
    return x[m], y[m]


def run(data, n_classes: int, ks, bitss, lams, patch, stride, pool, seed,
        tscale, out_dir: Path) -> list[dict]:
    xfit, yfit = subset(data["xtr"], data["ytr"], n_classes)
    xval, yval = subset(data["xva"], data["yva"], n_classes)
    xte, yte = subset(data["xte"], data["yte"], n_classes)
    print(f"\n=== K={n_classes}: fit {len(xfit):,} / val {len(xval):,} "
          f"/ test {len(xte):,} images   chance {100/n_classes:.2f}%")

    scale = 1.0 / np.sqrt(patch * patch * 3)
    rows = []
    for k in ks:
        src, dim = make_feats_src(k, patch, stride, pool, seed, scale, tscale)
        feats = build_feats(src)
        t0 = time.perf_counter()
        Ws = fit_head(feats, xfit, yfit, dim, n_classes, lams)
        train_s = time.perf_counter() - t0
        for lam, W in Ws.items():
            va = float((predict_float(feats, W, xval) == yval).mean())
            print(f"  k={k:<4} dim={dim:<5} lam={lam:<8g} val(float)={va*100:6.2f}%"
                  f"  ({train_s:.0f}s fit)")
            rows.append({"k": k, "dim": dim, "lam": lam, "val_float": va,
                         "train_seconds": train_s, "src": src, "W": W})

    # (k, bits) is the *size* axis and is declared, not selected: every point on
    # it gets scored on test, so the curve is a measurement rather than a
    # best-of. Only lambda is chosen, and it is chosen on val, through the real
    # quantized artifact — float-val ranking disagrees with quantized-val
    # ranking badly at low bit widths, so selecting on the float head would pick
    # the wrong lambda for exactly the points that matter most.
    results = []
    for bits in bitss:
        for kf in ks:
            cands = [r for r in rows if r["k"] == kf]
            best, best_rec = None, None
            tmp = f"tmp-K{n_classes}-k{kf}-{bits}b"
            for r in cands:
                d = build_artifact(tmp, r["src"], r["W"], bits,
                                   r["dim"], n_classes)
                rec = evaluate_arrays(d, xval, yval, n_classes, split="val",
                                      name=tmp)
                print(f"    [val] K={n_classes} k={kf} lam={r['lam']:g} "
                      f"{bits}b: {rec['accuracy'] * 100:6.2f}%  "
                      f"{rec['description_length']:,} B")
                if best is None or rec["accuracy"] > best["accuracy"]:
                    best, best_rec = rec, r

            name = f"cs-K{n_classes}-k{kf}-{bits}b"
            d = build_artifact(name, best_rec["src"], best_rec["W"], bits,
                               best_rec["dim"], n_classes)
            te = evaluate_arrays(
                d, xte, yte, n_classes, name=name, split="test",
                method=f"random {patch}x{patch} conv features (k={kf}) + ridge "
                       f"head, {bits}-bit global codebook, flip TTA",
                notes=f"K={n_classes}; lambda={best_rec['lam']}; "
                      f"{best_rec['dim']}-d features; "
                      f"{(best_rec['dim'] + 1) * n_classes:,}-param head; "
                      f"lambda selected on val (val acc {best['accuracy']:.4f})",
                train_seconds=best_rec["train_seconds"])
            te.update(val_accuracy=best["accuracy"], k=kf,
                      lam=best_rec["lam"], bits=bits,
                      dim=best_rec["dim"], fit_n=len(xfit))
            print(f"  -> TEST {summarize(te)}")
            results.append(te)
            (out_dir / f"{name}.json").write_text(json.dumps(te, indent=2))
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="npz with xtr,ytr,xva,yva,xte,yte")
    ap.add_argument("--classes", nargs="*", type=int, default=[10, 100, 1000])
    ap.add_argument("--k", nargs="*", type=int, default=[8, 16, 32, 64])
    ap.add_argument("--bits", nargs="*", type=int, default=[2, 3, 4])
    ap.add_argument("--lam", nargs="*", type=float, default=[1e1, 1e2, 1e3, 1e4])
    ap.add_argument("--patch", type=int, default=4)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--pool", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tscale", type=float, default=0.1)
    ap.add_argument("--out", default="results/class-scaling")
    a = ap.parse_args(argv)

    out_dir = REPO / a.out
    out_dir.mkdir(parents=True, exist_ok=True)
    z = np.load(a.data)
    data = {k: z[k] for k in ("xtr", "ytr", "xva", "yva", "xte", "yte")}
    g, pg = geometry(a.patch, a.stride, a.pool)
    print(f"conv grid {g}x{g} -> pooled {pg}x{pg}")

    allr = []
    for nc in a.classes:
        allr += run(data, nc, a.k, a.bits, a.lam, a.patch, a.stride, a.pool,
                    a.seed, a.tscale, out_dir)

    print("\n" + "=" * 78)
    print(f"{'K':>6}{'k':>5}{'bits':>6}{'bytes':>10}{'test':>9}{'chance':>9}"
          f"{'lift':>9}{'B/class':>10}")
    for r in allr:
        print(f"{r['n_classes']:>6}{r['k']:>5}{r['bits']:>6}"
              f"{r['description_length']:>10,}{r['accuracy']*100:>8.2f}%"
              f"{r['chance']*100:>8.2f}%{r['accuracy']/r['chance']:>8.1f}x"
              f"{r['description_length']/r['n_classes']:>10.1f}")
    (out_dir / "summary.json").write_text(json.dumps(allr, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
