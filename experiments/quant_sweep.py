"""How should a fixed set of weights be spent in n bits?

The baseline sweep exposed a real failure: at 1-2 bits the affine min/max grid
collapses to near chance, because ridge weights have heavy tails and the grid
spends almost all of its levels covering outliers that carry little of the
decision. This compares three schemes on identical fitted weights, so the only
variable is the coding.

    affine     — uniform grid over [min, max]            (the baseline)
    symmetric  — uniform grid centered on zero
    codebook   — Lloyd-max k-means over the weight values

The codebook is the interesting one on two counts. It puts its levels where the
mass actually is rather than where the extremes are, and its decoder is *shorter
source* than affine dequantization — a table lookup instead of an arithmetic
rescale. At these sizes that second point is not a rounding error.
"""

from __future__ import annotations

import argparse
import struct
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from experiments.baselines import (  # noqa: E402
    FEATURES, TEMPLATE, emit, featurize, fit_ridge,
)
from tinycifar import pack as P  # noqa: E402
from tinycifar.data import load  # noqa: E402
from tinycifar.evaluate import evaluate, summarize  # noqa: E402

# Codebook artifact: bits, k float16 centroids, then packed indices.
CODEBOOK_TEMPLATE = '''import numpy as np,pathlib
B=(pathlib.Path(__file__).parent/"w").read_bytes()
n=B[0];k=1<<n;D={D}
C=np.frombuffer(B,np.float16,k,1).astype(np.float32)
c=np.unpackbits(np.frombuffer(B[1+2*k:],np.uint8),bitorder="little")[:D*10*n].reshape(-1,n)
W=C[(c.astype(np.uint32)<<np.arange(n,dtype=np.uint32)).sum(1)].reshape(D,10)
def predict(x):
 x=x.astype(np.float32)
 f={EXPR}
 return np.argmax(f@W[:-1]+W[-1],1)
'''


def lloyd_max(w: np.ndarray, bits: int, iters: int = 60, seed: int = 0):
    """1-D k-means on the weight values. Returns (centroids, indices).

    Initialized on quantiles rather than at random: the weight distribution is
    unimodal and heavy-tailed, so quantile init lands close to the optimum and
    makes the result deterministic.
    """
    k = 1 << bits
    v = np.asarray(w, dtype=np.float64).reshape(-1)
    c = np.quantile(v, (np.arange(k) + 0.5) / k)
    c = np.unique(c)
    if len(c) < k:  # degenerate (e.g. near-constant weights) — pad the grid
        lo, hi = v.min(), v.max()
        c = np.linspace(lo, hi if hi > lo else lo + 1e-6, k)

    for _ in range(iters):
        idx = np.abs(v[:, None] - c[None, :]).argmin(1)
        new = c.copy()
        for j in range(k):
            m = idx == j
            if m.any():
                new[j] = v[m].mean()
        if np.allclose(new, c):
            break
        c = new

    c = c.astype(np.float16).astype(np.float64)      # centroids ship as float16
    idx = np.abs(v[:, None] - c[None, :]).argmin(1)
    return c.astype(np.float16), idx.astype(np.uint16)


def build_codebook_blob(W: np.ndarray, bits: int) -> bytes:
    c, idx = lloyd_max(W, bits)
    return struct.pack("<B", bits) + c.tobytes() + P.bitpack(idx, bits)


def relative_error(W: np.ndarray, approx: np.ndarray) -> float:
    return float(np.linalg.norm(approx - W) / (np.linalg.norm(W) or 1.0))


def run(feat: str, bits: int, scheme: str, W: np.ndarray, train_s: float) -> dict:
    D, expr = FEATURES[feat]

    if scheme == "codebook":
        blob = build_codebook_blob(W, bits)
        src = CODEBOOK_TEMPLATE.format(D=D + 1, EXPR=expr)
        c, idx = lloyd_max(W, bits)
        err = relative_error(W, c.astype(np.float32)[idx].reshape(W.shape))
    else:
        sym = scheme == "symmetric"
        codes, scale, zero = P.quantize(W, bits, symmetric=sym)
        blob = struct.pack("<B", bits) + struct.pack("<ff", scale, zero) + \
            P.bitpack(codes, bits)
        src = TEMPLATE.format(D=D + 1, EXPR=expr)
        err = relative_error(W, codes.astype(np.float64) * scale + zero)

    name = f"lin-{feat}-{bits}b-{scheme}"
    d = emit(name, {"predict.py": src.encode(), "w": blob})
    r = evaluate(
        d, name=name,
        method=f"ridge linear on {feat} ({D}d), {bits}-bit {scheme}",
        notes=f"weight reconstruction rel. error {err:.4f}",
        train_seconds=train_s,
    )
    r["quant_error"] = err
    print(f"  {summarize(r)}   relerr {err:.4f}")
    return r


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", nargs="*", default=["gray4", "rgb4", "rgb8", "rgb16"])
    ap.add_argument("--bits", nargs="*", type=int, default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--schemes", nargs="*",
                    default=["affine", "symmetric", "codebook"])
    ap.add_argument("--lam", type=float, default=1e2)
    a = ap.parse_args(argv)

    data = load()
    xtr, ytr = data[0], data[1]
    results = []

    for feat in a.features:
        D, expr = FEATURES[feat]
        t0 = time.perf_counter()
        W = fit_ridge(featurize(xtr, expr), ytr, a.lam)
        train_s = time.perf_counter() - t0
        print(f"\n{feat} ({D}d, {(D + 1) * 10:,} params, fit in {train_s:.1f}s)")

        for bits in a.bits:
            for scheme in a.schemes:
                try:
                    results.append(run(feat, bits, scheme, W, train_s))
                except Exception as e:
                    print(f"  ! {feat}-{bits}b-{scheme} failed: {e}")

    print("\n--- best per (feature, bits) ---")
    for feat in a.features:
        for bits in a.bits:
            cand = [r for r in results
                    if r["name"].startswith(f"lin-{feat}-{bits}b-")]
            if cand:
                b = max(cand, key=lambda r: r["accuracy"])
                gap = b["accuracy"] - min(c["accuracy"] for c in cand)
                print(f"  {feat:>6} {bits}b: {b['name'].split('-')[-1]:>9} "
                      f"{b['accuracy'] * 100:5.2f}%  (+{gap * 100:.2f} over worst)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
