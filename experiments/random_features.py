"""Ship the seed, not the projection.

A linear model on pixels has saturated the frontier near 41%. The cheapest way
past it is a nonlinearity — but a random projection large enough to help would
cost far more bytes than the linear model it replaces.

Unless it is not shipped at all. `h = relu(f @ R + b)` with `R` and `b` drawn
from a named PRNG at load time costs **four bytes of seed**, whatever its size.
Only the ridge head on top is real weight. This is the procedural-weights idea
in its simplest honest form, and it needs no backprop: one pass to accumulate
the Gram matrix, one solve.

The seed is legitimate description length, not a loophole, on one condition —
the decoder has to be able to reproduce the draw exactly. numpy guarantees the
`default_rng`/PCG64 stream is stable across versions (NEP 19), and the harness
verifies it the only way that counts: `evaluate` re-runs the artifact from its
serialized bytes in a fresh process, so a projection that failed to reproduce
would show up as a collapsed accuracy rather than a silent pass.

Memory is the one real constraint here: the hidden layer for 50,000 images at
k=4096 would be 800 MB as one array, so the Gram matrix is accumulated in
chunks and the hidden layer is never materialized in full.
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

from experiments.baselines import FEATURES, emit, featurize  # noqa: E402
from experiments.quant_sweep import build_codebook_blob, lloyd_max  # noqa: E402
from tinycifar import pack as P  # noqa: E402
from tinycifar.data import load  # noqa: E402
from tinycifar.evaluate import evaluate, summarize  # noqa: E402

CHUNK = 5000

TEMPLATE = '''import numpy as np,pathlib
B=(pathlib.Path(__file__).parent/"w").read_bytes()
n=B[0];k=1<<n;D={D}
C=np.frombuffer(B,np.float16,k,1).astype(np.float32)
c=np.unpackbits(np.frombuffer(B[1+2*k:],np.uint8),bitorder="little")[:D*10*n].reshape(-1,n)
W=C[(c.astype(np.uint32)<<np.arange(n,dtype=np.uint32)).sum(1)].reshape(D,10)
g=np.random.default_rng({SEED})
R=(g.standard_normal(({IN},{K}),dtype=np.float32)*{S}).astype(np.float32)
b=g.standard_normal({K},dtype=np.float32)*{BS}
def predict(x):
 x=x.astype(np.float32)
 f={EXPR}
 o=np.empty(len(x),np.int64)
 for i in range(0,len(x),4096):
  h=np.maximum(f[i:i+4096]@R+b,0)
  o[i:i+4096]=np.argmax(h@W[:-1]+W[-1],1)
 return o
'''


def projection(in_dim: int, k: int, seed: int, s: float, bs: float):
    """Exactly the draw the artifact performs — kept in one place so the
    trainer and the artifact cannot drift apart."""
    g = np.random.default_rng(seed)
    R = (g.standard_normal((in_dim, k), dtype=np.float32) * s).astype(np.float32)
    b = g.standard_normal(k, dtype=np.float32) * bs
    return R, b


def fit_head(F, y, R, b, lam: float):
    """Ridge on the hidden layer, accumulating the Gram matrix in chunks."""
    k = R.shape[1]
    G = np.zeros((k + 1, k + 1))
    Hy = np.zeros((k + 1, 10))
    for i in range(0, len(F), CHUNK):
        h = np.maximum(F[i : i + CHUNK] @ R + b, 0)
        h = np.hstack([h, np.ones((len(h), 1), dtype=np.float32)]).astype(np.float64)
        Y = np.zeros((len(h), 10))
        Y[np.arange(len(h)), y[i : i + CHUNK]] = 1.0
        G += h.T @ h
        Hy += h.T @ Y
    G.flat[:: k + 2] += lam
    return np.linalg.solve(G, Hy)


def run(feat, k, bits, seed, lam, s, bs, data, scheme="codebook") -> dict:
    xtr, ytr = data[0], data[1]
    in_dim, expr = FEATURES[feat]

    t0 = time.perf_counter()
    F = featurize(xtr, expr)[:, :-1].astype(np.float32)   # drop the bias column
    R, b = projection(in_dim, k, seed, s, bs)
    W = fit_head(F, ytr, R, b, lam)
    train_s = time.perf_counter() - t0

    if scheme == "codebook":
        blob = build_codebook_blob(W, bits)
    else:
        codes, scale, zero = P.quantize(W, bits, symmetric=True)
        blob = struct.pack("<B", bits) + struct.pack("<ff", scale, zero) + \
            P.bitpack(codes, bits)

    src = TEMPLATE.format(D=k + 1, EXPR=expr, SEED=seed, IN=in_dim, K=k,
                          S=repr(float(s)), BS=repr(float(bs)))
    name = f"rf-{feat}-k{k}-{bits}b"
    d = emit(name, {"predict.py": src.encode(), "w": blob})

    r = evaluate(
        d, name=name,
        method=f"random ReLU features (k={k}) on {feat}, {bits}-bit {scheme}",
        notes=f"projection from 4-byte seed {seed}; only the {(k + 1) * 10:,}-param "
              f"head is shipped; lambda={lam}",
        train_seconds=train_s,
    )
    print(f"  {summarize(r)}")
    return r


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", nargs="*", default=["rgb8"])
    ap.add_argument("--k", nargs="*", type=int, default=[128, 256, 512, 1024])
    ap.add_argument("--bits", nargs="*", type=int, default=[3, 4, 6])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--lam", type=float, default=1e2)
    ap.add_argument("--scale", type=float, default=0.0,
                    help="0 selects 1/sqrt(in_dim)")
    ap.add_argument("--bias-scale", type=float, default=0.1)
    a = ap.parse_args(argv)

    data = load()
    results = []
    for feat in a.features:
        in_dim = FEATURES[feat][0]
        s = a.scale or 1.0 / np.sqrt(in_dim)
        for k in a.k:
            print(f"\n{feat} ({in_dim}d) -> k={k} random ReLU features")
            for bits in a.bits:
                try:
                    results.append(
                        run(feat, k, bits, a.seed, a.lam, s, a.bias_scale, data)
                    )
                except Exception as e:
                    print(f"  ! rf-{feat}-k{k}-{bits}b failed: {e}")

    if results:
        best = max(results, key=lambda r: r["accuracy"])
        print(f"\nbest {best['accuracy'] * 100:.2f}% "
              f"({best['name']}, {best['description_length']:,} B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
