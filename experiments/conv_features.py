"""Random convolutional features — the free seed, applied where images live.

Dense random features bought a nonlinearity for four bytes and took the frontier
to 47%. But a dense projection throws away the structure of the input: it treats
a 32x32 image as 3072 unordered numbers. Convolution does not, and a random
convolutional filter bank costs exactly the same four bytes.

    x -> [random 4x4 conv filters, from seed] -> relu(. - threshold)
      -> average pool -> ridge head            (only the head is shipped)

This is the Coates & Ng picture — a single layer of filters plus a linear
classifier, which reaches the 70s on CIFAR-10 when the filters are learned by
K-means. Learned filters are exactly what we cannot afford: the survey puts
Coates' filter bank at ~237 KB, which is twenty times our whole budget. Random
filters are weaker per filter but cost nothing, so the question this experiment
asks is whether the width we can then afford makes up the difference.

Both the trainer and the artifact run the *same* feature source string, so the
two cannot drift apart. That mattered enough to be worth the awkwardness: a
silent mismatch between training-time and inference-time features would show up
as a plausible-looking accuracy, not a crash.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from experiments.baselines import emit  # noqa: E402
from experiments.quant_sweep import build_codebook_blob, lloyd_max  # noqa: E402
from tinycifar import pack as P  # noqa: E402
from tinycifar.data import load  # noqa: E402
from tinycifar.evaluate import evaluate, summarize  # noqa: E402

CHUNK = 2000

# The feature extractor, shared verbatim between trainer and artifact.
# Defines feats(x) for float32 x in [0,1], shape [n,32,32,3].
FEATS_SRC = '''g=np.random.default_rng({SEED})
R=g.standard_normal(({K},{PP}),dtype=np.float32)*{S}
t=g.standard_normal({K},dtype=np.float32)*{TS}
def feats(x):
 n=len(x)
 w=np.lib.stride_tricks.sliding_window_view(x,({P},{P}),(1,2))[:,::{ST},::{ST}]
 c=w.transpose(0,1,2,4,5,3).reshape(n,{G}*{G},{PP})
 c=(c-c.mean(2,keepdims=True))/np.sqrt(c.var(2,keepdims=True)+.01)
 h=np.maximum(c@R.T-t,0).reshape(n,{G},{G},{K})
 return h.reshape(n,{PG},{PS},{PG},{PS},{K}).mean((2,4)).reshape(n,-1)
'''

TEMPLATE = '''import numpy as np,pathlib
B=(pathlib.Path(__file__).parent/"w").read_bytes()
{DECODE}{FEATS}def predict(x):
 x=x.astype(np.float32)/255
 o=np.empty(len(x),np.int64)
 for i in range(0,len(x),500):
  z=x[i:i+500];f=feats(z){TTA}
  o[i:i+500]=np.argmax(f@W[:-1]+W[-1],1)
 return o
'''

# Flip test-time augmentation. The head is linear, so averaging the two feature
# vectors is identical to averaging the two logit vectors — which makes this
# about twenty bytes of source rather than a second scoring pass.
TTA_SRC = "+feats(z[:,:,::-1])"

# Per-class codebooks. One shared codebook over the whole head degrades sharply
# as width grows — k=512 at 2 bits collapsed to 30.60% — because the ten class
# columns have quite different weight scales and a single Lloyd-max grid fits
# whichever dominates. Ten codebooks cost 10 * 2^b float16 centroids: 160 bytes
# at 3 bits, which is under 2% of a 10 KB budget.
PERCOL_DECODE = '''n=B[0];k=1<<n;D={D}
C=np.frombuffer(B,np.float16,10*k,1).astype(np.float32).reshape(10,k)
c=np.unpackbits(np.frombuffer(B[1+20*k:],np.uint8),bitorder="little")[:D*10*n].reshape(-1,n)
W=C[np.arange(10),(c.astype(np.uint32)<<np.arange(n,dtype=np.uint32)).sum(1).reshape(D,10)]
'''

GLOBAL_DECODE = '''n=B[0];k=1<<n;D={D}
C=np.frombuffer(B,np.float16,k,1).astype(np.float32)
c=np.unpackbits(np.frombuffer(B[1+2*k:],np.uint8),bitorder="little")[:D*10*n].reshape(-1,n)
W=C[(c.astype(np.uint32)<<np.arange(n,dtype=np.uint32)).sum(1)].reshape(D,10)
'''


def build_percol_blob(W: np.ndarray, bits: int) -> bytes:
    """One Lloyd-max codebook per class column."""
    import struct

    cents, idxs = [], []
    for j in range(W.shape[1]):
        c, i = lloyd_max(W[:, j], bits)
        cents.append(c)
        idxs.append(i)
    C = np.stack(cents)                       # (10, 2^bits) float16
    idx = np.stack(idxs, 1).reshape(-1)       # (D, 10) row-major
    return struct.pack("<B", bits) + C.tobytes() + P.bitpack(idx, bits)


def geometry(patch: int, stride: int, pool: int) -> tuple[int, int]:
    """Grid size after conv, and pooled grid size. Raises if they disagree."""
    g = (32 - patch) // stride + 1
    if g % pool:
        raise ValueError(f"pool {pool} does not divide conv grid {g}")
    return g, g // pool


def make_feats_src(k, patch, stride, pool, seed, scale, tscale) -> tuple[str, int]:
    g, pg = geometry(patch, stride, pool)
    src = FEATS_SRC.format(
        SEED=seed, K=k, PP=patch * patch * 3, S=repr(float(scale)),
        TS=repr(float(tscale)), P=patch, ST=stride, G=g, PG=pg, PS=pool,
    )
    return src, k * pg * pg


def build_feats(src: str):
    """Materialize feats() from the same string the artifact will run."""
    ns = {"np": np}
    exec(compile(src, "<feats>", "exec"), ns)  # noqa: S102
    return ns["feats"]


def fit_head(feats, x, y, dim, lam: float, flip_aug: bool = True, tta: bool = True):
    """Ridge on the pooled features, Gram matrix accumulated in chunks.

    `flip_aug` shows the head each image both ways during fitting; `tta` makes
    it fit the same averaged feature the artifact will present at inference.
    Both are training-time only and cost nothing in the artifact.
    """
    G = np.zeros((dim + 1, dim + 1))
    Hy = np.zeros((dim + 1, 10))

    def batch(z):
        f = feats(z)
        if tta:
            f = f + feats(z[:, :, ::-1])
        return np.hstack([f, np.ones((len(f), 1), np.float32)]).astype(np.float64)

    for i in range(0, len(x), CHUNK):
        z = x[i : i + CHUNK].astype(np.float32) / 255
        yy = y[i : i + CHUNK]
        views = [z, z[:, :, ::-1]] if (flip_aug and not tta) else [z]
        for v in views:
            f = batch(v)
            Y = np.zeros((len(f), 10))
            Y[np.arange(len(f)), yy] = 1.0
            G += f.T @ f
            Hy += f.T @ Y

    G.flat[:: dim + 2] += lam
    return np.linalg.solve(G, Hy)


def run(k, bits, patch, stride, pool, seed, lam, scale, tscale, data,
        tta: bool = True, quant: str = "percol") -> dict:
    xtr, ytr = data[0], data[1]
    src, dim = make_feats_src(k, patch, stride, pool, seed, scale, tscale)

    t0 = time.perf_counter()
    W = fit_head(build_feats(src), xtr, ytr, dim, lam, tta=tta)
    train_s = time.perf_counter() - t0

    if quant == "percol":
        blob = build_percol_blob(W, bits)
        decode = PERCOL_DECODE.format(D=dim + 1)
    else:
        blob = build_codebook_blob(W, bits)
        decode = GLOBAL_DECODE.format(D=dim + 1)

    name = f"cf-k{k}-p{patch}s{stride}-{bits}b" + ("" if tta else "-notta") \
        + ("-pc" if quant == "percol" else "")
    d = emit(name, {
        "predict.py": TEMPLATE.format(
            DECODE=decode, FEATS=src, TTA=TTA_SRC if tta else "").encode(),
        "w": blob,
    })

    r = evaluate(
        d, name=name,
        method=f"random {patch}x{patch} conv features (k={k}) + ridge head, "
               f"{bits}-bit {quant} codebook" + (", flip TTA" if tta else ""),
        notes=f"filters from 4-byte seed {seed}; {dim}-d pooled features; "
              f"{(dim + 1) * 10:,}-param head shipped; lambda={lam}",
        train_seconds=train_s,
    )
    print(f"  {summarize(r)}  ({train_s:.0f}s train)")
    return r


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", nargs="*", type=int, default=[64, 128])
    ap.add_argument("--bits", nargs="*", type=int, default=[4])
    ap.add_argument("--patch", type=int, default=4)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--pool", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--lam", type=float, default=1e2)
    ap.add_argument("--scale", type=float, default=0.0, help="0 -> 1/sqrt(patch*patch*3)")
    ap.add_argument("--tscale", type=float, default=0.1)
    ap.add_argument("--quant", default="percol", choices=["percol", "global"])
    ap.add_argument("--no-tta", action="store_true",
                    help="disable flip test-time augmentation (to measure its delta)")
    a = ap.parse_args(argv)

    data = load()
    scale = a.scale or 1.0 / np.sqrt(a.patch * a.patch * 3)
    g, pg = geometry(a.patch, a.stride, a.pool)
    print(f"conv grid {g}x{g} -> pooled {pg}x{pg}")

    results = []
    for k in a.k:
        print(f"\nk={k} filters -> {k * pg * pg}-d features")
        for bits in a.bits:
            try:
                results.append(run(k, bits, a.patch, a.stride, a.pool,
                                   a.seed, a.lam, scale, a.tscale, data,
                                   tta=not a.no_tta, quant=a.quant))
            except Exception as e:
                print(f"  ! cf-k{k}-{bits}b failed: {e}")

    if results:
        b = max(results, key=lambda r: r["accuracy"])
        print(f"\nbest {b['accuracy'] * 100:.2f}% ({b['name']}, "
              f"{b['description_length']:,} B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
