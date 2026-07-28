"""Trivial, near-zero-compute baselines that seed the Pareto frontier.

Every model here is a linear map fitted in closed form (ridge regression on
one-hot targets), so the whole sweep runs on CPU in well under a minute. The
point is not accuracy — it is to plant honest points across four orders of
magnitude of size so later methods have something to beat.

Two things this sweep is designed to demonstrate:

1. **Input reduction is free.** Downsampling and graying are computed by a line
   of numpy in the artifact, so they cost ~zero bytes while cutting the weight
   matrix by up to 192x. At the small end this dominates every other lever.
2. **Code is not free.** At a 300-byte artifact the decoder source is the
   largest single line item, so the generated `predict.py` is written tersely.
   That is the honest MDL accounting: bits hidden in code are still bits.
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

from tinycifar import pack as P  # noqa: E402
from tinycifar.data import load  # noqa: E402
from tinycifar.evaluate import evaluate, summarize  # noqa: E402

ARTIFACTS = REPO / "artifacts"

# ---------------------------------------------------------------------------
# feature transforms — each is a (dimension, one-line numpy expression) pair.
# The expression is baked into the artifact verbatim; `x` is uint8 [N,32,32,3].
# ---------------------------------------------------------------------------

def _blockmean(k: int, gray: bool) -> str:
    """Expression that block-mean-pools 32x32 to k x k, optionally graying."""
    s = 32 // k
    base = "x.mean(3)" if gray else "x"
    shape = f"(-1,{k},{s},{k},{s})" if gray else f"(-1,{k},{s},{k},{s},3)"
    return f"{base}.reshape({shape}).mean((2,4)).reshape(len(x),-1)/255"


FEATURES = {
    #  name        dim   expression
    "gray4":  (16,   _blockmean(4, True)),
    "gray8":  (64,   _blockmean(8, True)),
    "rgb4":   (48,   _blockmean(4, False)),
    "rgb8":   (192,  _blockmean(8, False)),
    "gray16": (256,  _blockmean(16, True)),
    "rgb16":  (768,  _blockmean(16, False)),
    "raw":    (3072, "x.reshape(len(x),-1)/255"),
}

# ---------------------------------------------------------------------------
# the artifact template — deliberately terse; every byte here is on the bill
# ---------------------------------------------------------------------------

TEMPLATE = '''import numpy as np,pathlib,struct
B=(pathlib.Path(__file__).parent/"w").read_bytes()
n=B[0];s,z=struct.unpack_from("<ff",B,1)
D={D}
c=np.unpackbits(np.frombuffer(B[9:],np.uint8),bitorder="little")[:D*10*n].reshape(-1,n)
W=((c.astype(np.uint32)<<np.arange(n,dtype=np.uint32)).sum(1)*s+z).reshape(D,10)
def predict(x):
 x=x.astype(np.float32)
 f={EXPR}
 return np.argmax(f@W[:-1]+W[-1],1)
'''

CONSTANT_TEMPLATE = '''import numpy as np
def predict(x):
 return np.full(len(x),{c},np.int64)
'''


def build_weight_blob(W: np.ndarray, bits: int) -> bytes:
    """Header (bits, scale, zero) + densely packed codes."""
    codes, scale, zero = P.quantize(W, bits, symmetric=False)
    return struct.pack("<B", bits) + struct.pack("<ff", scale, zero) + P.bitpack(
        codes, bits
    )


def fit_ridge(F: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Closed-form ridge on one-hot targets. F already has its bias column."""
    Y = np.zeros((len(y), 10), dtype=np.float64)
    Y[np.arange(len(y)), y] = 1.0
    A = F.T @ F
    A.flat[:: A.shape[0] + 1] += lam
    return np.linalg.solve(A, F.T @ Y)


def featurize(x: np.ndarray, expr: str) -> np.ndarray:
    """Run a feature expression exactly as the artifact will run it."""
    f = eval(expr, {"np": np}, {"x": x.astype(np.float32)})  # noqa: S307
    return np.hstack([f, np.ones((len(f), 1), dtype=f.dtype)]).astype(np.float64)


def emit(name: str, files: dict[str, bytes]) -> Path:
    d = ARTIFACTS / name
    d.mkdir(parents=True, exist_ok=True)
    for k, v in files.items():
        (d / k).write_bytes(v)
    return d


def run_constant(ytr: np.ndarray) -> dict:
    c = int(np.bincount(ytr).argmax())
    d = emit("constant", {"predict.py": CONSTANT_TEMPLATE.format(c=c).encode()})
    r = evaluate(d, name="constant", method="majority class",
                 notes="the floor: the smallest artifact that runs at all",
                 train_seconds=0.0)
    print(" ", summarize(r))
    return r


def run_linear(feat: str, bits: int, lam: float, data, val_frac=0.0) -> dict:
    xtr, ytr, xte, yte = data
    D, expr = FEATURES[feat]

    t0 = time.perf_counter()
    F = featurize(xtr, expr)
    W = fit_ridge(F, ytr, lam)
    train_s = time.perf_counter() - t0

    blob = build_weight_blob(W, bits)
    src = TEMPLATE.format(D=D + 1, EXPR=expr)
    name = f"linear-{feat}-{bits}b"
    d = emit(name, {"predict.py": src.encode(), "w": blob})

    r = evaluate(
        d, name=name,
        method=f"ridge linear on {feat} ({D}d), {bits}-bit weights",
        notes=f"lambda={lam}; {(D + 1) * 10:,} params",
        train_seconds=train_s,
    )
    print(" ", summarize(r))
    return r


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", nargs="*", default=list(FEATURES))
    ap.add_argument("--bits", nargs="*", type=int,
                    default=[1, 2, 3, 4, 6, 8])
    ap.add_argument("--lam", type=float, default=1e2)
    a = ap.parse_args(argv)

    print("loading CIFAR-10 ...")
    data = load()
    print(f"train {data[0].shape}  test {data[2].shape}\n")

    results = [run_constant(data[1])]
    for feat in a.features:
        for bits in a.bits:
            try:
                results.append(run_linear(feat, bits, a.lam, data))
            except Exception as e:                      # keep the sweep going
                print(f"  ! linear-{feat}-{bits}b failed: {e}")

    best = max(results, key=lambda r: r["accuracy"])
    print(f"\n{len(results)} artifacts. best accuracy "
          f"{best['accuracy'] * 100:.2f}% ({best['name']}, "
          f"{best['description_length']:,} B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
