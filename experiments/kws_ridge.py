"""Closed-form ridge over a generated audio front end, at sub-kilobyte sizes.

This is step 2 of the Speech Commands de-risk: given that a log-mel front end
costs about a hundred source bytes and stores nothing
(`experiments/kws_extractors.py`), does anything classify?

The machinery is the repo's own sub-4 KB family, unchanged in kind: features
that cost only source, a ridge head solved in closed form, a Lloyd-max codebook,
and the whole thing shipped as a numpy-only artifact scored through the harness.
Only the front end is new.

**The trainer and the artifact share one source string.** The feature expression
below is `exec`'d by the trainer and pasted verbatim into `predict.py`, so the
two cannot drift -- there is no second implementation to keep in sync. This is
the failure mode that would silently inflate every number here, and it is
designed out rather than tested for.

Selection, stated plainly:

  * Split membership is the dataset's own `validation_list.txt` /
    `testing_list.txt` (see `kws_prep.py`). Nothing here re-splits.
  * `(frames, bands, bits)` is the **size axis** and is *declared*: every point
    on it is scored on test, so the table is a measurement and not a best-of-N.
  * Only the ridge penalty is chosen, and it is chosen on validation, through
    the real quantized artifact -- picking it on the float head selects the
    wrong penalty at low bit widths.

Run:
    python3 experiments/kws_ridge.py --out /tmp/kws_ridge
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from experiments.golf import unpack_expr  # noqa: E402
from experiments.quant_sweep import lloyd_max  # noqa: E402
from tinycifar import artifact as A  # noqa: E402
from tinycifar.evaluate_nclass import evaluate_arrays, summarize  # noqa: E402

K = 12          # ten words + _silence_ + _unknown_
SR = 16000


# --------------------------------------------------------------------------
# the front end, as source
# --------------------------------------------------------------------------

def feat_src(kind: str, frames: int, bands: int) -> str:
    """The feature expression, binding `v`. This exact string ships."""
    L = SR // frames
    # A frame count that does not divide 16,000 costs one slice; 16, 32 and 40
    # divide it and pay nothing.
    xs = "x" if frames * L == SR else f"x[:,:{frames * L}]"
    if kind == "env":
        return f"v=np.log1p(abs({xs}*1.).reshape(len(x),{frames},{L}).mean(2))"
    spec = f"abs(np.fft.rfft({xs}.reshape(len(x),{frames},{L})*1.))"
    if kind == "oct":
        # Octave-spaced edges over the L/2+1 rfft bins: two constants.
        e = f"(2**np.linspace(0,{math.log2(L / 2):.3f},{bands})).astype(int)"
    elif kind == "mel":
        # bin = (700*L/16000) * (10**(mel/2595) - 1), mel swept 0..2840.
        c = round(700 * L / SR, 1)
        e = f"({c}*10**np.linspace(0,1.094,{bands})-{c}).astype(int)"
    else:
        raise ValueError(kind)
    return f"v=np.log1p(np.add.reduceat({spec},{e},2)).reshape(len(x),-1)"


def compile_feat(src: str):
    """The trainer's feature function is the artifact's, compiled from one string."""
    ns = {"np": np}
    exec(f"def f(x,np=np):\n {src}\n return v", ns)   # noqa: S102
    return ns["f"]


def features(f, x: np.ndarray, chunk: int = 2000) -> np.ndarray:
    return np.concatenate(
        [f(x[i:i + chunk]).astype(np.float32) for i in range(0, len(x), chunk)])


# --------------------------------------------------------------------------
# the head
# --------------------------------------------------------------------------

# Two decoders, because where the bias row goes turned out to matter more than
# anything else in this file.
#
# `TEMPLATE` quantizes the bias with the weights, which is what the CIFAR
# family does. Here it fails: standardization is folded into the shipped weights
# so that no per-feature mean or scale has to be transmitted, and the residue of
# that fold lands entirely in the bias row, which comes out one to two orders of
# magnitude larger than any weight. A global Lloyd-max codebook then spends its
# centroids bracketing twelve outliers and quantizes the other thousands to
# nearly nothing -- which is exactly the non-monotonic-in-bit-width signature
# the first sweep produced.
#
# `TEMPLATE_SB` ships the twelve biases as float16 instead, 24 bytes, and lets
# the codebook see only the weight matrix. Both are measured; see kws-derisk.md.
TEMPLATE = '''import numpy as np
B=open(__file__[:-10]+"w","rb").read()
C=np.frombuffer(B,np.float16,{NC}).astype(np.float32)
W=C[{UNPACK}].reshape({D1},{K})
def predict(x):
 {FEAT}
 return np.argmax(v@W[:-1]+W[-1],1)
'''

TEMPLATE_SB = '''import numpy as np
B=open(__file__[:-10]+"w","rb").read()
C=np.frombuffer(B,np.float16,{NCB}).astype(np.float32)
W=C[{UNPACK}].reshape({D},{K})
def predict(x):
 {FEAT}
 return np.argmax(v@W+C[{NC}:],1)
'''


def fit(V: np.ndarray, y: np.ndarray, lam: float):
    """Ridge onto one-hot targets, with standardization folded into the weights
    so no per-feature mean or scale has to ship."""
    mu, sd = V.mean(0), V.std(0) + 1e-6
    Z = (V - mu) / sd
    Z = np.c_[Z, np.ones(len(Z), dtype=np.float32)]
    T = np.eye(K, dtype=np.float32)[y]
    G = Z.T @ Z
    G[np.diag_indices_from(G)] += lam
    W = np.linalg.solve(G, Z.T @ T)          # [D+1, K] in standardized space
    Wf = W[:-1] / sd[:, None]                # fold the scale in
    bf = W[-1] - mu @ Wf                     # and the mean
    return np.vstack([Wf, bf]).astype(np.float32)


def quantize(W: np.ndarray, bits: int):
    c, idx = lloyd_max(W, bits)
    return c, idx.reshape(W.shape)


def pack_indices(idx: np.ndarray, bits: int) -> bytes:
    """Contiguous little-endian bit packing, matching `golf.unpack_expr`.

    The first version of this packed `8 // bits` codes per byte, which is right
    for the widths that divide 8 and silently wrong for the ones that do not:
    at 3 bits it wasted two bits per byte and produced a layout the decoder does
    not read. Every 3-bit artifact in the first sweep therefore scored at chance
    -- 8.10% against 8.33% -- and looked like a quantizer failure rather than a
    packing failure. `check_roundtrip` below is the gate that would have caught
    it, and it now runs on every artifact this file emits.
    """
    flat = idx.reshape(-1).astype(np.uint8)
    if bits == 8:
        return flat.tobytes()
    b = ((flat[:, None] >> np.arange(bits, dtype=np.uint8)) & 1).reshape(-1)
    return np.packbits(b, bitorder="little").tobytes()


def check_roundtrip(payload: bytes, idx: np.ndarray, bits: int, offset: int):
    """Unpack the shipped bytes with the shipped expression and demand the
    original indices back. A codelength that does not round-trip is not a
    codelength, and a weight that does not round-trip is not a weight."""
    n = idx.size
    ns = {"np": np, "B": payload}
    back = np.asarray(eval(unpack_expr(bits, n, offset), ns)).reshape(-1)[:n]
    if not np.array_equal(back, idx.reshape(-1)):
        raise RuntimeError(
            f"{bits}-bit payload does not round-trip through the decoder "
            f"({int((back != idx.reshape(-1)).sum())} of {n} codes differ)")


def make_files(src: str, W: np.ndarray, bits: int, split_bias: bool):
    if not split_bias:
        c, idx = quantize(W, bits)
        payload = c.astype(np.float16).tobytes() + pack_indices(idx, bits)
        check_roundtrip(payload, idx, bits, 2 * len(c))
        code = TEMPLATE.format(
            NC=len(c), UNPACK=unpack_expr(bits, W.size, 2 * len(c)),
            D1=W.shape[0], K=K, FEAT=src)
        return {"predict.py": code.encode(), "w": payload}, len(c)

    Wq, b = W[:-1], W[-1]
    c, idx = quantize(Wq, bits)
    nc = len(c)
    payload = (c.astype(np.float16).tobytes() + b.astype(np.float16).tobytes()
               + pack_indices(idx, bits))
    check_roundtrip(payload, idx, bits, 2 * (nc + K))
    code = TEMPLATE_SB.format(
        NC=nc, NCB=nc + K, UNPACK=unpack_expr(bits, Wq.size, 2 * (nc + K)),
        D=Wq.shape[0], K=K, FEAT=src)
    return {"predict.py": code.encode(), "w": payload}, nc


def predict_local(files, v, D1, bits, n_c, split_bias: bool):
    """Mirror of the emitted decoder, for sweeping without a subprocess.

    It reads the *packed weight file*, not the float head, so the validation
    ranking is the ranking of the artifact that would actually ship.
    """
    B = files["w"]
    if not split_bias:
        c = np.frombuffer(B, np.float16, n_c).astype(np.float32)
        ns = {"np": np, "B": B}
        i = eval(unpack_expr(bits, D1 * K, 2 * n_c), ns)          # noqa: S307
        W = c[np.asarray(i)].reshape(D1, K)
        return np.argmax(v @ W[:-1] + W[-1], 1)
    cb = np.frombuffer(B, np.float16, n_c + K).astype(np.float32)
    ns = {"np": np, "B": B}
    i = eval(unpack_expr(bits, (D1 - 1) * K, 2 * (n_c + K)), ns)  # noqa: S307
    W = cb[: n_c][np.asarray(i)].reshape(D1 - 1, K)
    return np.argmax(v @ W + cb[n_c:], 1)


# --------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(REPO / "data" / "sc12.npz"))
    ap.add_argument("--out", default="/tmp/kws_ridge")
    ap.add_argument("--kinds", nargs="*", default=["mel"])
    ap.add_argument("--frames", nargs="*", type=int, default=[16, 32])
    ap.add_argument("--bands", nargs="*", type=int, default=[8, 12, 16, 20])
    ap.add_argument("--bits", nargs="*", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--lam", nargs="*", type=float,
                    default=[1e1, 1e2, 1e3, 1e4, 1e5])
    ap.add_argument("--ntrain", type=int, default=0, help="0 = all")
    ap.add_argument("--joint-bias", action="store_true",
                    help="quantize the bias row with the weights (the CIFAR "
                         "family's convention; measured here and it fails)")
    a = ap.parse_args(argv)
    split_bias = not a.joint_bias

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    z = np.load(a.data)
    xtr, ytr = z["xtrain"], z["ytrain"]
    xva, yva = z["xval"], z["yval"]
    xte, yte = z["xtest"], z["ytest"]
    if a.ntrain:
        xtr, ytr = xtr[: a.ntrain], ytr[: a.ntrain]
    print(f"train {xtr.shape[0]:,}  val {xva.shape[0]:,}  test {xte.shape[0]:,}"
          f"  chance {100 / K:.2f}%")

    for kind in a.kinds:
        for frames in a.frames:
            for bands in a.bands:
                src = feat_src(kind, frames, bands)
                f = compile_feat(src)
                t0 = time.perf_counter()
                Vtr = features(f, xtr)
                Vva = features(f, xva)
                D = Vtr.shape[1]
                print(f"\n{kind} F={frames} M={bands}: D={D} "
                      f"({time.perf_counter() - t0:.0f}s of feature extraction)")

                heads = {lam: fit(Vtr, ytr, lam) for lam in a.lam}
                for bits in a.bits:
                    # lambda on validation, through the quantized artifact
                    best = None
                    for lam, W in heads.items():
                        files, nc = make_files(src, W, bits, split_bias)
                        p = predict_local(files, Vva, W.shape[0], bits, nc,
                                          split_bias)
                        acc = float((p == yva).mean())
                        if best is None or acc > best[0]:
                            best = (acc, lam, files)
                    vacc, lam, files = best
                    name = f"kws-{kind}-F{frames}-M{bands}-{bits}b"
                    d = out / name
                    d.mkdir(exist_ok=True)
                    A.write_dir(files, d)
                    rec = evaluate_arrays(
                        d, xte, yte, K, name=name, split="test",
                        method=f"{kind} front end, ridge head, {bits}-bit codebook",
                        notes=f"lambda={lam:g} chosen on val; val {vacc * 100:.2f}%")
                    rec.update(kind=kind, frames=frames, bands=bands, bits=bits,
                               dim=D, lam=lam, val_accuracy=vacc,
                               split_bias=split_bias,
                               head_bytes=(D + 1) * K * bits / 8,
                               src_bytes=len(files["predict.py"]),
                               coef_bytes=len(files["w"]))
                    (out / f"{name}.json").write_text(json.dumps(rec, indent=2))
                    print(f"  {bits}b lam={lam:<8g} val {vacc * 100:5.2f}%  "
                          f"{summarize(rec)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
