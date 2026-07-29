"""The Hutter framing: a classifier is a compressor of labels.

The model-artifact track asks "how few bytes of model reach accuracy p?" and
reports two numbers that cannot be added together. This track asks the question
the Hutter Prize asks, which has one answer in one unit:

    total bits = |decoder program| + |arithmetic-coded labels|

Both sides hold the images. The encoder codes each test label against its
model's predictive distribution; the decoder, holding the same program, decodes
label i, updates on (x_i, y_i), and decodes label i+1. This is Dawid's
prequential principle and it is a valid code — the decoder's state is a function
of symbols it has already decoded, so it never needs anything it does not have.

Two consequences make this worth building.

**Accuracy and size stop being separate axes.** A model that is more accurate
buys shorter labels; a model that is bigger costs more program. One scalar
orders every point, and the Pareto frontier collapses to a minimum.

**Almost nothing needs shipping.** The weights are not transmitted — they are
*re-derived* by the decoder from data both sides already hold. What is shipped
is the learning algorithm.

That second point is a genuine result and also a warning, so it is stated here
rather than buried: **if the training set is free to both sides, the size axis
of the model-artifact track is largely an artifact of forbidding the model to
look at the data.** Both framings are defensible and they answer different
questions. This file reports both a cold variant (online from scratch, the
honest measure of learning ability) and a warm variant (the decoder refits on
the training set first), so the size of that effect is visible rather than
assumed.

Everything is verified by decoding back. A codelength that does not round-trip
is not a codelength.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from experiments.conv_features import build_feats, fit_head, make_feats_src  # noqa: E402
from tinycifar import artifact as A  # noqa: E402
from tinycifar.coder import Decoder, Encoder, quantize_probs  # noqa: E402
from tinycifar.data import load  # noqa: E402

K = 10


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


# ---------------------------------------------------------------------------
# component models — each maps a feature vector to a distribution over classes
# ---------------------------------------------------------------------------

class Prior:
    """Adaptive class frequencies (Krichevsky-Trofimov). Costs nothing and is
    the floor every other model has to beat."""

    def __init__(self):
        self.c = np.full(K, 0.5)

    def predict(self, f):
        return self.c / self.c.sum()

    def update(self, f, y):
        self.c[y] += 1.0


class OnlineLogistic:
    """Multinomial logistic regression by SGD — one step per example."""

    def __init__(self, dim, lr=0.05, l2=1e-6):
        self.W = np.zeros((dim + 1, K))
        self.lr, self.l2 = lr, l2
        self.n = 0

    def _x(self, f):
        return np.append(f, 1.0)

    def predict(self, f):
        return softmax(self._x(f) @ self.W)

    def update(self, f, y):
        x = self._x(f)
        p = softmax(x @ self.W)
        p[y] -= 1.0
        self.n += 1
        lr = self.lr / (1.0 + self.n / 5000.0)          # decaying step
        self.W -= lr * (np.outer(x, p) + self.l2 * self.W)


class NearestCentroid:
    """Running class means; distributions from negative squared distance."""

    def __init__(self, dim, temp=1.0):
        self.s = np.zeros((K, dim))
        self.c = np.zeros(K)
        self.temp = temp

    def predict(self, f):
        seen = self.c > 0
        if seen.sum() < 2:
            return np.ones(K) / K
        m = np.where(self.c[:, None] > 0, self.s / np.maximum(self.c[:, None], 1), 0.0)
        d = -((m - f) ** 2).sum(1)
        d = np.where(seen, d, -np.inf)
        scale = np.abs(d[seen]).mean() or 1.0
        return softmax(d / (scale * self.temp))

    def update(self, f, y):
        self.s[y] += f
        self.c[y] += 1


# ---------------------------------------------------------------------------
# the mixer — cmix's core idea, one symbol at a time instead of one bit
# ---------------------------------------------------------------------------

class LogisticMixer:
    """Mix component predictions in the logit domain with online weights.

    cmix and PAQ mix many models this way: the mixed logit is a weighted sum of
    component logits, and the weights follow the gradient of coding loss, so the
    mixer learns which model to trust as it goes. Because the update depends
    only on already-coded symbols, the decoder can run it identically.
    """

    def __init__(self, n_models, lr=0.05, w_init=None):
        # Equal weights are the wrong default here. With M models the mixed
        # logit is (1/M) * log p, which flattens a well-calibrated component
        # towards uniform: a model worth 1.55 bits/label coded at 3.25. cmix can
        # afford 1/M because its components are individually calibrated and it
        # has 10^9 symbols to correct over; we have 10^4 and one strong model.
        # So start by trusting the strong component and let the gradient move.
        self.w = np.ones(n_models) / n_models if w_init is None else np.asarray(
            w_init, dtype=np.float64)
        self.lr = lr

    def mix(self, ps):
        self.L = np.log(np.clip(np.asarray(ps), 1e-8, None))   # (M, K)
        self.q = softmax(self.w @ self.L)
        return self.q

    def update(self, y):
        g = self.q.copy()
        g[y] -= 1.0
        self.w -= self.lr * (self.L @ g)


class APM:
    """Secondary estimation: an adaptive map from (predicted p, class) to a
    corrected p, indexed by a coarse bucket of the prediction. PAQ calls this
    SSE; it fixes systematic over- and under-confidence that the mixer cannot."""

    def __init__(self, buckets=24, rate=0.02):
        self.b = buckets
        self.t = np.linspace(0.05, 0.95, buckets)[:, None].repeat(K, 1)
        self.rate = rate

    def _idx(self, p):
        return np.clip((p * (self.b - 1)).astype(int), 0, self.b - 1)

    def apply(self, p):
        self._i = self._idx(p)
        out = 0.7 * p + 0.3 * self.t[self._i, np.arange(K)]
        return out / out.sum()

    def update(self, y):
        tgt = np.zeros(K)
        tgt[y] = 1.0
        self.t[self._i, np.arange(K)] += self.rate * (tgt - self.t[self._i, np.arange(K)])


# ---------------------------------------------------------------------------
# the coding run
# ---------------------------------------------------------------------------

def build_models(dim, warm_W=None, use=("prior", "logistic", "centroid")):
    ms = []
    for name in use:
        if name == "prior":
            ms.append(Prior())
        elif name == "logistic":
            m = OnlineLogistic(dim)
            if warm_W is not None:
                m.W = warm_W.copy()
            ms.append(m)
        elif name == "centroid":
            ms.append(NearestCentroid(dim))
    return ms


def code(F, y, dim, warm_W=None, use=("prior", "logistic", "centroid"),
         apm=True, decode_check=True, w_init=None, lr=None):
    """Encode all labels, then decode them back. Returns a report dict."""
    def fresh():
        ms = build_models(dim, warm_W, use)
        if lr is not None:
            for m in ms:
                if isinstance(m, OnlineLogistic):
                    m.lr = lr
        return ms, LogisticMixer(len(ms), w_init=w_init), (APM() if apm else None)

    models, mixer, sse = fresh()
    enc = Encoder()
    ideal = 0.0
    correct = 0
    for i in range(len(F)):
        f = F[i]
        p = mixer.mix([m.predict(f) for m in models])
        if sse is not None:
            p = sse.apply(p)
        freq = quantize_probs(p)

        yi = int(y[i])
        correct += int(np.argmax(p) == yi)
        ideal += -np.log2(freq[yi] / freq.sum())
        enc.encode(yi, freq)

        mixer.update(yi)
        if sse is not None:
            sse.update(yi)
        for m in models:
            m.update(f, yi)
    blob = enc.finish()

    ok = None
    if decode_check:
        models, mixer, sse = fresh()
        dec = Decoder(blob)
        out = np.empty(len(F), dtype=np.int64)
        for i in range(len(F)):
            f = F[i]
            p = mixer.mix([m.predict(f) for m in models])
            if sse is not None:
                p = sse.apply(p)
            yi = dec.decode(quantize_probs(p))
            out[i] = yi
            mixer.update(yi)
            if sse is not None:
                sse.update(yi)
            for m in models:
                m.update(f, yi)
        ok = bool((out == np.asarray(y)).all())

    return {
        "bytes": len(blob),
        "ideal_bytes": ideal / 8,
        "bits_per_label": 8 * len(blob) / len(F),
        "online_accuracy": correct / len(F),
        "roundtrip": ok,
        "n": len(F),
    }


def program_bytes(feats_src: str, warm: bool) -> dict:
    """Description length of the decoder program.

    Measuring the whole source files would charge the decoder for argparse,
    docstrings and experiment scaffolding it never runs — 10 KB, which swamped
    the label term and made the total meaningless. Instead the source of exactly
    the definitions the decode loop touches is pulled with `inspect.getsource`,
    so the measured bytes are the code that actually ran rather than a
    hand-maintained copy of it that could drift.

    Docstrings are stripped, since they are commentary rather than program.
    """
    import inspect
    import re

    parts = [softmax, Prior, OnlineLogistic, NearestCentroid, LogisticMixer,
             APM, build_models, quantize_probs, Decoder, code]
    if warm:
        parts += [fit_head]

    src = "\n".join(inspect.getsource(p) for p in parts)
    src = re.sub(r'"""[\s\S]*?"""', "", src)          # drop docstrings
    src = re.sub(r"^\s*#.*$", "", src, flags=re.M)    # drop comments
    src = re.sub(r"\n\s*\n+", "\n", src)

    files = {"decode.py": src.encode(), "feats.py": feats_src.encode()}
    s = A.measure(files)
    return {"raw": s.raw, "gzip": s.gzip, "xz": s.xz,
            "description_length": s.description_length}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=64, help="conv filters")
    ap.add_argument("--pool", type=int, default=5)
    ap.add_argument("--patch", type=int, default=4)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--lam", type=float, default=1e2)
    ap.add_argument("--mode", default="both", choices=["cold", "warm", "both"])
    ap.add_argument("--no-apm", action="store_true")
    ap.add_argument("--stream", default="test", choices=["test", "train"],
                    help="which label stream to code; train has 50,000 symbols "
                         "and is where fixed program cost should amortize")
    a = ap.parse_args(argv)

    xtr, ytr, xte, yte = load()
    scale = 1.0 / np.sqrt(a.patch * a.patch * 3)
    src, dim = make_feats_src(a.k, a.patch, a.stride, a.pool, a.seed, scale, 0.1)
    feats = build_feats(src)

    def featurize(imgs):
        return np.vstack([feats(imgs[i:i + 500].astype(np.float32) / 255)
                          for i in range(0, len(imgs), 500)])

    t0 = time.perf_counter()
    # The normalizing constant comes from the TRAINING set, not the test set:
    # both sides hold the training data, and using test statistics would be a
    # transductive leak the decoder is not entitled to.
    Ftr_head = featurize(xtr[:5000])
    scale_f = float(np.abs(Ftr_head).mean()) or 1.0
    Ftr_head /= scale_f

    xs, ys = (xtr, ytr) if a.stream == "train" else (xte, yte)
    F = featurize(xs) / scale_f
    print(f"features: {F.shape} ({a.stream} stream) in {time.perf_counter() - t0:.0f}s")

    uniform = len(ys) * np.log2(10) / 8
    print(f"uniform label cost: {uniform:,.0f} B\n")

    modes = ["cold", "warm"] if a.mode == "both" else [a.mode]
    for mode in modes:
        warm_W = None
        if mode == "warm":
            t0 = time.perf_counter()
            W = fit_head(feats, xtr, ytr, dim, a.lam, tta=False)

            # fit_head saw unnormalized features; the online stream is divided
            # by scale_f, so the weights must be multiplied by it to match.
            warm_W = W.copy()
            warm_W[:-1] *= scale_f

            # Ridge outputs are not logits — softmax of them is nearly uniform,
            # which would cost ~3.3 bits/label however accurate the model is.
            # The temperature is chosen on the training set, so the decoder can
            # repeat the search exactly.
            logits = np.hstack([Ftr_head, np.ones((len(Ftr_head), 1))]) @ warm_W
            ytr_head = ytr[:5000]
            best, temp = np.inf, 1.0
            for t in np.geomspace(0.5, 200.0, 40):
                z = t * logits
                z -= z.max(1, keepdims=True)
                ce = float(np.mean(np.log(np.exp(z).sum(1)) - z[np.arange(len(z)), ytr_head]))
                if ce < best:
                    best, temp = ce, float(t)
            warm_W *= temp
            print(f"warm start: ridge on train in {time.perf_counter() - t0:.0f}s "
                  f"(temperature {temp:.1f}, train CE {best / np.log(2):.3f} bits)")

        prog = program_bytes(src, warm=(mode == "warm"))
        t0 = time.perf_counter()
        # warm: trust the pretrained head from the start, and take small SGD
        # steps so online updates refine it instead of destroying it.
        # cold: start on the prior, which is all that is known at step zero.
        w_init = [0.0, 1.0, 0.0] if mode == "warm" else [1.0, 0.0, 0.0]
        r = code(F, ys, dim, warm_W=warm_W, apm=not a.no_apm,
                 w_init=w_init, lr=0.002 if mode == "warm" else None)
        total = r["bytes"] + prog["description_length"]
        print(f"[{mode}] program {prog['description_length']:,} B "
              f"(raw {prog['raw']:,}), labels {r['bytes']:,} B "
              f"({r['bits_per_label']:.3f} bits/label), "
              f"online acc {r['online_accuracy'] * 100:.2f}%, "
              f"roundtrip {'ok' if r['roundtrip'] else 'FAILED'}")
        print(f"       total = {total:,} B  "
              f"(vs {uniform:,.0f} B uniform, "
              f"{100 * (1 - r['bytes'] / uniform):.1f}% saved on labels) "
              f"[{time.perf_counter() - t0:.0f}s]\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
