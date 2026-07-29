"""Tests for the arithmetic coder.

The property that matters is not compression ratio but exact round-trip under an
*adaptive* model. If the decoder's frequency tables ever diverge from the
encoder's, the labels come back wrong — and in the prequential track that is the
difference between a valid code and a number that means nothing.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tinycifar.coder import (  # noqa: E402
    Decoder, Encoder, TOTAL, ideal_bits, quantize_probs,
)


def _roundtrip(syms, probs_fn):
    enc = Encoder()
    for i, s in enumerate(syms):
        enc.encode(int(s), quantize_probs(probs_fn(i, syms[:i])))
    blob = enc.finish()

    dec = Decoder(blob)
    out = []
    for i in range(len(syms)):
        out.append(dec.decode(quantize_probs(probs_fn(i, out[:i]))))
    return blob, out


def test_uniform_roundtrip():
    syms = np.random.default_rng(0).integers(0, 10, 2000)
    blob, out = _roundtrip(syms, lambda i, past: np.ones(10))
    assert out == list(syms)
    assert abs(len(blob) - len(syms) * np.log2(10) / 8) < 8


def test_skewed_is_much_smaller():
    rng = np.random.default_rng(1)
    p = np.array([0.9] + [0.1 / 9] * 9)
    syms = rng.choice(10, size=3000, p=p)
    blob, out = _roundtrip(syms, lambda i, past: p)
    assert out == list(syms)
    ideal = sum(ideal_bits(p, s) for s in syms) / 8
    assert len(blob) < ideal + 8
    assert len(blob) < 0.25 * 3000 * np.log2(10) / 8


def test_adaptive_model_roundtrips():
    """The case the prequential track depends on: tables built from history."""
    rng = np.random.default_rng(2)
    syms = rng.choice(10, size=4000, p=np.array([5, 3, 2, 1, 1, 1, 1, 1, 1, 1]) / 17)

    def adaptive(i, past):
        counts = np.ones(10)
        for s in past:
            counts[s] += 1
        return counts / counts.sum()

    blob, out = _roundtrip(syms, adaptive)
    assert out == list(syms), "adaptive roundtrip diverged"
    # An adaptive coder should beat uniform on a skewed source.
    assert len(blob) < 4000 * np.log2(10) / 8


def test_confident_and_wrong_is_expensive_but_survives():
    """A model that is confidently wrong must still produce a decodable stream."""
    syms = [3] * 200
    p = np.zeros(10)
    p[7] = 1.0
    blob, out = _roundtrip(syms, lambda i, past: p)
    assert out == syms


def test_quantize_probs_sums_and_floors():
    for p in (np.ones(10), np.array([1.0] + [0.0] * 9),
              np.array([0.5, 0.5] + [0.0] * 8)):
        f = quantize_probs(p)
        assert f.sum() == TOTAL
        assert (f >= 1).all(), "a zero-frequency symbol would cost infinite bits"


def test_quantize_probs_tracks_the_distribution():
    p = np.array([0.7, 0.2, 0.1] + [0.0] * 7)
    f = quantize_probs(p)
    assert f[0] > f[1] > f[2] > f[3]


def test_coded_length_tracks_ideal():
    """Actual bytes should sit just above the ideal codelength, not below it."""
    rng = np.random.default_rng(3)
    p = np.array([0.4, 0.3, 0.1, 0.05, 0.05, 0.03, 0.03, 0.02, 0.01, 0.01])
    syms = rng.choice(10, size=5000, p=p)
    blob, out = _roundtrip(syms, lambda i, past: p)
    assert out == list(syms)
    ideal = sum(ideal_bits(p, s) for s in syms) / 8
    assert ideal - 2 < len(blob) < ideal + 8, (len(blob), ideal)


def test_single_symbol_and_empty():
    enc = Encoder()
    assert len(enc.finish()) >= 1
    blob, out = _roundtrip([4], lambda i, past: np.ones(10))
    assert out == [4]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
