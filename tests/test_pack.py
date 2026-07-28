"""Tests for quantization and bit packing."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tinycifar import pack as P  # noqa: E402


def test_bitpack_roundtrip_all_widths():
    rng = np.random.default_rng(0)
    for bits in range(1, 17):
        n = 1000
        codes = rng.integers(0, 1 << bits, size=n, dtype=np.uint16)
        back = P.bitunpack(P.bitpack(codes, bits), bits, n)
        assert np.array_equal(back, codes), f"{bits}-bit roundtrip failed"


def test_bitpack_density():
    """Packing must actually spend `bits` per code, not a byte per code."""
    codes = np.zeros(8000, dtype=np.uint16)
    for bits in (1, 2, 3, 4, 5, 6):
        assert len(P.bitpack(codes, bits)) == 8000 * bits // 8


def test_tensor_roundtrip_preserves_shape():
    rng = np.random.default_rng(1)
    w = rng.standard_normal((7, 13, 3)).astype(np.float32)
    back, off = P.unpack_tensor(P.pack_tensor(w, 8))
    assert back.reshape(w.shape).shape == w.shape
    assert off == len(P.pack_tensor(w, 8))


def test_quantization_error_decreases_monotonically():
    rng = np.random.default_rng(2)
    w = rng.standard_normal(4096).astype(np.float32)
    errs = [P.roundtrip_error(w, b) for b in range(2, 13)]
    assert all(a > b for a, b in zip(errs, errs[1:])), errs
    assert errs[-1] < 0.01


def test_eight_bit_is_accurate():
    rng = np.random.default_rng(3)
    w = rng.standard_normal(10000).astype(np.float32)
    assert P.roundtrip_error(w, 8) < 0.01


def test_symmetric_beats_affine_at_one_bit():
    """A ternary/binary method wants the symmetric grid centered on zero."""
    rng = np.random.default_rng(4)
    w = rng.standard_normal(4096).astype(np.float32)
    assert P.roundtrip_error(w, 1, symmetric=True) < P.roundtrip_error(w, 1)


def test_constant_tensor_does_not_divide_by_zero():
    w = np.full(100, 2.5, dtype=np.float32)
    back, _ = P.unpack_tensor(P.pack_tensor(w, 4))
    assert np.allclose(back, 2.5, atol=1e-5)


def test_pack_model_roundtrip_mixed_widths():
    rng = np.random.default_rng(5)
    ts = [rng.standard_normal((4, 5)).astype(np.float32),
          rng.standard_normal(9).astype(np.float32)]
    back = P.unpack_model(P.pack_model(ts, bits=[6, 8]), 2)
    assert len(back) == 2
    for a, b in zip(ts, back):
        assert np.allclose(a.reshape(-1), b.reshape(-1), atol=0.1)


def test_bytes_per_param_matches_target():
    """Sanity on the headline claim: n-bit packing costs ~n/8 bytes/param."""
    rng = np.random.default_rng(6)
    w = rng.standard_normal(80000).astype(np.float32)
    for bits in (2, 4, 8):
        overhead = len(P.pack_tensor(w, bits)) - 80000 * bits / 8
        assert 0 <= overhead < 32, (bits, overhead)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
