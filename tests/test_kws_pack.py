"""The weight-payload round-trip gate, and proof that it can fail.

`experiments/kws_ridge.pack_indices` writes the quantizer's codes into the file
an artifact ships; `golf.unpack_expr` generates the source that reads them back.
Nothing else connects the two, and a mismatch is silent: the artifact is still
numpy-only, still self-contained, still returns well-formed labels, and still
passes every check the harness makes. It just ships noise.

That happened. The first Speech Commands sweep packed `8 // bits` codes per
byte, which is right for widths dividing 8 and wrong for 3-bit, and every 3-bit
artifact scored at chance while looking like a quantizer problem.

These tests pin the layout and — the part that matters — pin that the gate
fires on a payload that does not round-trip. A detector that has only ever been
run against correct input has not been tested.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from experiments.golf import unpack_expr  # noqa: E402
from experiments.kws_ridge import check_roundtrip, pack_indices  # noqa: E402

WIDTHS = [1, 2, 3, 4, 8]


@pytest.mark.parametrize("bits", WIDTHS)
def test_roundtrip(bits):
    n = 1000
    idx = np.random.default_rng(0).integers(0, 1 << bits, n).astype(np.uint8)
    payload = pack_indices(idx, bits)
    ns = {"np": np, "B": payload}
    back = np.asarray(eval(unpack_expr(bits, n, 0), ns)).reshape(-1)[:n]
    assert np.array_equal(back, idx)


@pytest.mark.parametrize("bits", WIDTHS)
def test_no_wasted_bits(bits):
    """The packing must be dense. The bug was 19 bytes where 14 were needed."""
    n = 1000
    idx = np.zeros(n, dtype=np.uint8)
    assert len(pack_indices(idx, bits)) == -(-n * bits // 8)


@pytest.mark.parametrize("bits", WIDTHS)
def test_gate_accepts_a_correct_payload(bits):
    idx = np.random.default_rng(1).integers(0, 1 << bits, 512).astype(np.uint8)
    check_roundtrip(pack_indices(idx, bits), idx, bits, 0)


@pytest.mark.parametrize("bits", WIDTHS)
def test_gate_rejects_a_wrong_payload(bits):
    """Mutation: hand the gate bytes that are not the codes it was given."""
    idx = np.full(512, (1 << bits) - 1, dtype=np.uint8)
    wrong = pack_indices(np.zeros(512, dtype=np.uint8), bits)
    with pytest.raises(RuntimeError, match="does not round-trip"):
        check_roundtrip(wrong, idx, bits, 0)


def test_gate_rejects_the_original_bug():
    """The exact defect: `8 // bits` codes per byte at a width that does not
    divide 8. This is the layout that shipped noise, reconstructed."""
    bits, n = 3, 512
    idx = np.random.default_rng(2).integers(0, 8, n).astype(np.uint8)
    per = 8 // bits
    pad = (-n) % per
    flat = np.concatenate([idx, np.zeros(pad, np.uint8)])
    buggy = np.zeros(len(flat) // per, np.uint8)
    for i in range(per):
        buggy |= flat[i::per] << (i * bits)
    assert buggy.tobytes() != pack_indices(idx, bits)
    with pytest.raises(RuntimeError, match="does not round-trip"):
        check_roundtrip(buggy.tobytes(), idx, bits, 0)
