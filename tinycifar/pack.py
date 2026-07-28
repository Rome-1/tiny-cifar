"""Quantize and bit-pack weight tensors.

Shared by every method: once a model's parameters are fixed, the remaining
lever is bits per parameter. These helpers do the two cheap, universal wins —
uniform affine quantization to n bits, and dense bit packing — and leave the
entropy coding to the artifact's own compression (see `artifact.measure`,
which already takes the min over raw/gzip/xz).

Header format for a packed tensor (little-endian)::

    uint8   bits
    uint8   ndim
    uint32  shape[ndim]
    float32 scale
    float32 zero
    bytes   payload   (ceil(n_elements * bits / 8) bytes)

The unpacking side is ~15 lines of numpy, which is what an artifact ships.
"""

from __future__ import annotations

import struct

import numpy as np


def quantize(w: np.ndarray, bits: int = 8, symmetric: bool = False):
    """Uniform affine quantization. Returns (codes, scale, zero).

    Dequantization is ``codes * scale + zero``.
    """
    if not 1 <= bits <= 16:
        raise ValueError(f"bits must be in 1..16, got {bits}")
    w = np.asarray(w, dtype=np.float64)
    levels = (1 << bits) - 1

    if symmetric:
        m = float(np.abs(w).max()) or 1.0
        half = levels // 2
        scale = m / half if half else 1.0
        codes = np.clip(np.rint(w / scale) + half, 0, levels)
        zero = -half * scale
    else:
        lo, hi = float(w.min()), float(w.max())
        scale = (hi - lo) / levels if hi > lo else 1.0
        codes = np.clip(np.rint((w - lo) / scale), 0, levels)
        zero = lo

    return codes.astype(np.uint16), float(scale), float(zero)


def bitpack(codes: np.ndarray, bits: int) -> bytes:
    """Pack unsigned codes into a dense little-endian bitstream."""
    codes = np.asarray(codes, dtype=np.uint16).reshape(-1)
    if bits == 8:
        return codes.astype(np.uint8).tobytes()
    if bits == 16:
        return codes.astype("<u2").tobytes()
    bit_mat = ((codes[:, None] >> np.arange(bits)[None, :]) & 1).astype(np.uint8)
    return np.packbits(bit_mat.reshape(-1), bitorder="little").tobytes()


def bitunpack(buf: bytes, bits: int, n: int) -> np.ndarray:
    if bits == 8:
        return np.frombuffer(buf, dtype=np.uint8, count=n).astype(np.uint16)
    if bits == 16:
        return np.frombuffer(buf, dtype="<u2", count=n).astype(np.uint16)
    raw = np.unpackbits(np.frombuffer(buf, dtype=np.uint8), bitorder="little")
    raw = raw[: n * bits].reshape(n, bits).astype(np.uint16)
    return (raw << np.arange(bits, dtype=np.uint16)[None, :]).sum(axis=1)


def pack_tensor(w: np.ndarray, bits: int = 8, symmetric: bool = False) -> bytes:
    """Quantize + serialize one tensor, header included."""
    codes, scale, zero = quantize(w, bits, symmetric)
    shape = np.asarray(w).shape
    head = struct.pack("<BB", bits, len(shape)) + struct.pack(
        f"<{len(shape)}I", *shape
    ) + struct.pack("<ff", scale, zero)
    return head + bitpack(codes, bits)


def unpack_tensor(buf: bytes, offset: int = 0):
    """Inverse of `pack_tensor`. Returns (array, new_offset)."""
    bits, ndim = struct.unpack_from("<BB", buf, offset)
    offset += 2
    shape = struct.unpack_from(f"<{ndim}I", buf, offset)
    offset += 4 * ndim
    scale, zero = struct.unpack_from("<ff", buf, offset)
    offset += 8
    n = int(np.prod(shape)) if shape else 1
    nbytes = (n * bits + 7) // 8
    codes = bitunpack(buf[offset : offset + nbytes], bits, n)
    return codes.astype(np.float32) * scale + zero, offset + nbytes


def pack_model(tensors: list[np.ndarray], bits=8, symmetric: bool = False) -> bytes:
    """Pack a list of tensors into one blob."""
    if isinstance(bits, int):
        bits = [bits] * len(tensors)
    return b"".join(pack_tensor(t, b, symmetric) for t, b in zip(tensors, bits))


def unpack_model(buf: bytes, count: int) -> list[np.ndarray]:
    out, off = [], 0
    for _ in range(count):
        t, off = unpack_tensor(buf, off)
        out.append(t)
    return out


def roundtrip_error(w: np.ndarray, bits: int, symmetric: bool = False) -> float:
    """Relative L2 error from quantizing at `bits` — use to pick a bit width."""
    back, _ = unpack_tensor(pack_tensor(w, bits, symmetric))
    w = np.asarray(w, dtype=np.float32)
    denom = np.linalg.norm(w) or 1.0
    return float(np.linalg.norm(back.reshape(w.shape) - w) / denom)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    w = rng.standard_normal((64, 128)).astype(np.float32)
    print(f"{'bits':>5} {'bytes':>8} {'rel.err':>9}")
    for b in (1, 2, 3, 4, 5, 6, 8):
        print(f"{b:>5} {len(pack_tensor(w, b)):>8,} {roundtrip_error(w, b):>9.4f}")
