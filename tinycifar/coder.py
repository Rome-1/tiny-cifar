"""An arithmetic coder, so codelengths are measured rather than asserted.

The prequential track claims a number of bytes. It would be easy — and weaker —
to report the ideal codelength, sum over i of -log2 q(y_i), and call it done.
That quantity is real but it is not a file. This module codes the labels for
real and reports the length of the actual byte string, which is within a couple
of bits of the ideal and, more to the point, can be decoded back.

The coder is the classic Witten-Neal-Cleary integer arithmetic coder with
32-bit registers and underflow (E3) handling. Symbols are coded against an
integer frequency table supplied per symbol, so the model may be adaptive: the
decoder reconstructs each table from symbols it has already decoded, which is
what makes online learning a valid code.

**Integers are not a stylistic choice.** Encoder and decoder must agree
bit-for-bit on every probability, and floating-point sums do not reassociate
identically across machines or numpy versions. Probabilities are therefore
quantized to integers once, by `quantize_probs`, and the coder only ever sees
those integers.
"""

from __future__ import annotations

import numpy as np

BITS = 32
TOP = (1 << BITS) - 1
HALF = 1 << (BITS - 1)
QUARTER = 1 << (BITS - 2)
THREE_QUARTER = 3 * QUARTER

# Frequency totals must stay well below the register width or the range can
# collapse; 2^16 is the standard safe ceiling for a 32-bit coder.
TOTAL = 1 << 16


def quantize_probs(p, total: int = TOTAL) -> np.ndarray:
    """Turn a float distribution into integer frequencies summing to `total`.

    Every symbol keeps at least one count. A symbol with zero probability would
    otherwise cost infinite bits the moment it actually occurred — and in an
    online setting it eventually does.
    """
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    n = len(p)
    if total < n:
        raise ValueError(f"total {total} too small for {n} symbols")
    p = np.clip(p, 1e-12, None)
    p /= p.sum()

    f = np.floor(p * (total - n)).astype(np.int64) + 1
    short = total - int(f.sum())
    if short:                      # hand the rounding remainder to the largest
        f[np.argsort(-p)[:abs(short)]] += np.sign(short)
    return f


class Encoder:
    """Arithmetic encoder. Call `encode` per symbol, then `finish`."""

    def __init__(self) -> None:
        self.low = 0
        self.high = TOP
        self.pending = 0
        self._bits: list[int] = []

    def _bit(self, b: int) -> None:
        self._bits.append(b)
        self._bits.extend([1 - b] * self.pending)
        self.pending = 0

    def encode(self, sym: int, freq: np.ndarray) -> None:
        cum = np.concatenate([[0], np.cumsum(freq)])
        total = int(cum[-1])
        rng = self.high - self.low + 1
        self.high = self.low + (rng * int(cum[sym + 1])) // total - 1
        self.low = self.low + (rng * int(cum[sym])) // total

        while True:
            if self.high < HALF:
                self._bit(0)
            elif self.low >= HALF:
                self._bit(1)
                self.low -= HALF
                self.high -= HALF
            elif self.low >= QUARTER and self.high < THREE_QUARTER:
                self.pending += 1
                self.low -= QUARTER
                self.high -= QUARTER
            else:
                break
            self.low = (self.low << 1) & TOP
            self.high = ((self.high << 1) | 1) & TOP

    def finish(self) -> bytes:
        self.pending += 1
        self._bit(0 if self.low < QUARTER else 1)
        bits = self._bits + [0] * (-len(self._bits) % 8)
        return bytes(
            int("".join(map(str, bits[i : i + 8])), 2)
            for i in range(0, len(bits), 8)
        )


class Decoder:
    """Arithmetic decoder. Feed it the same frequency tables, in order."""

    def __init__(self, data: bytes) -> None:
        self._bits = [(b >> (7 - i)) & 1 for b in data for i in range(8)]
        self._pos = 0
        self.low = 0
        self.high = TOP
        self.value = 0
        for _ in range(BITS):
            self.value = (self.value << 1) | self._next()

    def _next(self) -> int:
        if self._pos < len(self._bits):
            b = self._bits[self._pos]
            self._pos += 1
            return b
        return 0                      # past the end, feed zeros

    def decode(self, freq: np.ndarray) -> int:
        cum = np.concatenate([[0], np.cumsum(freq)])
        total = int(cum[-1])
        rng = self.high - self.low + 1
        scaled = ((self.value - self.low + 1) * total - 1) // rng
        sym = int(np.searchsorted(cum, scaled, side="right") - 1)
        sym = min(max(sym, 0), len(freq) - 1)

        self.high = self.low + (rng * int(cum[sym + 1])) // total - 1
        self.low = self.low + (rng * int(cum[sym])) // total

        while True:
            if self.high < HALF:
                pass
            elif self.low >= HALF:
                self.low -= HALF
                self.high -= HALF
                self.value -= HALF
            elif self.low >= QUARTER and self.high < THREE_QUARTER:
                self.low -= QUARTER
                self.high -= QUARTER
                self.value -= QUARTER
            else:
                break
            self.low = (self.low << 1) & TOP
            self.high = ((self.high << 1) | 1) & TOP
            self.value = ((self.value << 1) | self._next()) & TOP
        return sym


def ideal_bits(p, sym: int) -> float:
    """-log2 q(sym): what the coder should charge, for cross-checking."""
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    return float(-np.log2(max(p[sym] / p.sum(), 1e-300)))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    syms = rng.integers(0, 10, 5000)

    enc = Encoder()
    for s in syms:
        enc.encode(int(s), quantize_probs(np.ones(10)))
    blob = enc.finish()

    dec = Decoder(blob)
    back = [dec.decode(quantize_probs(np.ones(10))) for _ in syms]
    print("uniform 10-symbol:", len(blob), "bytes,",
          "roundtrip", "ok" if back == list(syms) else "FAILED",
          f"(ideal {len(syms) * np.log2(10) / 8:.0f})")
