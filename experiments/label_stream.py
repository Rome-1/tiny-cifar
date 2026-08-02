"""The MDL label stream as a function of class count and test-set size.

The two-part board prices a model against the cost of just sending the answers:

    L_uniform = N * log2(K) / 8   bytes

This script exists to check one specific claim before any experiment is built on
it: that CIFAR-10's test stream and an ImageNet-100 validation stream of 50
images per class are *the same size*, because log2(100) = 2 log2(10) and
5,000 x 2 = 10,000 x 1. If that holds, class count can be varied while the
budget the model has to beat is held fixed, which is the only way the
bytes-per-class curve is a controlled measurement rather than two experiments.

Run: python3 experiments/label_stream.py
"""

from __future__ import annotations

import math


def uniform_bits(n: int, k: int) -> float:
    """Bits to send n labels drawn uniformly from k classes."""
    return n * math.log2(k)


def uniform_bytes(n: int, k: int) -> float:
    return uniform_bits(n, k) / 8.0


ROWS = [
    ("CIFAR-10 test",                     10_000,   10),
    ("ImageNet-100 val, 50/class",         5_000,  100),
    ("ImageNet-1000 val, 50/class",       50_000, 1000),
    ("ImageNet-1000 val, 10/class",       10_000, 1000),
    ("ImageNet-100 val, 10/class",         1_000,  100),
    ("ImageNet-10 val, 50/class",            500,   10),
]


def main() -> int:
    print(f"{'stream':<32}{'N':>8}{'K':>6}{'bits/label':>12}"
          f"{'bits':>14}{'bytes':>12}")
    for name, n, k in ROWS:
        b = uniform_bits(n, k)
        print(f"{name:<32}{n:>8}{k:>6}{math.log2(k):>12.4f}"
              f"{b:>14,.1f}{b / 8:>12,.1f}")

    a = uniform_bytes(10_000, 10)
    b = uniform_bytes(5_000, 100)
    print(f"\nCIFAR-10 test        : {a:,.4f} B  (floor {math.floor(a):,} B, "
          f"ceil {math.ceil(a):,} B)")
    print(f"ImageNet-100 50/class: {b:,.4f} B  (floor {math.floor(b):,} B, "
          f"ceil {math.ceil(b):,} B)")
    print(f"identical: {a == b}   difference: {a - b:.10f} B")

    # The identity is structural, not a coincidence of rounding:
    #   5000 * log2(100) = 5000 * 2 * log2(10) = 10000 * log2(10)
    lhs = 5_000 * math.log2(100)
    rhs = 10_000 * math.log2(10)
    print(f"\n5000*log2(100) = {lhs:,.6f} bits")
    print(f"10000*log2(10) = {rhs:,.6f} bits")
    print(f"equal to float precision: {abs(lhs - rhs) < 1e-9}")

    # The repo quotes 4,152 B for CIFAR-10 (docs/findings.md:137). That is the
    # floor of 4,152.41, i.e. the value truncated, not rounded up. Recorded here
    # so the convention is explicit rather than inferred.
    print(f"\nrepo's quoted CIFAR-10 figure: 4,152 B "
          f"= floor({a:,.2f}) -- truncation, not ceiling")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
