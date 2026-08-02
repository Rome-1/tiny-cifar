"""Turn class-scaling result records into the bytes-per-class table.

Reads the JSON records `class_scaling.py` wrote and reports, per (K, bits):

* measured artifact bytes, and the bytes the head-cost arithmetic predicts,
  `source + 2^b * 2 + (dim+1) * K * b / 8` — the point being to check whether
  the measured curve is the arithmetic or something else;
* accuracy against chance, as a lift, because 12% means opposite things at
  10 classes and at 1000;
* the two-part MDL total, `artifact + N * H(labels | model) / 8`, against the
  no-model budget `N * log2(K) / 8`. Cross-entropy is upper-bounded here by the
  error-rate coding bound rather than fitted from logits — see `price_labels`.

Run: python3 experiments/class_scaling_report.py results/class-scaling
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def price_labels(acc: float, n: int, k: int) -> float:
    """Bytes to send n labels given a classifier of accuracy `acc`.

    The bound used is the standard one: send one bit-ish flag for right/wrong
    (H(acc) bits, coded), and when wrong send which of the other k-1 classes it
    was (log2(k-1) bits). That is achievable by a decoder that has the model,
    and it needs no fitted temperature, so it cannot be tuned. It is an upper
    bound on what a calibrated coder would spend, which makes every MDL verdict
    below conservative in the direction of "the model does not pay".
    """
    p = min(max(acc, 1e-12), 1 - 1e-12)
    h = -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
    bits = n * (h + (1 - p) * math.log2(k - 1))
    return bits / 8.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dirs", nargs="+", type=Path)
    a = ap.parse_args(argv)

    recs = []
    for d in a.dirs:
        for f in sorted(d.glob("cs-*.json")):
            recs.append(json.loads(f.read_text()))
    if not recs:
        print("no records found", file=sys.stderr)
        return 1
    recs.sort(key=lambda r: (r["n_classes"], r["bits"], r["k"]))

    hdr = (f"{'K':>6}{'k':>4}{'b':>3}{'dim':>6}{'bytes':>10}{'head B':>9}"
           f"{'test':>8}{'chance':>8}{'lift':>7}{'B/class':>9}"
           f"{'MDL tot':>10}{'budget':>9}{'pays?':>7}")
    print(hdr)
    print("-" * len(hdr))
    for r in recs:
        K, b = r["n_classes"], r["bits"]
        dim = r["dim"]
        head = (dim + 1) * K * b / 8
        n = r["n"]
        lab = price_labels(r["accuracy"], n, K)
        total = r["description_length"] + lab
        budget = n * math.log2(K) / 8
        print(f"{K:>6}{r['k']:>4}{b:>3}{dim:>6}{r['description_length']:>10,}"
              f"{head:>9,.0f}{r['accuracy'] * 100:>7.2f}%"
              f"{100 / K:>7.2f}%{r['accuracy'] * K:>6.1f}x"
              f"{r['description_length'] / K:>9.1f}"
              f"{total:>10,.0f}{budget:>9,.0f}"
              f"{'yes' if total < budget else 'no':>7}")

    print("\nsmallest artifact per K, and what it scores:")
    for K in sorted({r["n_classes"] for r in recs}):
        rs = [r for r in recs if r["n_classes"] == K]
        s = min(rs, key=lambda r: r["description_length"])
        t = max(rs, key=lambda r: r["accuracy"])
        print(f"  K={K:<5} floor {s['description_length']:>8,} B "
              f"@ {s['accuracy'] * 100:5.2f}% ({s['accuracy'] * K:.1f}x chance)"
              f"   |  best {t['accuracy'] * 100:5.2f}% "
              f"@ {t['description_length']:,} B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
