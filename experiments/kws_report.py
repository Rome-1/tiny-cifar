"""Tabulate the Speech Commands sweep and price it on the MDL board.

Same two-part accounting as `experiments/class_scaling_report.py`, so the
verdict here is comparable to the CIFAR-10 and ImageNet ones:

    total = artifact bytes + arithmetic-coded label bytes

`price_labels` is imported rather than reimplemented -- it is the same
conservative error-rate bound, with no fitted temperature and therefore nothing
tunable. The budget it is measured against is the cost of sending the 4,890 test
labels with no model at all.

Run:
    python3 experiments/kws_report.py /tmp/kws_ridge [/tmp/kws_table ...]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from experiments.class_scaling_report import price_labels  # noqa: E402

K = 12


def label_budget(y: np.ndarray) -> tuple[float, float]:
    """(uniform bound, empirical-entropy bound) in bytes for a label stream."""
    n = len(y)
    p = np.bincount(y, minlength=K) / n
    p = p[p > 0]
    return n * math.log2(K) / 8, float(n * -(p * np.log2(p)).sum() / 8)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--data", default=str(REPO / "data" / "sc12.npz"))
    a = ap.parse_args(argv)

    yte = np.load(a.data)["ytest"]
    uni, emp = label_budget(yte)
    print(f"test stream: {len(yte):,} labels over {K} classes")
    print(f"  uniform    {len(yte)} x log2({K}) / 8 = {uni:,.1f} B")
    print(f"  empirical  H = {emp * 8 / len(yte):.4f} bits/label "
          f"= {emp:,.1f} B  (the split is near-balanced by construction)")
    budget = math.floor(uni)
    print(f"  budget used below: {budget:,} B (the uniform figure, truncated, "
          f"matching the repo's CIFAR convention)\n")

    recs = []
    for d in a.dirs:
        for f in sorted(d.glob("*.json")):
            recs.append(json.loads(f.read_text()))
    if not recs:
        print("no records found", file=sys.stderr)
        return 1
    recs.sort(key=lambda r: r["description_length"])

    print(f"{'artifact':<26}{'bytes':>8}{'src':>6}{'coef':>7}{'dim':>6}"
          f"{'val':>8}{'test':>8}{'lift':>7}{'labels':>9}{'total':>9}"
          f"{'vs budget':>11}  pays?")
    print("-" * 118)
    payers = []
    for r in recs:
        acc = r["accuracy"]
        lab = price_labels(acc, len(yte), K)
        total = r["description_length"] + lab
        pays = total < budget
        if pays:
            payers.append((total, r))
        print(f"{r['name']:<26}{r['description_length']:>8,}"
              f"{r.get('src_bytes', 0):>6}{r.get('coef_bytes', 0):>7,}"
              f"{r.get('dim', 0):>6}"
              f"{r.get('val_accuracy', float('nan')) * 100:>7.2f}%"
              f"{acc * 100:>7.2f}%{acc * K:>6.1f}x{lab:>9,.0f}{total:>9,.0f}"
              f"{total - budget:>+11,.0f}  {'YES' if pays else 'no'}")

    best = max(recs, key=lambda r: r["accuracy"])
    small = min(recs, key=lambda r: r["description_length"])
    print(f"\nbest accuracy : {best['accuracy'] * 100:.2f}% @ "
          f"{best['description_length']:,} B ({best['name']})")
    print(f"smallest      : {small['accuracy'] * 100:.2f}% @ "
          f"{small['description_length']:,} B ({small['name']})")

    print(f"\nKILL CRITERION 2 -- a ~1 KB artifact must beat chance "
          f"({100 / K:.2f}%) by a wide margin.")
    kb = [r for r in recs if r["description_length"] <= 1100]
    if not kb:
        print("  no artifact at or under 1,100 B -- CRITERION FIRES by absence")
        return 0
    b = max(kb, key=lambda r: r["accuracy"])
    se = math.sqrt(b["accuracy"] * (1 - b["accuracy"]) / len(yte)) * 100
    z = (b["accuracy"] * 100 - 100 / K) / se
    print(f"  best at or under 1,100 B: {b['name']} "
          f"{b['accuracy'] * 100:.2f}% @ {b['description_length']:,} B")
    print(f"  {b['accuracy'] * K:.1f}x chance, {z:.0f} standard errors clear "
          f"(SE {se:.2f} pp on n={len(yte):,})")
    print(f"  verdict: {'DOES NOT FIRE' if z > 10 else 'FIRES'}")

    # The Pareto frontier, and what each point on it would need in order to pay.
    # Reading the MDL gap as accuracy rather than bytes is the useful form: the
    # byte gap grows with size, the accuracy gap turns out not to.
    front, best_acc = [], -1.0
    for r in sorted(recs, key=lambda r: (r["description_length"], -r["accuracy"])):
        if r["accuracy"] > best_acc:
            best_acc = r["accuracy"]
            front.append(r)
    print(f"\nPARETO FRONTIER ({len(front)} points), and the accuracy each size "
          f"would need to pay for its own transmission")
    print(f"{'artifact':<26}{'bytes':>8}{'src':>6}{'test':>8}"
          f"{'needed':>9}{'short by':>10}")
    for r in front:
        need = next((p for p in np.arange(0.05, 0.9995, 0.0001)
                     if r["description_length"] + price_labels(float(p), len(yte), K)
                     < budget), None)
        gap = "" if need is None else f"{(need - r['accuracy']) * 100:>9.1f} pp"
        print(f"{r['name']:<26}{r['description_length']:>8,}"
              f"{r.get('src_bytes', 0):>6}{r['accuracy'] * 100:>7.2f}%"
              f"{'  n/a' if need is None else f'{need * 100:>8.1f}%'}{gap}")

    print("\nMDL VERDICT")
    if payers:
        t, r = min(payers)
        print(f"  {len(payers)} artifact(s) pay for their own transmission; "
              f"best is {r['name']} at {t:,.0f} B against a {budget:,} B budget "
              f"({budget - t:,.0f} B under)")
    else:
        gap = min(r["description_length"] + price_labels(r["accuracy"], len(yte), K)
                  for r in recs) - budget
        print(f"  nothing pays. Cheapest total is {gap:,.0f} B over the "
              f"{budget:,} B budget.")
        # What accuracy would the smallest artifact need?
        need = None
        for p in np.arange(0.10, 0.999, 0.0005):
            if small["description_length"] + price_labels(
                    float(p), len(yte), K) < budget:
                need = p
                break
        print(f"  for the {small['description_length']:,} B floor artifact to "
              f"pay it would need {need * 100:.1f}% top-1; it gets "
              f"{small['accuracy'] * 100:.2f}%."
              if need else "  no accuracy makes the floor artifact pay.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
