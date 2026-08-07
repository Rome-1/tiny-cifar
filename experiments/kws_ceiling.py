"""What can the front end reach before quantization? The ceiling diagnostic.

The first `kws_ridge.py` sweep came back non-monotonic in bit width -- 3-bit
scoring below 2-bit and far below 4-bit -- which is the signature of a quantizer
failing, not of a front end failing. Before any effort goes into the quantizer
it is worth knowing what there is to recover, so this script fits the same ridge
head in float and reports the ceiling.

It ships nothing and measures no bytes. It exists to separate two questions the
sweep confounds: *are these features informative* and *does a global codebook
survive on them*.

Every number is on the dataset's own validation split except the one column
marked test, which is scored once per declared configuration.

Run:
    python3 experiments/kws_ceiling.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from experiments.kws_ridge import (  # noqa: E402
    K, compile_feat, feat_src, features, fit,
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(REPO / "data" / "sc12.npz"))
    ap.add_argument("--kinds", nargs="*", default=["mel", "oct", "env"])
    ap.add_argument("--frames", nargs="*", type=int, default=[16, 32, 49])
    ap.add_argument("--bands", nargs="*", type=int, default=[8, 12, 16, 20, 26])
    ap.add_argument("--lam", nargs="*", type=float,
                    default=[1e0, 1e1, 1e2, 1e3, 1e4, 1e5])
    a = ap.parse_args(argv)

    z = np.load(a.data)
    xtr, ytr = z["xtrain"], z["ytrain"]
    xva, yva = z["xval"], z["yval"]
    xte, yte = z["xtest"], z["ytest"]
    print(f"train {len(xtr):,}  val {len(xva):,}  test {len(xte):,}  "
          f"chance {100 / K:.2f}%\n")

    print(f"{'kind':<6}{'F':>4}{'M':>4}{'dim':>6}{'lam':>9}"
          f"{'val':>9}{'test':>9}{'head B @4b':>12}")
    print("-" * 60)
    rows = []
    for kind in a.kinds:
        for frames in a.frames:
            for bands in a.bands:
                if kind == "env" and bands != a.bands[0]:
                    continue          # env has no band axis
                src = feat_src(kind, frames, bands)
                f = compile_feat(src)
                Vtr, Vva, Vte = (features(f, x) for x in (xtr, xva, xte))
                D = Vtr.shape[1]
                best = max(
                    ((float((np.argmax(
                        Vva @ (W := fit(Vtr, ytr, lam))[:-1] + W[-1], 1) == yva
                    ).mean()), lam, W) for lam in a.lam),
                    key=lambda t: t[0])
                vacc, lam, W = best
                tacc = float((np.argmax(Vte @ W[:-1] + W[-1], 1) == yte).mean())
                rows.append((kind, frames, bands, D, lam, vacc, tacc))
                print(f"{kind:<6}{frames:>4}{bands:>4}{D:>6}{lam:>9g}"
                      f"{vacc * 100:>8.2f}%{tacc * 100:>8.2f}%"
                      f"{(D + 1) * K * 4 / 8:>12,.0f}")

    b = max(rows, key=lambda r: r[5])
    print(f"\nbest on val: {b[0]} F={b[1]} M={b[2]} D={b[3]} -> "
          f"val {b[5] * 100:.2f}%, test {b[6] * 100:.2f}%")
    print("This is an upper bound on what any quantized artifact over these "
          "features can score. It ships nothing and is not a Pareto point.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
