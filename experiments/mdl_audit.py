"""Price the existing frontier in MDL currency.

The model-artifact leaderboard reports two numbers that cannot be added: bytes
of model and percent correct. In the Hutter framing they *can* be added, because
a classifier is a compressor of labels, and then a blunt question becomes
askable: **does a model pay for itself?**

    two-part total = artifact bytes + arithmetic-coded label bytes

The baseline to beat is transmitting the 10,000 test labels with no model at
all: 10000 * log2(10) / 8 = 4,152 bytes. A model earns its place only if it
saves more label bytes than it costs to ship. That gives every point on the
board a budget it must live inside, and it is not obvious our points do.

Cross-entropy is measured on the *quantized* weights actually shipped, with a
temperature fitted on the training set — the decoder holds the training set too,
so it can repeat that fit exactly, and without it ridge margins are not logits
and would code far worse than they should.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from experiments.conv_features import (  # noqa: E402
    build_feats, fit_head, make_feats_src,
)
from experiments.quant_sweep import lloyd_max  # noqa: E402
from tinycifar.data import load  # noqa: E402

UNIFORM = 10000 * np.log2(10) / 8


def quantize_percol(W, bits):
    """Reconstruct W exactly as the shipped per-class-codebook artifact does."""
    out = np.empty_like(W)
    for j in range(W.shape[1]):
        c, i = lloyd_max(W[:, j], bits)
        out[:, j] = c.astype(np.float32)[i]
    return out


def quantize_global(W, bits):
    """One codebook over the whole head — what the sub-2 KB artifacts ship."""
    c, i = lloyd_max(W, bits)
    return c.astype(np.float32)[i].reshape(W.shape)


def artifact_accuracy(name: str):
    p = REPO / "results" / f"{name}.json"
    return json.loads(p.read_text())["accuracy"] if p.exists() else None


def fit_temperature(logits, y, grid=np.geomspace(0.5, 400.0, 60)):
    """Pick the temperature minimising cross-entropy. Deterministic, so the
    decoder can repeat it."""
    best, temp = np.inf, 1.0
    for t in grid:
        z = t * logits
        z -= z.max(1, keepdims=True)
        ce = float(np.mean(np.log(np.exp(z).sum(1)) - z[np.arange(len(z)), y]))
        if ce < best:
            best, temp = ce, float(t)
    return temp, best / np.log(2)


def cross_entropy_bits(logits, y, temp):
    z = temp * logits
    z -= z.max(1, keepdims=True)
    ce = np.log(np.exp(z).sum(1)) - z[np.arange(len(z)), y]
    return float(ce.mean() / np.log(2))


def artifact_bytes(name: str) -> int | None:
    p = REPO / "results" / f"{name}.json"
    if p.exists():
        return json.loads(p.read_text())["description_length"]
    return None


def audit(k, bits, patch, stride, pool, seed, lam, data, artifact_name=None,
          quant="percol"):
    xtr, ytr, xte, yte = data
    src, dim = make_feats_src(k, patch, stride, pool, seed,
                              1.0 / np.sqrt(patch * patch * 3), 0.1)
    feats = build_feats(src)

    t0 = time.perf_counter()
    W = fit_head(feats, xtr, ytr, dim, lam, tta=True)
    Wq = (quantize_global if quant == "global" else quantize_percol)(W, bits)

    def logits_for(imgs):
        F = np.vstack([
            feats(imgs[i:i + 500].astype(np.float32) / 255)
            + feats(imgs[i:i + 500].astype(np.float32)[:, :, ::-1] / 255)
            for i in range(0, len(imgs), 500)
        ])
        return np.hstack([F, np.ones((len(F), 1))]) @ Wq

    ltr = logits_for(xtr[:10000])
    temp, tr_ce = fit_temperature(ltr, ytr[:10000])
    lte = logits_for(xte)
    ce = cross_entropy_bits(lte, yte, temp)
    acc = float((lte.argmax(1) == yte).mean())

    label_bytes = ce * len(yte) / 8
    art = artifact_bytes(artifact_name) if artifact_name else None

    # The refit must reproduce the artifact it is being priced against. This
    # check exists because it once did not: the audit refit with per-class
    # codebooks while the named 961 B artifact ships a global one, and the table
    # paired one model's size with another model's accuracy.
    stored = artifact_accuracy(artifact_name) if artifact_name else None
    if stored is not None and abs(stored - acc) > 0.015:
        raise ValueError(
            f"{artifact_name}: refit scores {acc:.4f} but the shipped artifact "
            f"scores {stored:.4f} — the audit is not pricing this artifact "
            f"(check the quantization scheme)")

    return {
        "k": k, "bits": bits, "dim": dim, "acc": acc,
        "bits_per_label": ce, "label_bytes": label_bytes,
        "artifact_bytes": art,
        "total": (art + label_bytes) if art else None,
        "temp": temp, "train_ce": tr_ce,
        "seconds": time.perf_counter() - t0,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs", nargs="*", default=[
        "6,4,artifacts:cf-k6-p4s2-4b", "16,4,artifacts:cf-k16-p4s2-4b-pc",
        "64,4,artifacts:cf-k64-p4s2-4b", "128,6,artifacts:cf-k128-p4s2-6b",
    ], help="k,bits,artifacts:NAME")
    ap.add_argument("--patch", type=int, default=4)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--pool", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--lam", type=float, default=1e2)
    a = ap.parse_args(argv)

    data = load()
    print(f"baseline: {UNIFORM:,.0f} B to send 10,000 labels with no model\n")
    print(f"{'model':>18} {'acc':>7} {'bits/lab':>9} {'labels':>9} "
          f"{'artifact':>9} {'total':>9}  verdict")
    print("-" * 78)

    rows = []
    for spec in a.configs:
        k, bits, name = spec.split(",")
        name = name.split(":", 1)[1] if ":" in name else None
        quant = "global" if (name and not name.endswith("-pc")) else "percol"
        r = audit(int(k), int(bits), a.patch, a.stride, a.pool, a.seed,
                  a.lam, data, name, quant=quant)
        rows.append(r)
        tot = r["total"]
        verdict = "—"
        if tot:
            verdict = f"PAYS ({UNIFORM - tot:+,.0f} B)" if tot < UNIFORM \
                else f"costs {tot - UNIFORM:,.0f} B more"
        print(f"{name or '?':>18} {r['acc'] * 100:6.2f}% {r['bits_per_label']:9.3f} "
              f"{r['label_bytes']:9,.0f} {str(r['artifact_bytes'] or '—'):>9} "
              f"{tot if tot else 0:9,.0f}  {verdict}")

    best = min((r for r in rows if r["total"]), key=lambda r: r["total"], default=None)
    if best:
        print(f"\nbest two-part total: {best['total']:,.0f} B "
              f"(k={best['k']}, {best['bits']}-bit)")
        budget = UNIFORM - min(r["label_bytes"] for r in rows)
        print(f"largest artifact that could still pay for itself: {budget:,.0f} B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
