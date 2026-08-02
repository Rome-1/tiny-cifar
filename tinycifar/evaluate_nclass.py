"""Evaluate an artifact against an arbitrary labelled image set.

`tinycifar.evaluate` hardcodes CIFAR-10: it loads the ten-class dataset itself
and hardcodes ten in its per-class breakdown. The class-scaling measurement
needs the same harness pointed at 10, 100 and 1000 classes, so this module
takes the images and labels as arguments instead of loading them.

Everything that makes the harness a harness is *imported*, not copied:
`_DRIVER` (the sandbox, the audit hook, the batch-consistency probe) and
`artifact` (serialization, the size metric, the import check) are the same
objects the CIFAR path uses. If the sandbox is tightened there, it is tightened
here in the same commit. Only the dataset plumbing and the per-class arithmetic
are new.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from . import artifact as A
from .evaluate import _DRIVER

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"


def evaluate_arrays(
    path: str | Path,
    x: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    name: str | None = None,
    split: str = "test",
    method: str = "",
    notes: str = "",
    train_seconds: float | None = None,
    save: bool = False,
    timeout: int = 3600,
) -> dict:
    """Measure, verify and score an artifact on (x, y). Same contract as
    `tinycifar.evaluate.evaluate`, with the dataset passed in.

    `save` defaults to False here: the class-scaling sweep writes hundreds of
    points and `results/` is the CIFAR leaderboard's input.
    """
    path = Path(path)
    name = name or path.stem
    files = A.read_dir(path)

    if "predict.py" not in files:
        raise ValueError(f"{path}: artifact has no predict.py")
    violations = A.check_imports(files)
    if violations:
        raise ValueError(f"{path}: not self-contained — " + "; ".join(violations))

    size = A.measure(files)
    blob = A.serialize(files)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        unpacked = td / "artifact"
        unpacked.mkdir()
        A.write_dir(A.deserialize(blob), unpacked)   # round trip through bytes

        imgs = td / "images.npz"
        np.savez(imgs, x=x)

        driver = td / "_driver.py"
        driver.write_text(_DRIVER)
        out = td / "preds"

        env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
        t0 = time.perf_counter()
        proc = subprocess.run(
            ["nice", "-n", "15", sys.executable, str(driver),
             str(unpacked), str(imgs), str(out)],
            capture_output=True, text=True, timeout=timeout, env=env,
        )
        wall = time.perf_counter() - t0
        if proc.returncode != 0:
            raise RuntimeError(f"artifact failed to run:\n{proc.stderr[-3000:]}")
        p = np.load(str(out) + ".npy")
        meta = json.loads((td / "preds.npy.meta.json").read_text())

    if p.shape[0] != y.shape[0]:
        raise ValueError(
            f"predict returned {p.shape[0]} labels for {y.shape[0]} images")

    consistent = meta["batch_consistent"] / meta["batch_checked"]
    if consistent < 1.0:
        raise ValueError(
            f"{path}: predictions depend on batch composition "
            f"({consistent:.1%} agreement on a shuffled subset) — transductive "
            "methods are not comparable to per-image inference")

    if p.min() < 0 or p.max() >= n_classes:
        raise ValueError(
            f"{path}: predicted labels outside [0,{n_classes}) "
            f"(saw {p.min()}..{p.max()})")

    acc = float((p == y).mean())
    # Top-1 only. Reporting top-5 at 1000 classes would be the standard
    # ImageNet courtesy, but the CIFAR frontier is top-1 and mixing the two
    # would make the curve incomparable to the board it is meant to extend.
    record = {
        "name": name,
        "method": method,
        "notes": notes,
        "split": split,
        "n_classes": n_classes,
        "accuracy": acc,
        "chance": 1.0 / n_classes,
        "n": int(y.shape[0]),
        "size": size.as_dict(),
        "description_length": size.description_length,
        "inference_seconds": round(meta["inference_seconds"], 2),
        "eval_wall_seconds": round(wall, 2),
        "train_seconds": train_seconds,
        "self_contained": not violations,
        "violations": violations,
        "files": {k: len(v) for k, v in sorted(files.items())},
    }
    if save:
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / f"{name}.json").write_text(json.dumps(record, indent=2))
    return record


def summarize(r: dict) -> str:
    return (
        f"{r['name']}: {r['accuracy'] * 100:.2f}% on {r['n']} {r['split']} "
        f"images, {r['n_classes']} classes (chance {r['chance'] * 100:.2f}%)"
        f"  |  {r['description_length']:,} B ({r['size']['best_codec']})"
    )
