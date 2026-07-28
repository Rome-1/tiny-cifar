"""Evaluate an artifact and record a Pareto point.

Evaluation deliberately goes through the measured bytes: the artifact is
serialized, the serialization is unpacked into an empty temporary directory,
and inference runs there in a fresh subprocess. If a submission depends on
anything that did not survive that round trip, it fails here rather than
quietly inflating the leaderboard.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from . import artifact as A
from .data import CACHE, load

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"

_DRIVER = r'''
import json, sys, time
import numpy as np
sys.path.insert(0, sys.argv[1])
import predict as M

z = np.load(sys.argv[2])
x, y = z["xte"], z["yte"]
if len(sys.argv) > 4 and sys.argv[4] == "train":
    x, y = z["xtr"], z["ytr"]

t0 = time.perf_counter()
p = np.asarray(M.predict(x)).reshape(-1)
dt = time.perf_counter() - t0
if p.shape[0] != y.shape[0]:
    raise SystemExit(f"predict returned {p.shape[0]} labels for {y.shape[0]} images")

json.dump({
    "accuracy": float((p == y).mean()),
    "n": int(y.shape[0]),
    "inference_seconds": dt,
    "per_class": [float((p[y == c] == c).mean()) for c in range(10)],
}, open(sys.argv[3], "w"))
'''


def evaluate(
    path: str | Path,
    name: str | None = None,
    split: str = "test",
    method: str = "",
    notes: str = "",
    train_seconds: float | None = None,
    save: bool = True,
    timeout: int = 3600,
) -> dict:
    """Measure, verify and score an artifact. Returns the result record."""
    path = Path(path)
    name = name or path.stem
    files = A.read_dir(path)

    if "predict.py" not in files:
        raise ValueError(f"{path}: artifact has no predict.py")
    violations = A.check_imports(files)

    size = A.measure(files)
    blob = A.serialize(files)

    if not CACHE.exists():          # materialize the npz the driver reads
        load()

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        unpacked = td / "artifact"
        unpacked.mkdir()
        A.write_dir(A.deserialize(blob), unpacked)   # round trip through bytes

        driver = td / "_driver.py"
        driver.write_text(_DRIVER)
        out = td / "out.json"

        env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
        t0 = time.perf_counter()
        proc = subprocess.run(
            ["nice", "-n", "15", sys.executable, str(driver),
             str(unpacked), str(CACHE), str(out), split],
            capture_output=True, text=True, timeout=timeout, env=env,
        )
        wall = time.perf_counter() - t0
        if proc.returncode != 0:
            raise RuntimeError(
                f"artifact failed to run:\n{proc.stderr[-3000:]}"
            )
        scored = json.loads(out.read_text())

    record = {
        "name": name,
        "method": method,
        "notes": notes,
        "split": split,
        "accuracy": scored["accuracy"],
        "n": scored["n"],
        "per_class": scored["per_class"],
        "size": size.as_dict(),
        "description_length": size.description_length,
        "inference_seconds": round(scored["inference_seconds"], 2),
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
    flag = "" if r["self_contained"] else "  ** NOT SELF-CONTAINED **"
    return (
        f"{r['name']}: {r['accuracy'] * 100:.2f}% on {r['n']} {r['split']} images"
        f"  |  {r['description_length']:,} B"
        f" ({r['size']['best_codec']}; raw {r['size']['raw']:,}"
        f" / gz {r['size']['gzip']:,} / xz {r['size']['xz']:,}){flag}"
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", help="artifact directory or single predict.py")
    ap.add_argument("--name")
    ap.add_argument("--method", default="")
    ap.add_argument("--notes", default="")
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--train-seconds", type=float)
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args(argv)

    r = evaluate(
        a.path, name=a.name, split=a.split, method=a.method, notes=a.notes,
        train_seconds=a.train_seconds, save=not a.no_save,
    )
    print(summarize(r))
    for v in r["violations"]:
        print(f"  ! {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
