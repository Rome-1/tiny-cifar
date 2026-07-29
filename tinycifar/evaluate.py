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

# Stage one runs in its own process and never holds the labels. Everything the
# artifact can reach is either its own directory or the images it was handed.
_DRIVER = r'''
import json, os, sys, sysconfig, time
import numpy as np

ART, IMGS, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
x = np.load(IMGS)["x"]

# Past this point the artifact may read nothing outside its own directory.
# Without this a 111-byte artifact opens the dataset and reports 100%.
_allow = [os.path.realpath(ART),
          os.path.realpath(OUT + ".npy"),
          os.path.realpath(OUT + ".npy.meta.json")]
_tmp = os.path.realpath(os.path.dirname(IMGS))
# Fixed interpreter roots only. sys.path would admit whatever PYTHONPATH
# happens to hold, which on this box included another crew's source tree.
for p in list(sysconfig.get_paths().values()) + [sys.prefix, sys.base_prefix,
                                                 np.__path__[0]]:
    if p and os.path.isdir(p) and os.path.realpath(p) != _tmp:
        _allow.append(os.path.realpath(p))

# An audit hook is telemetry, not a sandbox: os.popen forks a child that
# inherits no hook, and ctypes calls libc directly. A 323-byte artifact using
# os.popen scored 100% against the previous version of this file. Escape routes
# are refused outright; the process has no legitimate need for any of them.
_BANNED = ("subprocess.Popen", "os.system", "os.exec", "os.posix_spawn",
           "os.spawn", "os.fork", "os.forkpty", "pty.spawn",
           "ctypes.dlopen", "ctypes.dlsym", "ctypes.call_function",
           "ctypes.addressof", "ctypes.create_string_buffer",
           "socket.__new__", "socket.connect", "urllib.Request",
           "mmap.__new__", "os.putenv", "os.truncate")

def _audit(event, args):
    if event in _BANNED or event.startswith(("subprocess.", "ctypes.", "socket.")):
        raise PermissionError("artifact attempted " + event)
    if event in ("open", "os.open"):
        p = args[0]
        if isinstance(p, (str, bytes, os.PathLike)):
            try:
                rp = os.path.realpath(os.fsdecode(p))
            except Exception:
                return
            if not any(rp == a or rp.startswith(a + os.sep) for a in _allow):
                raise PermissionError("artifact read outside its directory: " + rp)

sys.addaudithook(_audit)
sys.argv = ["predict"]          # leave no dataset paths lying around
sys.path.insert(0, ART)
import predict as M

t0 = time.perf_counter()
p = np.asarray(M.predict(x)).reshape(-1)
dt = time.perf_counter() - t0

# predict() is handed the whole test set at once, so nothing structural stops
# an artifact from clustering it, self-training on its own confident guesses, or
# exploiting the known class balance — all of which raise accuracy at zero
# shipped bytes and are not comparable to a per-image inference model. Re-run a
# shuffled subset: any method whose prediction for an image depends on the other
# images in the batch disagrees with itself here.
sub = np.random.default_rng(12345).permutation(len(x))[:1000]
q = np.asarray(M.predict(x[sub])).reshape(-1)
consistent = int((q == p[sub]).sum())

np.save(OUT, p.astype(np.int64))
json.dump({"inference_seconds": dt, "n": int(p.shape[0]),
           "batch_consistent": consistent, "batch_checked": len(sub)},
          open(OUT + ".npy.meta.json", "w"))
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

    if violations:
        # Previously this was recorded and the artifact still ranked, so an
        # artifact could import a framework, load a pretrained network, and land
        # on the board at 200 bytes with a warning emoji next to it.
        raise ValueError(
            f"{path}: not self-contained — " + "; ".join(violations))

    size = A.measure(files)
    blob = A.serialize(files)

    xtr, ytr, xte, yte = load()
    x, y = (xtr, ytr) if split == "train" else (xte, yte)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        unpacked = td / "artifact"
        unpacked.mkdir()
        A.write_dir(A.deserialize(blob), unpacked)   # round trip through bytes

        # The sandboxed process is handed images and nothing else.
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
            raise RuntimeError(
                f"artifact failed to run:\n{proc.stderr[-3000:]}"
            )
        p = np.load(str(out) + ".npy")
        meta = json.loads((td / "preds.npy.meta.json").read_text())

    if p.shape[0] != y.shape[0]:
        raise ValueError(f"predict returned {p.shape[0]} labels for {y.shape[0]} images")

    consistent = meta["batch_consistent"] / meta["batch_checked"]
    if consistent < 1.0:
        raise ValueError(
            f"{path}: predictions depend on batch composition "
            f"({consistent:.1%} agreement on a shuffled subset) — transductive "
            "methods are not comparable to per-image inference")

    scored = {
        "accuracy": float((p == y).mean()),
        "n": int(y.shape[0]),
        "inference_seconds": meta["inference_seconds"],
        "per_class": [float((p[y == c] == c).mean()) for c in range(10)],
    }

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
