"""CIFAR-10 loading.

The dataset is NOT part of any artifact's description length — it is the signal
we are modeling, available to both encoder and decoder. Only the model artifact
is measured.
"""

from __future__ import annotations

import pickle
import tarfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
RAW = DATA / "cifar-10-batches-py"
CACHE = DATA / "cifar10.npz"

CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


def _ensure_extracted() -> None:
    if RAW.exists():
        return
    archive = DATA / "cifar-10-python.tar.gz"
    if not archive.exists():
        raise FileNotFoundError(
            f"{archive} missing. Fetch it from "
            "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        )
    with tarfile.open(archive) as tf:
        tf.extractall(DATA)


def _load_batch(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as fh:
        d = pickle.load(fh, encoding="bytes")
    x = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y = np.array(d[b"labels"], dtype=np.int64)
    return np.ascontiguousarray(x, dtype=np.uint8), y


def from_parquet(train: Path, test: Path) -> None:
    """Build the npz cache from HuggingFace `uoft-cs/cifar10` parquet files.

    The canonical toronto.edu tarball is served slowly enough (~1 MB/min from
    this box) to be unusable; the HF mirror is the same data over a fast CDN.
    Images are stored there as PNG bytes, so they are decoded here once.
    """
    import io

    import pyarrow.parquet as pq
    from PIL import Image

    def decode(path: Path):
        t = pq.read_table(path)
        blobs = t.column("img").combine_chunks().field("bytes").to_pylist()
        y = np.array(t.column("label").to_pylist(), dtype=np.int64)
        x = np.stack([
            np.asarray(Image.open(io.BytesIO(b)).convert("RGB"), dtype=np.uint8)
            for b in blobs
        ])
        if x.shape[1:] != (32, 32, 3):
            raise ValueError(f"{path}: unexpected image shape {x.shape[1:]}")
        return x, y

    xtr, ytr = decode(train)
    xte, yte = decode(test)
    if len(xtr) != 50000 or len(xte) != 10000:
        raise ValueError(f"expected 50000/10000, got {len(xtr)}/{len(xte)}")

    DATA.mkdir(exist_ok=True)
    np.savez(CACHE, xtr=xtr, ytr=ytr, xte=xte, yte=yte)


def load(cache: bool = True):
    """Return (x_train, y_train, x_test, y_test).

    Images are uint8 [N, 32, 32, 3] in HWC order; labels int64 [N].
    """
    if cache and CACHE.exists():
        z = np.load(CACHE)
        return z["xtr"], z["ytr"], z["xte"], z["yte"]

    _ensure_extracted()
    xs, ys = [], []
    for i in range(1, 6):
        x, y = _load_batch(RAW / f"data_batch_{i}")
        xs.append(x)
        ys.append(y)
    xtr, ytr = np.concatenate(xs), np.concatenate(ys)
    xte, yte = _load_batch(RAW / "test_batch")

    if cache:
        DATA.mkdir(exist_ok=True)
        np.savez(CACHE, xtr=xtr, ytr=ytr, xte=xte, yte=yte)
    return xtr, ytr, xte, yte


def load_flat(dtype=np.float32, scale: float = 1 / 255.0):
    """Same as `load` but images flattened to [N, 3072] and scaled."""
    xtr, ytr, xte, yte = load()
    f = lambda a: (a.reshape(len(a), -1).astype(dtype) * scale)
    return f(xtr), ytr, f(xte), yte


if __name__ == "__main__":
    xtr, ytr, xte, yte = load()
    print(f"train {xtr.shape} {xtr.dtype}  test {xte.shape}")
    print("class balance (test):", np.bincount(yte))
