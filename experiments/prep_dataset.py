"""Build a class-scaling npz (xtr,ytr,xva,yva,xte,yte) from a labelled corpus.

Two sources, one output format. The output is what `class_scaling.py` eats, and
the splits are fixed here so that no downstream script can quietly re-carve them.

Conventions, all fixed before any number was measured:

* **Nested classes.** Label id `c` means the same class at every K, and the
  K-class problem is exactly the images of classes `[0, K)`. The 10-class run
  therefore sees a strict subset of the 100-class run's images, which is what
  makes the comparison a class-count comparison and not a dataset comparison.
* **`--test-per-class 50`.** Fifty test images per class makes the two-part MDL
  budget `N log2 K` equal at K=10 with 10,000 images and at K=100 with 5,000 —
  both 4,152 B (see `experiments/label_stream.py`). Holding the label stream
  fixed while K moves is the only clean control available here.
* **Validation is carved from train**, per class, and never from test.

CIFAR-100 (`--source cifar100`) is the fully-open leg: it is served ungated from
the HuggingFace mirror of the Toronto tarball, 32x32, and spans K = 10 to 100
on one decade of class count.

Downsampled ImageNet (`--source imagenet-npz`) is the leg that would extend that
to 1000. It reads Chrabaszcz et al.'s `train_data_batch_*` / `val_data` pickles
if and only if they are already on disk; it downloads nothing, because the
source is behind a terms-of-access gate that this script has no business
negotiating.
"""

from __future__ import annotations

import argparse
import io
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def _from_parquet(train: Path, test: Path, label_col: str):
    import pyarrow.parquet as pq
    from PIL import Image

    def decode(path: Path):
        t = pq.read_table(path)
        img_col = "img" if "img" in t.column_names else "image"
        blobs = t.column(img_col).combine_chunks().field("bytes").to_pylist()
        y = np.array(t.column(label_col).to_pylist(), dtype=np.int64)
        x = np.stack([
            np.asarray(Image.open(io.BytesIO(b)).convert("RGB"), dtype=np.uint8)
            for b in blobs
        ])
        if x.shape[1:] != (32, 32, 3):
            raise ValueError(f"{path}: unexpected image shape {x.shape[1:]}")
        return x, y

    return decode(train), decode(test)


def load_cifar100(label_col: str = "fine_label"):
    from huggingface_hub import snapshot_download
    p = Path(snapshot_download("uoft-cs/cifar100", repo_type="dataset",
                               allow_patterns=["*.parquet"]))
    tr = next(p.rglob("train-*.parquet"))
    te = next(p.rglob("test-*.parquet"))
    return _from_parquet(tr, te, label_col)


def load_imagenet_hf(per_class: int, n_classes: int, seed: int = 0):
    """Downsampled ImageNet 32x32 from the ungated HuggingFace repack.

    Repo: benjamin-paine/imagenet-1k-32x32 (`gated: false`; card declares
    `license: other`, `license_details: imagenet-agreement`, and reproduces the
    ImageNet Terms of Access in full). Those terms permit non-commercial
    research and education, which is what this is. Nothing here works around a
    gate: the repo has none, and the files resolve anonymously.

    Only `per_class` training images per class are decoded. The label column is
    read first and the selection made on row indices, so the ~780k images we do
    not use are never decoded — that is most of the wall clock and most of the
    peak RAM. Validation (50 per class, all of it) is decoded whole.
    """
    import io

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download, list_repo_files
    from PIL import Image

    repo = "benjamin-paine/imagenet-1k-32x32"
    files = [f for f in list_repo_files(repo, repo_type="dataset")
             if f.endswith(".parquet")]
    g = np.random.default_rng(seed)

    def decode_rows(t, rows):
        blobs = t.column("image").combine_chunks().field("bytes")
        out = np.empty((len(rows), 32, 32, 3), np.uint8)
        for i, r in enumerate(rows):
            im = Image.open(io.BytesIO(blobs[int(r)].as_py())).convert("RGB")
            a = np.asarray(im, dtype=np.uint8)
            if a.shape != (32, 32, 3):
                raise ValueError(f"unexpected image shape {a.shape}")
            out[i] = a
        return out

    # --- train: label-first selection, then decode only what we keep
    xs, ys = [], []
    kept = np.zeros(n_classes, np.int64)
    for f in sorted(x for x in files if "/train-" in x):
        p = Path(hf_hub_download(repo, f, repo_type="dataset"))
        t = pq.read_table(p, columns=["label"])
        lab = np.asarray(t.column("label").to_pylist(), dtype=np.int64)
        take = []
        for c in np.unique(lab):
            if c >= n_classes or kept[c] >= per_class:
                continue
            idx = np.flatnonzero(lab == c)
            idx = idx[g.permutation(len(idx))][: per_class - kept[c]]
            kept[c] += len(idx)
            take.append(idx)
        if not take:
            continue
        take = np.sort(np.concatenate(take))
        full = pq.read_table(p)
        xs.append(decode_rows(full, take))
        ys.append(lab[take])
        del full
        print(f"  {f}: kept {len(take):,} (total {sum(map(len, ys)):,})")
    xtr, ytr = np.concatenate(xs), np.concatenate(ys)

    # --- validation: this is the held-out set ImageNet ships, 50 per class
    vf = [x for x in files if "/validation-" in x][0]
    t = pq.read_table(Path(hf_hub_download(repo, vf, repo_type="dataset")))
    lab = np.asarray(t.column("label").to_pylist(), dtype=np.int64)
    rows = np.flatnonzero(lab < n_classes)
    return (xtr, ytr), (decode_rows(t, rows), lab[rows])


def load_imagenet_npz(root: Path):
    """Chrabaszcz et al. downsampled ImageNet 32x32 pickles, already on disk.

    Labels in those files are 1..1000; they are shifted to 0..999 here.
    """
    def rd(paths):
        xs, ys = [], []
        for f in paths:
            with open(f, "rb") as fh:
                d = pickle.load(fh)
            x = np.asarray(d["data"], dtype=np.uint8)
            x = x.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            xs.append(np.ascontiguousarray(x))
            ys.append(np.asarray(d["labels"], dtype=np.int64) - 1)
        return np.concatenate(xs), np.concatenate(ys)

    tr = sorted(root.glob("train_data_batch_*"))
    va = sorted(root.glob("val_data*"))
    if not tr or not va:
        raise FileNotFoundError(
            f"{root}: no train_data_batch_* / val_data. This script does not "
            "download ImageNet; see docs/imagenet-derisk.md for why.")
    return rd(tr), rd(va)


def build(train, test, n_classes: int, val_per_class: int,
          test_per_class: int, seed: int = 0) -> dict:
    (xtr, ytr), (xte, yte) = train, test
    g = np.random.default_rng(seed)

    def take(x, y, per_class, rest=False):
        keep, other = [], []
        for c in range(n_classes):
            idx = np.flatnonzero(y == c)
            idx = idx[g.permutation(len(idx))]
            keep.append(idx[:per_class])
            other.append(idx[per_class:])
        keep = np.concatenate(keep)
        other = np.concatenate(other)
        return (x[keep], y[keep], x[other], y[other]) if rest else (x[keep], y[keep])

    xva, yva, xfit, yfit = take(xtr, ytr, val_per_class, rest=True)
    xt, yt = take(xte, yte, test_per_class)

    o = g.permutation(len(xfit))
    return {"xtr": xfit[o], "ytr": yfit[o], "xva": xva, "yva": yva,
            "xte": xt, "yte": yt}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True,
                    choices=["cifar100", "cifar100-coarse",
                             "imagenet-hf", "imagenet-npz"])
    ap.add_argument("--root", type=Path, help="for imagenet-npz: directory of pickles")
    ap.add_argument("--train-per-class", type=int, default=500,
                    help="imagenet-hf: images per class to decode from train. "
                         "500 matches CIFAR-100 exactly, which is the point.")
    ap.add_argument("--classes", type=int, required=True)
    ap.add_argument("--val-per-class", type=int, default=50)
    ap.add_argument("--test-per-class", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    if a.source == "cifar100":
        train, test = load_cifar100("fine_label")
    elif a.source == "cifar100-coarse":
        train, test = load_cifar100("coarse_label")
    elif a.source == "imagenet-hf":
        train, test = load_imagenet_hf(a.train_per_class, a.classes, a.seed)
    else:
        train, test = load_imagenet_npz(a.root)

    n_avail = int(max(train[1].max(), test[1].max())) + 1
    if a.classes > n_avail:
        raise ValueError(f"asked for {a.classes} classes, source has {n_avail}")

    d = build(train, test, a.classes, a.val_per_class, a.test_per_class, a.seed)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, **d)
    print(f"{a.out}: fit {d['xtr'].shape} val {d['xva'].shape} "
          f"test {d['xte'].shape}, {a.classes} classes")
    import math
    print(f"  uniform label stream over test: "
          f"{len(d['yte']) * math.log2(a.classes) / 8:,.1f} B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
