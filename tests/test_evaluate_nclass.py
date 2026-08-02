"""The K-class evaluator must enforce everything the CIFAR one enforces.

`evaluate_nclass` exists so the class-scaling measurement can point the harness
at 100 and 1000 classes. It reuses `evaluate._DRIVER` rather than copying it,
and these tests are what keeps that reuse honest: if the sandbox stops refusing
a dataset read, or the batch-consistency probe stops firing, this file fails.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tinycifar.evaluate_nclass import evaluate_arrays  # noqa: E402

K = 100
N = 40


def _write(tmp_path: Path, src: str) -> Path:
    d = tmp_path / "art"
    d.mkdir()
    (d / "predict.py").write_text(src)
    return d


@pytest.fixture
def images():
    g = np.random.default_rng(0)
    x = g.integers(0, 256, (N, 32, 32, 3), dtype=np.uint8)
    y = np.arange(N) % K
    return x, y


def test_constant_artifact_scores_and_measures(tmp_path, images):
    x, y = images
    d = _write(tmp_path, "import numpy as np\n"
                         "def predict(x):\n return np.zeros(len(x),np.int64)\n")
    r = evaluate_arrays(d, x, y, K, name="c")
    assert r["n_classes"] == K
    assert r["chance"] == pytest.approx(1 / K)
    assert r["accuracy"] == pytest.approx((y == 0).mean())
    assert r["description_length"] == r["size"]["description_length"]
    assert r["self_contained"]


def test_out_of_range_label_is_rejected(tmp_path, images):
    x, y = images
    d = _write(tmp_path, "import numpy as np\n"
                         f"def predict(x):\n return np.full(len(x),{K},np.int64)\n")
    with pytest.raises(ValueError, match="outside"):
        evaluate_arrays(d, x, y, K, name="oob")


def test_batch_dependent_prediction_is_rejected(tmp_path, images):
    x, y = images
    # Prediction depends on the other images in the batch — the transductive
    # loophole the CIFAR harness closes, closed here too.
    d = _write(tmp_path, "import numpy as np\n"
                         "def predict(x):\n"
                         " return (np.arange(len(x))+len(x))%100\n")
    with pytest.raises(ValueError, match="batch composition"):
        evaluate_arrays(d, x, y, K, name="batch")


def test_non_numpy_import_is_rejected(tmp_path, images):
    x, y = images
    d = _write(tmp_path, "import torch\n"
                         "def predict(x):\n return [0]*len(x)\n")
    with pytest.raises(ValueError, match="not self-contained"):
        evaluate_arrays(d, x, y, K, name="torch")
