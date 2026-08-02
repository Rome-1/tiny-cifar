"""Verify generated artifacts decode and run correctly, without needing CIFAR.

The failure this guards against is subtle and expensive: a generated
`predict.py` whose feature expression or weight-decode disagrees with what the
trainer assumed. That mismatch would not crash — it would quietly produce a
plausible-looking but wrong accuracy.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments import baselines as B  # noqa: E402
from tinycifar import artifact as A  # noqa: E402


def _fake_images(n=32, seed=0):
    return np.random.default_rng(seed).integers(
        0, 256, size=(n, 32, 32, 3), dtype=np.uint8
    )


def _load_predict(src: str, blob: bytes, tmp):
    """Exec a generated predict.py the way the artifact runner would."""
    d = tmp / "art"
    d.mkdir(exist_ok=True)
    (d / "w").write_bytes(blob)
    (d / "predict.py").write_text(src)
    ns = {"__file__": str(d / "predict.py")}
    exec(compile(src, "predict.py", "exec"), ns)  # noqa: S102
    return ns


def test_feature_dims_match_declaration():
    """Every FEATURES entry must really produce the dimension it claims."""
    x = _fake_images()
    for name, (dim, expr) in B.FEATURES.items():
        f = eval(expr, {"np": np}, {"x": x.astype(np.float32)})
        assert f.shape == (len(x), dim), f"{name}: {f.shape} != (N,{dim})"
        assert 0.0 <= f.min() and f.max() <= 1.0, f"{name} not in [0,1]"


def test_artifact_reconstructs_weights(tmp_path=None):
    import pathlib
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        rng = np.random.default_rng(0)
        D = B.FEATURES["gray4"][0] + 1
        W = rng.standard_normal((D, 10)).astype(np.float64)

        blob = B.build_weight_blob(W, 8)
        src = B.TEMPLATE.format(D=D, EXPR=B.FEATURES["gray4"][1])
        ns = _load_predict(src, blob, tmp)

        err = np.abs(ns["W"] - W).max() / np.abs(W).max()
        assert err < 0.02, f"weight decode error {err}"


def test_predictions_match_the_trainers_own_math():
    """The artifact's forward pass must equal featurize() @ W offline."""
    import pathlib
    import tempfile

    x = _fake_images(64, seed=3)
    for feat in ("gray4", "rgb8", "gray16"):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            D, expr = B.FEATURES[feat]
            rng = np.random.default_rng(1)
            W = rng.standard_normal((D + 1, 10))

            blob = B.build_weight_blob(W, 8)
            ns = _load_predict(B.TEMPLATE.format(D=D + 1, EXPR=expr), blob, tmp)

            got = ns["predict"](x)
            want = np.argmax(B.featurize(x, expr) @ ns["W"], 1)
            agree = (got == want).mean()
            assert agree == 1.0, f"{feat}: artifact disagrees ({agree:.2%})"


def test_predict_returns_valid_labels():
    import pathlib
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        D, expr = B.FEATURES["gray8"]
        W = np.random.default_rng(2).standard_normal((D + 1, 10))
        ns = _load_predict(
            B.TEMPLATE.format(D=D + 1, EXPR=expr),
            B.build_weight_blob(W, 4), tmp,
        )
        p = ns["predict"](_fake_images(16))
        assert p.shape == (16,) and p.min() >= 0 and p.max() < 10


def test_generated_artifacts_are_self_contained():
    D, expr = B.FEATURES["gray8"]
    src = B.TEMPLATE.format(D=D + 1, EXPR=expr)
    files = {"predict.py": src.encode(), "w": b"\x00" * 100}
    assert A.check_imports(files) == []
    assert A.check_imports({"predict.py": B.CONSTANT_TEMPLATE.format(c=3).encode()}) == []


def test_tiny_model_actually_fits_in_a_kilobyte():
    """The headline claim of the small end of the sweep."""
    D, expr = B.FEATURES["gray4"]
    files = {
        "predict.py": B.TEMPLATE.format(D=D + 1, EXPR=expr).encode(),
        "w": B.build_weight_blob(np.random.default_rng(0).standard_normal(
            (D + 1, 10)), 4),
    }
    size = A.measure(files)
    assert size.description_length < 1024, size


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
