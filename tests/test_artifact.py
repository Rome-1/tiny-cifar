"""Tests for the size metric — the number everything else is judged against."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tinycifar import artifact as A  # noqa: E402


def test_roundtrip():
    files = {"predict.py": b"x = 1\n", "w.bin": os.urandom(1000), "a/b.txt": b""}
    assert A.deserialize(A.serialize(files)) == files


def test_container_overhead_is_small():
    """The whole point of a hand-rolled container: negligible overhead at 1 KB."""
    payload = os.urandom(1000)
    blob = A.serialize({"w.bin": payload})
    overhead = len(blob) - len(payload)
    assert overhead < 16, f"{overhead} B of container overhead is too much"


def test_deterministic_and_order_independent():
    a = A.serialize({"a": b"1", "b": b"2"})
    b = A.serialize({"b": b"2", "a": b"1"})
    assert a == b


def test_varint_multibyte():
    """Files over 127 bytes exercise the continuation path."""
    for n in (0, 1, 127, 128, 300, 70000):
        files = {"w": b"\x00" * n}
        assert A.deserialize(A.serialize(files))["w"] == files["w"]


def test_description_length_is_min_and_never_exceeds_raw():
    incompressible = {"w.bin": os.urandom(5000)}
    compressible = {"w.bin": b"\x00" * 5000}

    s = A.measure(incompressible)
    assert s.description_length == min(s.raw, s.gzip, s.xz)
    assert s.description_length <= s.raw

    c = A.measure(compressible)
    assert c.description_length < 200, "zeros should compress to nearly nothing"
    assert c.best_codec in ("gzip", "xz")


def test_random_bytes_barely_compress():
    """Guards the metric against a codec that somehow 'wins' on noise."""
    s = A.measure({"w.bin": os.urandom(4096)})
    assert s.description_length >= 4000


def test_measure_matches_a_real_quantized_tensor():
    rng = np.random.default_rng(0)
    w = rng.integers(-1, 2, size=8192, dtype=np.int8)      # ternary
    packed = np.packbits((w + 1).astype(np.uint8).reshape(-1, 1) & 1)
    s = A.measure({"w.bin": packed.tobytes()})
    assert s.description_length <= len(packed) + 16


def test_import_check_flags_torch():
    bad = A.check_imports({"predict.py": b"import torch\n"})
    assert bad and "torch" in bad[0]


def test_import_check_allows_numpy_and_siblings():
    files = {
        "predict.py": b"import numpy as np\nfrom decode import go\nimport lzma\n",
        "decode.py": b"def go():\n    return 1\n",
    }
    assert A.check_imports(files) == []


def test_import_check_flags_from_import():
    bad = A.check_imports({"predict.py": b"from sklearn.svm import SVC\n"})
    assert bad and "sklearn" in bad[0]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
