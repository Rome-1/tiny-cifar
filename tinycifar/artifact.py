"""Artifact packing and the size metric.

An *artifact* is a set of files that, together with a declared runtime, maps
CIFAR-10 test images to predicted labels. Its **description length in bytes is
the metric we minimize.**

The contract
------------
An artifact directory must contain ``predict.py`` exposing::

    predict(x: np.ndarray) -> np.ndarray

where ``x`` is uint8 ``[N, 32, 32, 3]`` (HWC, unnormalized) and the return is
int labels ``[N]``. It may read sibling files in the artifact via paths relative
to ``__file__``. It may not read anything else.

The declared runtime is CPython 3 + numpy. Nothing else. In particular the
weights of the model may not hide inside a framework the artifact imports, and
the artifact may not fetch anything at inference time.

Why serialize by hand
---------------------
Off-the-shelf archives are unusable at our scale: ``tar`` pads every member to a
512-byte block and appends 1024 bytes of terminator, so a 900-byte model
"weighs" 2 KB before compression, and ``zip`` spends ~90 bytes of header per
member. At a 1 KB Pareto point that container overhead would dominate the thing
being measured. So the canonical serialization below is a minimal concatenation
whose only overhead is a length prefix and the filename itself.

The headline number
-------------------
``description_length`` is the **minimum** over {raw, gzip, xz} of the canonical
serialization. Taking the minimum is the principled choice: every one of those
decoders already lives in the declared runtime, so a submission could always
ship whichever encoding is smallest and unpack it on the way in. Reporting only
the raw size would flatter methods that leave obvious redundancy on the table;
reporting only the compressed size would penalize a method that already did its
own entropy coding well. The minimum rewards neither artifact of the container.
"""

from __future__ import annotations

import ast
import lzma
import zlib
from dataclasses import dataclass, asdict
from pathlib import Path

# The declared runtime: CPython stdlib + numpy. An artifact importing anything
# outside this is not self-contained, and its true description length would have
# to include whatever it imported.
ALLOWED_IMPORTS = {
    "numpy", "np",
    "math", "cmath", "struct", "array", "gzip", "lzma", "zlib", "bz2", "base64",
    "binascii", "pickle", "json", "itertools", "functools", "operator",
    "collections", "os", "sys", "io", "pathlib", "random", "hashlib", "typing",
}

MAGIC = b"TC1"


# --------------------------------------------------------------------------
# canonical serialization
# --------------------------------------------------------------------------

def _uvarint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _read_uvarint(buf: bytes, i: int) -> tuple[int, int]:
    n, shift = 0, 0
    while True:
        b = buf[i]
        i += 1
        n |= (b & 0x7F) << shift
        if not b & 0x80:
            return n, i
        shift += 7


def serialize(files: dict[str, bytes]) -> bytes:
    """Deterministic minimal container: MAGIC, then per file (sorted by name)
    varint(len(name)) name varint(len(data)) data."""
    out = bytearray(MAGIC)
    for name in sorted(files):
        data = files[name]
        nb = name.encode()
        out += _uvarint(len(nb)) + nb + _uvarint(len(data)) + data
    return bytes(out)


def deserialize(blob: bytes) -> dict[str, bytes]:
    if not blob.startswith(MAGIC):
        raise ValueError("not a tinycifar artifact")
    i, files = len(MAGIC), {}
    while i < len(blob):
        n, i = _read_uvarint(blob, i)
        name = blob[i:i + n].decode()
        i += n
        n, i = _read_uvarint(blob, i)
        files[name] = blob[i:i + n]
        i += n
    return files


def read_dir(path: str | Path) -> dict[str, bytes]:
    path = Path(path)
    if path.is_file():
        return {path.name: path.read_bytes()}
    return {
        str(p.relative_to(path)): p.read_bytes()
        for p in sorted(path.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts
    }


def write_dir(files: dict[str, bytes], path: str | Path) -> Path:
    path = Path(path)
    for name, data in files.items():
        dest = path / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    return path


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Size:
    raw: int
    gzip: int
    xz: int
    n_files: int

    @property
    def description_length(self) -> int:
        """The headline metric: bytes of the shortest shippable encoding."""
        return min(self.raw, self.gzip, self.xz)

    @property
    def best_codec(self) -> str:
        return min(
            (("raw", self.raw), ("gzip", self.gzip), ("xz", self.xz)),
            key=lambda kv: kv[1],
        )[0]

    def as_dict(self) -> dict:
        d = asdict(self)
        d["description_length"] = self.description_length
        d["best_codec"] = self.best_codec
        return d

    def __str__(self) -> str:
        return (
            f"{self.description_length:,} B ({self.best_codec}) "
            f"[raw {self.raw:,} / gz {self.gzip:,} / xz {self.xz:,}]"
        )


def _gzip_min(blob: bytes) -> int:
    """gzip at level 9, minus the header/trailer that a self-extracting
    artifact would not need to ship. Uses raw deflate + 4 bytes of length."""
    return len(zlib.compress(blob, 9)) + 4


def _xz_min(blob: bytes) -> int:
    filt = [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}]
    return len(lzma.compress(blob, format=lzma.FORMAT_RAW, filters=filt)) + 4


def measure(target: str | Path | dict[str, bytes]) -> Size:
    """Measure an artifact directory, file, or in-memory file map."""
    files = target if isinstance(target, dict) else read_dir(target)
    blob = serialize(files)
    return Size(
        raw=len(blob),
        gzip=_gzip_min(blob),
        xz=_xz_min(blob),
        n_files=len(files),
    )


# --------------------------------------------------------------------------
# self-containment check
# --------------------------------------------------------------------------

def check_imports(files: dict[str, bytes]) -> list[str]:
    """Return a list of violations: imports outside the declared runtime."""
    bad = []
    for name, data in files.items():
        if not name.endswith(".py"):
            continue
        try:
            tree = ast.parse(data.decode())
        except (SyntaxError, UnicodeDecodeError) as e:
            bad.append(f"{name}: unparseable ({e})")
            continue
        local = {n.rsplit(".", 1)[0] for n in files if n.endswith(".py")}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                mods = [node.module or ""]
            else:
                continue
            for m in mods:
                root = m.split(".")[0]
                if root and root not in ALLOWED_IMPORTS and root not in local:
                    bad.append(f"{name}: imports '{m}' (outside declared runtime)")
    return bad


if __name__ == "__main__":
    import sys

    for arg in sys.argv[1:]:
        files = read_dir(arg)
        print(f"{arg}: {measure(files)}")
        for v in check_imports(files):
            print(f"  ! {v}")
