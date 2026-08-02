"""Test-time augmentation for the trained-CNN artifacts, priced in bytes.

The docs called TTA "pure profit until it saturates". That is wrong as stated.
Every extra transform is more source in `predict.py`, and source is charged at
the same rate as weights. At 69 KB a hundred bytes of source is noise; at 470 B
it is a fifth of the artifact. So the question is never "does TTA help" but
"does the gain beat the source bytes, at this size".

This module answers it for the CNN half of the frontier without retraining
anything. It execs an existing artifact's own `predict.py` to get the exact
forward pass that ships, scores every candidate view on the validation split,
and then re-emits `predict.py` with the winning aggregation so the byte cost is
measured rather than estimated. The weight file is untouched.

Shifts are `np.roll`, which wraps rather than pads. Wrapping is wrong at the
image border, and it is also fourteen characters. Whether the wrap costs more
accuracy than an edge-replicating pad saves in bytes is a question the sweep
answers rather than the author.

Selection is on `tinycifar.data.load_dev()`'s validation split. Test is scored
once, by `--emit`, through `tinycifar.evaluate`.
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

from experiments.baselines import emit  # noqa: E402
from tinycifar import artifact as A  # noqa: E402
from tinycifar.data import load_dev  # noqa: E402
from tinycifar.evaluate import evaluate, summarize  # noqa: E402

# The tail of a golfed trained-CNN decoder. Everything above `def predict` is
# the weight unpacker and the forward pass, and is reused verbatim.
TAIL_FLIP = '''def predict(x):
 return np.concatenate([np.argmax(fw(z)+fw(z[:,:,::-1]),1)for z in np.array_split(x.astype(np.float32)/255,-(-len(x)//250))])
'''

TAIL_PLAIN = '''def predict(x):
 return np.concatenate([np.argmax(fw(z),1)for z in np.array_split(x.astype(np.float32)/255,-(-len(x)//250))])
'''

# The general form: sum a per-shift aggregate over a list of (di, dj) shifts.
# `np.roll` wraps the border; the "edge" variant replicates it, which is more
# faithful to what the random-crop augmentation trained on and costs ~40 more
# source bytes. Both are emitted and measured.
TAIL_ROLL = '''def T(z):
 return {AGG}
def predict(x):
 return np.concatenate([np.argmax(sum(T(np.roll(z,d,(1,2)))for d in {S}),1)for z in np.array_split(x.astype(np.float32)/255,-(-len(x)//250))])
'''

TAIL_EDGE = '''def T(z):
 return {AGG}
def predict(x):
 return np.concatenate([np.argmax(sum(T(p[:,{R}+i:{R}+32+i,{R}+j:{R}+32+j])for i,j in {S}),1)for p in[np.pad(z,((0,0),({R},{R}),({R},{R}),(0,0)),"edge")for z in np.array_split(x.astype(np.float32)/255,-(-len(x)//250))]])
'''

AGG_FLIP = "fw(z)+fw(z[:,:,::-1])"
AGG_PLAIN = "fw(z)"


def shift_repr(shifts) -> str:
    return "[" + ",".join(f"({i},{j})" for i, j in shifts) + "]"


def make_tail(shifts, flip: bool, mode: str = "wrap") -> str:
    """Emit the smallest tail that expresses this view set."""
    if list(shifts) == [(0, 0)]:
        return TAIL_FLIP if flip else TAIL_PLAIN
    agg = AGG_FLIP if flip else AGG_PLAIN
    if mode == "wrap":
        return TAIL_ROLL.format(AGG=agg, S=shift_repr(shifts))
    r = max(max(abs(i), abs(j)) for i, j in shifts)
    return TAIL_EDGE.format(AGG=agg, S=shift_repr(shifts), R=r)


def retail(src: str, shifts, flip: bool, mode: str = "wrap") -> str:
    head = src.split("def predict(x):")[0]
    return head + make_tail(shifts, flip, mode)


def mcnemar(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    """Exact two-sided McNemar for two correctness vectors on the same images.

    The +1.06 pp selection-noise floor in the docs is about the *maximum* of an
    unpaired sweep. Two TTA schedules over identical weights and identical
    images are paired, and the paired test is what settles them; both are
    reported so neither is mistaken for the other.
    """
    from math import comb

    n01 = int((~a & b).sum())
    n10 = int((a & ~b).sum())
    n = n01 + n10
    if n == 0:
        return n10, n01, 1.0
    k = min(n01, n10)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    return n10, n01, p


def load_fw(art: Path):
    """Materialize the artifact's own forward pass, weights and all."""
    ns = {"__file__": str(art / "predict.py")}
    exec(compile((art / "predict.py").read_text(), "<artifact>", "exec"), ns)  # noqa: S102
    return ns["fw"]


def view_logits(fw, x, shift, flip, mode="wrap", chunk=250):
    """Logits for one view, in the artifact's own arithmetic."""
    out = np.empty((len(x), 10), np.float32)
    i0, j0 = shift
    r = max(abs(i0), abs(j0))
    for i in range(0, len(x), chunk):
        z = x[i:i + chunk].astype(np.float32) / 255
        if shift != (0, 0):
            if mode == "wrap":
                z = np.roll(z, shift, (1, 2))
            else:
                p = np.pad(z, ((0, 0), (r, r), (r, r), (0, 0)), "edge")
                z = p[:, r + i0:r + 32 + i0, r + j0:r + 32 + j0]
        if flip:
            z = z[:, :, ::-1]
        out[i:i + chunk] = fw(z)
    return out


# Candidate view sets, all nested around the incumbent single flip.
CROSS1 = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
CROSS2 = CROSS1 + [(2, 0), (-2, 0), (0, 2), (0, -2)]
DIAG1 = [(0, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
BOX1 = [(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)]
BOX2 = [(i, j) for i in (-2, -1, 0, 1, 2) for j in (-2, -1, 0, 1, 2)]

SCHEDULES = {
    "none": ([(0, 0)], False, "wrap"),
    "flip": ([(0, 0)], True, "wrap"),                    # the incumbent
    "cross1": (CROSS1, False, "wrap"),
    "flip+cross1": (CROSS1, True, "wrap"),
    "flip+diag1": (DIAG1, True, "wrap"),
    "flip+box1": (BOX1, True, "wrap"),
    "flip+cross2": (CROSS2, True, "wrap"),
    "flip+box2": (BOX2, True, "wrap"),
    "flip+cross1e": (CROSS1, True, "edge"),
    "flip+box1e": (BOX1, True, "edge"),
    "flip+box2e": (BOX2, True, "edge"),
}


def all_views(scheds) -> list:
    seen = []
    for name in scheds:
        shifts, _, mode = SCHEDULES[name]
        for s in shifts:
            key = (s, mode if s != (0, 0) else "wrap")
            if key not in seen:
                seen.append(key)
    return seen


def sweep(art: Path, scheds, x, y, quiet=False):
    fw = load_fw(art)
    src = (art / "predict.py").read_text()
    blob = (art / "w").read_bytes()
    base = A.measure({"predict.py": src.encode(), "w": blob}).description_length

    cache = {}
    t0 = time.perf_counter()
    for s, mode in all_views(scheds):
        for f in (False, True):
            cache[(s, mode, f)] = view_logits(fw, x, s, f, mode)
    if not quiet:
        print(f"  {2 * len(cache) // 2} views in {time.perf_counter() - t0:.0f}s")

    rows, ref_ok = [], None
    for name in scheds:
        shifts, flip, mode = SCHEDULES[name]
        lo = sum(cache[(s, mode if s != (0, 0) else "wrap", f)] for s in shifts
                 for f in ((False, True) if flip else (False,)))
        ok = lo.argmax(1) == y
        if name == "flip":
            ref_ok = ok
        files = {"predict.py": retail(src, shifts, flip, mode).encode(),
                 "w": blob}
        n = A.measure(files).description_length
        rows.append(dict(name=name, val=float(ok.mean()), bytes=n,
                         delta_bytes=n - base, ok=ok, mode=mode,
                         nviews=len(shifts) * (2 if flip else 1), files=files))
    for r in rows:
        lo_, wi_, p = mcnemar(ref_ok, r["ok"]) if ref_ok is not None else (0, 0, 1.0)
        r.update(win=wi_, lose=lo_, p=p)
    return rows, base


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("artifacts", nargs="+")
    ap.add_argument("--scheds", nargs="*", default=list(SCHEDULES))
    ap.add_argument("--emit", default="",
                    help="schedule to emit and score on TEST (once)")
    ap.add_argument("--suffix", default="-tta")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)

    _, _, xva, yva = load_dev()
    report = []
    for spec in a.artifacts:
        art = REPO / "artifacts" / Path(spec).name
        print(f"\n{art.name}")
        rows, base = sweep(art, a.scheds, xva, yva)
        ref = next(r for r in rows if r["name"] == "flip")
        for r in rows:
            print(f"  {r['name']:<13} {r['nviews']:2d} views  "
                  f"val {r['val'] * 100:6.2f}%  ({(r['val'] - ref['val']) * 100:+5.2f})  "
                  f"{r['bytes']:>7,} B  ({r['delta_bytes']:+4d})  "
                  f"paired +{r['win']}/-{r['lose']} p={r['p']:.3g}")
            report.append({k: v for k, v in r.items()
                           if k not in ("files", "ok")}
                          | {"artifact": art.name, "base_bytes": base})

        if a.emit:
            r = next(x for x in rows if x["name"] == a.emit)
            name = art.name + a.suffix
            d = emit(name, r["files"])
            rec = evaluate(
                d, name=name,
                method=f"{art.name} with {a.emit} TTA ({r['nviews']} views), "
                       "weights unchanged",
                notes=f"selected on val ({r['val'] * 100:.2f}%); "
                      f"source {len(r['files']['predict.py'])} B")
            print(f"  TEST: {summarize(rec)}")

    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
