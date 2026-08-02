"""Test-time augmentation for the 470 B oblivious table — the hardest band.

This is where the "TTA is pure profit" claim breaks. The table's decoder is 209
bytes and the whole artifact is 470, so a hundred bytes of aggregation is a
fifth of the model. Worse, the table has no linear head to fold views into: it
emits a label, not a score, so combining T views means one-hot votes and a
`np.eye`, which is source the other two families never have to pay for.

The table also has no flip TTA to begin with — the incumbent is a single view —
so here even the first transform is an open question rather than an extra one.

Three things are measured, and only the third is TTA:

  1. the incumbent, one view, for reference;
  2. TTA over the shipped table, which changes only `predict.py`;
  3. the same table refit on flip-augmented training data, which changes only
     `w` and costs *zero* source bytes — the free lever TTA is competing with
     in this band.

Shifts are in pixels but the features are 4x4 block means, so a one-pixel roll
moves a quarter of one block. Shift sets at 1, 2 and 4 pixels are all swept for
that reason.

Selection is on `tinycifar.data.load_dev()`; test is scored once via `--emit`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from experiments.baselines import emit  # noqa: E402
from experiments.oblivious_table import cell_index, feats, fill_table  # noqa: E402
from experiments.tta_cnn import mcnemar, shift_repr  # noqa: E402
from tinycifar import artifact as A  # noqa: E402
from tinycifar.data import load_dev  # noqa: E402
from tinycifar.evaluate import evaluate, summarize  # noqa: E402

BASE = '''import numpy as np
B=np.frombuffer(open(__file__[:-10]+"w","rb").read(),np.uint8)
def predict(x):
 v=x.reshape(-1,{G},{S},{G},{S},3).mean((2,4)).reshape(len(x),-1)
 return B[{OFF}+(v[:,B[:{B}]]>B[{B}:{OFF}])@(1<<np.arange({B}))]
'''

VOTE = '''import numpy as np
B=np.frombuffer(open(__file__[:-10]+"w","rb").read(),np.uint8)
E=np.eye(10,10,0,int)
def g(x):
 v=x.reshape(-1,{G},{S},{G},{S},3).mean((2,4)).reshape(len(x),-1)
 return E[B[{OFF}+(v[:,B[:{B}]]>B[{B}:{OFF}])@(1<<np.arange({B}))]]
def predict(x):
 return sum(g(np.roll(x,d,(1,2)){F})for{M}d in {SH}).argmax(1)
'''

SHIFTSETS = {
    "1px": [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)],
    "2px": [(0, 0), (2, 0), (-2, 0), (0, 2), (0, -2)],
    "4px": [(0, 0), (4, 0), (-4, 0), (0, 4), (0, -4)],
    "2px9": [(i, j) for i in (-2, 0, 2) for j in (-2, 0, 2)],
    "4px9": [(i, j) for i in (-4, 0, 4) for j in (-4, 0, 4)],
    "none": [(0, 0)],
}


def source(g, b, shifts, flip):
    off = 2 * b
    if list(shifts) == [(0, 0)] and not flip:
        return BASE.format(G=g, S=32 // g, B=b, OFF=off)
    return VOTE.format(G=g, S=32 // g, B=b, OFF=off,
                       F="[:,:,::m]" if flip else "",
                       M=" m in(1,-1)for " if flip else " ",
                       SH=shift_repr(shifts))


def predict_local(blob, x, g, b, shifts, flip):
    B = np.frombuffer(blob, np.uint8)
    off = 2 * b
    w = 1 << np.arange(b)
    votes = np.zeros((len(x), 10), np.int32)
    for d in shifts:
        z = np.roll(x, d, (1, 2)) if d != (0, 0) else x
        for zz in ((z, z[:, :, ::-1]) if flip else (z,)):
            v = feats(zz, g)
            lab = B[off + (v[:, B[:b]] > B[b:off]) @ w]
            votes[np.arange(len(x)), lab] += 1
    return votes.argmax(1)


def refit_table(blob, g, b, xfit, yfit, flip_aug):
    """Same splits, same bit order; only the table's cell labels are refit."""
    B = np.frombuffer(blob, np.uint8)
    pairs = [(int(B[i]), int(B[b + i])) for i in range(b)]
    xs = [xfit, xfit[:, :, ::-1]] if flip_aug else [xfit]
    idx = np.concatenate([cell_index(feats(z, g), pairs) for z in xs])
    table = fill_table(idx, np.concatenate([yfit] * len(xs)), b)
    return bytes(B[:2 * b]) + table.tobytes()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifact", default="obt-g8-b9")
    ap.add_argument("-g", type=int, default=8)
    ap.add_argument("-b", type=int, default=9)
    ap.add_argument("--emit", default="", help="'<shiftset>,<flip>' to score on TEST")
    ap.add_argument("--name", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)

    art = REPO / "artifacts" / a.artifact
    blob = (art / "w").read_bytes()
    xfit, yfit, xva, yva = load_dev()

    # `refit-plain` is a control: it should reproduce the shipped table byte for
    # byte, which is the check that this module's idea of the decoder matches
    # the one that shipped.
    tables = {"shipped": blob,
              "refit-plain": refit_table(blob, a.g, a.b, xfit, yfit, False),
              "refit-flipaug": refit_table(blob, a.g, a.b, xfit, yfit, True)}
    print("  refit-plain reproduces the shipped table: "
          f"{tables['refit-plain'] == blob}")
    base = A.measure({"predict.py": source(a.g, a.b, [(0, 0)], False).encode(),
                      "w": blob}).description_length
    print(f"{a.artifact}: incumbent {base:,} B")

    rows, ref_ok = [], None
    for tname, tb in tables.items():
        for sname, shifts in SHIFTSETS.items():
            for flip in (False, True):
                src = source(a.g, a.b, shifts, flip)
                files = {"predict.py": src.encode(), "w": tb}
                n = A.measure(files).description_length
                ok = predict_local(tb, xva, a.g, a.b, shifts, flip) == yva
                nm = f"{tname}/{sname}{'+flip' if flip else ''}"
                if tname == "shipped" and sname == "none" and not flip:
                    ref_ok = ok
                rows.append(dict(name=nm, val=float(ok.mean()), bytes=n,
                                 delta_bytes=n - base, ok=ok, table=tname,
                                 shifts=sname, flip=flip, files=files,
                                 nviews=len(shifts) * (2 if flip else 1)))

    for r in rows:
        lo_, wi_, p = mcnemar(ref_ok, r["ok"])
        r.update(win=wi_, lose=lo_, p=p)
    for r in sorted(rows, key=lambda r: -r["val"]):
        print(f"  {r['name']:<24} {r['nviews']:2d} views  val {r['val'] * 100:6.2f}%  "
              f"{r['bytes']:>5,} B ({r['delta_bytes']:+4d})  "
              f"paired +{r['win']}/-{r['lose']} p={r['p']:.3g}")

    if a.emit:
        sname, flip = a.emit.split(",")[0], a.emit.split(",")[1] == "flip"
        tname = a.emit.split(",")[2] if len(a.emit.split(",")) > 2 else "shipped"
        r = next(x for x in rows if x["table"] == tname
                 and x["shifts"] == sname and x["flip"] == flip)
        nm = a.name or f"{a.artifact}-{sname}{'flip' if flip else ''}"
        d = emit(nm, r["files"])
        rec = evaluate(
            d, name=nm,
            method=f"oblivious table, {tname} table, {r['nviews']}-view TTA "
                   f"({sname}{', flip' if flip else ''})",
            notes=f"selected on val ({r['val'] * 100:.2f}%)")
        print(f"\nTEST: {summarize(rec)}")

    if a.out:
        Path(a.out).write_text(json.dumps(
            [{k: v for k, v in r.items() if k not in ("ok", "files")}
             for r in rows], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
