"""An oblivious table over b thresholded block-means: the 500-700 B band.

The conv-ridge family has a hard floor around 694 B — source plus a minimal
head — and at that size it scores 14.6%. It cannot enter the 500-700 B band at
all. The best artifact below 700 B is `lin-gray4-5b-symmetric`: 26.46% at 480 B.

This module builds the one object that can enter that band cheaply. Of the two
candidates (a plain axis-aligned decision tree, an oblivious table over b
thresholded bits) it builds the table, for reasons that are about *source*
bytes rather than accuracy:

  * The two are the same family — a table over b thresholded bits **is** an
    oblivious tree — and prior measurement puts them within 0.5 pp. So the
    choice is decided by everything other than accuracy.
  * A plain tree must ship its topology: per node a feature index, a threshold,
    and child pointers, plus a decoder that walks it. An oblivious table ships
    b feature indices, b thresholds, and a flat 2^b table, and its decoder is
    three straight-line numpy expressions with no traversal at all. At this
    size source is two-thirds of the artifact, so that is the deciding term.
  * The table's payload is a run of small integers, which is exactly the shape
    lzma monetizes; a tree's mixed index/threshold/pointer record is not.

The second-order trick from the MNIST golf precedent applies here and is
implemented: the *order* of the b bits is free to choose (permuting bit
significance permutes the table and changes not a single prediction), so it is
chosen for compressibility, and cells no training image reaches are filled to
extend runs rather than with a global constant.

Everything — grid, b, feature indices, thresholds — is selected on the
validation split from `tinycifar.data.load_dev()`. Test is scored once, at the
end, through `tinycifar.evaluate`.
"""

from __future__ import annotations

import argparse
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

# The decoder. `w` holds b feature indices, b thresholds, then the table.
# The feature expression is a block mean over a g x g grid of RGB, which costs
# no bytes beyond the literals in the reshape.
TEMPLATE = '''import numpy as np
B=np.frombuffer(open(__file__[:-10]+"w","rb").read(),np.uint8)
def predict(x):
 v=x.reshape(-1,{G},{S},{G},{S},3).mean((2,4)).reshape(len(x),-1)
 return B[{OFF}+(v[:,B[:{B}]]>B[{B}:{OFF}])@(1<<np.arange({B}))]
'''

# Nibble-packed variant: half the raw table, but the packed nibbles compress
# far less well. Which one wins is measured, not assumed.
TEMPLATE_NIB = '''import numpy as np
B=np.frombuffer(open(__file__[:-10]+"w","rb").read(),np.uint8)
def predict(x):
 v=x.reshape(-1,{G},{S},{G},{S},3).mean((2,4)).reshape(len(x),-1)
 i=(v[:,B[:{B}]]>B[{B}:{OFF}])@(1<<np.arange({B}))
 return B[{OFF}+(i>>1)]>>(i%2*4)&15
'''

# A 16x16 grid has 768 features, so its indices do not fit in a byte. Paying
# for uint16 indices costs ~30 source bytes; whether the finer grid is worth
# them is a question for the sweep, not for the author.
TEMPLATE16 = '''import numpy as np
B=np.frombuffer(open(__file__[:-10]+"w","rb").read(),np.uint8)
F=np.frombuffer(B,np.uint16,{B})
def predict(x):
 v=x.reshape(-1,{G},{S},{G},{S},3).mean((2,4)).reshape(len(x),-1)
 return B[{OFF}+(v[:,F]>B[{FT}:{OFF}])@(1<<np.arange({B}))]
'''

TEMPLATE16_NIB = '''import numpy as np
B=np.frombuffer(open(__file__[:-10]+"w","rb").read(),np.uint8)
F=np.frombuffer(B,np.uint16,{B})
def predict(x):
 v=x.reshape(-1,{G},{S},{G},{S},3).mean((2,4)).reshape(len(x),-1)
 i=(v[:,F]>B[{FT}:{OFF}])@(1<<np.arange({B}))
 return B[{OFF}+(i>>1)]>>(i%2*4)&15
'''


# --------------------------------------------------------------------------
# features and candidate splits
# --------------------------------------------------------------------------

def feats(x: np.ndarray, g: int) -> np.ndarray:
    """Block-mean the image to a g x g RGB grid, exactly as the artifact does."""
    s = 32 // g
    return x.reshape(-1, g, s, g, s, 3).mean((2, 4)).reshape(len(x), -1)


def candidates(F: np.ndarray, n_thresh: int) -> np.ndarray:
    """Integer thresholds per feature, at evenly spaced quantiles.

    Thresholds are rounded to integers here rather than at emit time so that
    what is learned is exactly what ships.
    """
    qs = np.linspace(0, 100, n_thresh + 2)[1:-1]
    t = np.rint(np.percentile(F, qs, axis=0)).astype(np.int64)
    out = []
    for f in range(F.shape[1]):
        for v in np.unique(t[:, f]):
            if 0 <= v < 255:
                out.append((f, int(v)))
    return np.array(out, dtype=np.int64)


# --------------------------------------------------------------------------
# greedy oblivious tree
# --------------------------------------------------------------------------

def build_levels(F: np.ndarray, y: np.ndarray, cand: np.ndarray, depth: int):
    """Grow an oblivious tree level by level, greedily.

    At each level one (feature, threshold) pair is applied to *every* cell at
    once; the pair chosen maximizes the number of training points that fall in
    a cell whose majority label is their own. Returns the list of chosen pairs,
    which is nested: the first k are exactly the depth-k tree.
    """
    idx = np.zeros(len(y), dtype=np.int64)
    live = np.ones(len(cand), dtype=bool)
    chosen = []
    for level in range(depth):
        base = idx * 20 + y
        best, best_score = None, -1
        for c in np.flatnonzero(live):
            f, t = cand[c]
            key = base + 10 * (F[:, f] > t)
            cnt = np.bincount(key, minlength=(1 << level) * 20).reshape(-1, 10)
            score = int(cnt.max(1).sum())
            if score > best_score:
                best_score, best = score, c
        f, t = int(cand[best, 0]), int(cand[best, 1])
        chosen.append((f, t))
        idx = idx * 2 + (F[:, f] > t)
        live[best] = False
    return chosen


def refine(F: np.ndarray, y: np.ndarray, pairs, cand: np.ndarray, passes: int):
    """Coordinate descent on the b split tests.

    An oblivious tree's partition is the conjunction of b independent binary
    tests, so nothing about it depends on the order they were greedily picked
    in — which means each test can be re-chosen in place, holding the other
    b-1 fixed. Greedy level-wise growth is myopic; this repairs the early
    levels once the later ones are known. Scored on the fit split only.
    """
    pairs = list(pairs)
    b = len(pairs)
    for _ in range(passes):
        changed = False
        for k in range(b):
            rest = [p for j, p in enumerate(pairs) if j != k]
            other = cell_index(F, rest) if rest else np.zeros(len(y), np.int64)
            base = other * 20 + y
            taken = set(rest)
            best, best_score = pairs[k], -1
            for c in range(len(cand)):
                f, t = int(cand[c, 0]), int(cand[c, 1])
                if (f, t) in taken:
                    continue
                cnt = np.bincount(base + 10 * (F[:, f] > t),
                                  minlength=(1 << (b - 1)) * 20).reshape(-1, 10)
                score = int(cnt.max(1).sum())
                if score > best_score:
                    best_score, best = score, (f, t)
            if best != pairs[k]:
                changed = True
                pairs[k] = best
        if not changed:
            break
    return pairs


def cell_index(F: np.ndarray, pairs, order=None) -> np.ndarray:
    """Cell index under a bit ordering. `order[k]` is the significance of bit k."""
    b = len(pairs)
    f = np.array([p[0] for p in pairs])
    t = np.array([p[1] for p in pairs])
    w = np.zeros(b, dtype=np.int64)
    w[np.arange(b) if order is None else np.asarray(order)] = 1 << np.arange(b)
    return (F[:, f] > t) @ w


def fill_table(idx: np.ndarray, y: np.ndarray, b: int) -> np.ndarray:
    """Majority label per cell; unreached cells take their nearest populated
    neighbour in Hamming distance, which both predicts better than a global
    constant and lengthens runs for the compressor."""
    cnt = np.bincount(idx * 10 + y, minlength=(1 << b) * 10).reshape(-1, 10)
    table = cnt.argmax(1).astype(np.uint8)
    seen = cnt.sum(1) > 0
    if seen.all():
        return table
    table[~seen] = int(np.bincount(y).argmax())
    # one Hamming hop at a time, so a cell inherits from the closest neighbour
    cells = np.arange(1 << b)
    for _ in range(b):
        if seen.all():
            break
        votes = np.zeros((1 << b, 10), dtype=np.int64)
        for k in range(b):
            nb = cells ^ (1 << k)
            m = seen[nb]
            votes[m, table[nb][m]] += 1
        newly = (~seen) & (votes.sum(1) > 0)
        table[newly] = votes[newly].argmax(1).astype(np.uint8)
        seen = seen | newly
    return table


# --------------------------------------------------------------------------
# artifact emission
# --------------------------------------------------------------------------

def make_files(g: int, pairs, order, table: np.ndarray, nibble: bool):
    b = len(pairs)
    f = np.array([p[0] for p in pairs], dtype=np.int64)
    t = np.array([p[1] for p in pairs], dtype=np.int64)
    # `order` permutes which bit is which power of two; fold it into the order
    # the (feature, threshold) pairs are stored in, so the decoder stays a
    # plain dot with 1<<arange and costs nothing for it. `cell_index` gives
    # pair order[k] the weight 2**k, so stored position k holds pair order[k].
    o = np.asarray(order)
    wide = int(f.max()) > 255
    if wide:
        payload = f[o].astype("<u2").tobytes() + t[o].astype(np.uint8).tobytes()
        tmpl = TEMPLATE16_NIB if nibble else TEMPLATE16
        off = 3 * b
    else:
        payload = f[o].astype(np.uint8).tobytes() + t[o].astype(np.uint8).tobytes()
        tmpl = TEMPLATE_NIB if nibble else TEMPLATE
        off = 2 * b
    tb = ((table[0::2] | (table[1::2] << 4)).astype(np.uint8).tobytes()
          if nibble else table.tobytes())
    src = tmpl.format(G=g, S=32 // g, B=b, FT=2 * b, OFF=off)
    return {"predict.py": src.encode(), "w": payload + tb}


def predict_local(files, x: np.ndarray, g: int, b: int, nibble: bool):
    """Mirror of the emitted decoder, for sweeping without a subprocess."""
    B = np.frombuffer(files["w"], np.uint8)
    wide = b"np.uint16" in files["predict.py"]
    F = np.frombuffer(files["w"], np.uint16, b) if wide else B[:b]
    off = 3 * b if wide else 2 * b
    v = feats(x, g)
    i = (v[:, F] > B[off - b:off]) @ (1 << np.arange(b))
    if nibble:
        return B[off + (i >> 1)] >> (i % 2 * 4) & 15
    return B[off + i]


# --------------------------------------------------------------------------
# bit-order search, for compressibility only
# --------------------------------------------------------------------------

def best_order(g, pairs, Ffit, yfit, nibble, trials, rng):
    """Search bit orderings for the smallest artifact. A permutation of bit
    significance is a permutation of the table: predictions are bit-identical
    for every ordering, so this is a pure size lever and cannot leak anything
    about any split."""
    b = len(pairs)
    best = None
    orders = [list(range(b)), list(range(b))[::-1]]
    orders += [list(rng.permutation(b)) for _ in range(trials)]
    plain = None
    for o in orders:
        idx = cell_index(Ffit, pairs, o)
        table = fill_table(idx, yfit, b)
        files = make_files(g, pairs, o, table, nibble)
        n = A.measure(files).description_length
        if plain is None:
            plain = n                       # identity order, for the record
        if best is None or n < best[0]:
            best = (n, o, table, files)
    return best + (plain,)


# --------------------------------------------------------------------------
# sweep
# --------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grids", nargs="*", type=int, default=[4, 8, 16])
    ap.add_argument("--depth", type=int, default=13)
    ap.add_argument("--thresholds", type=int, default=7)
    ap.add_argument("--bmin", type=int, default=4)
    ap.add_argument("--refine", type=int, default=0,
                    help="coordinate-descent passes over the b split tests")
    ap.add_argument("--orders", type=int, default=24)
    ap.add_argument("--ceiling", type=int, default=700)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--final", default="", help="emit+score this name on test")
    a = ap.parse_args(argv)

    rng = np.random.default_rng(a.seed)
    xfit, yfit, xval, yval = load_dev()
    print(f"fit {xfit.shape[0]}  val {xval.shape[0]}")

    rows = []
    for g in a.grids:
        t0 = time.perf_counter()
        Ffit = feats(xfit, g)  # val goes through predict_local, not features
        cand = candidates(Ffit, a.thresholds)
        pairs = build_levels(Ffit, yfit, cand, a.depth)
        print(f"grid {g}: {len(cand)} candidates, depth {a.depth} grown in "
              f"{time.perf_counter() - t0:.0f}s")
        for b in range(a.bmin, a.depth + 1):
            p = pairs[:b]
            if a.refine:
                t1 = time.perf_counter()
                p = refine(Ffit, yfit, p, cand, a.refine)
                print(f"  g{g} b={b}: refined in {time.perf_counter() - t1:.0f}s")
            for nibble in (False, True):
                n, order, table, files, plain = best_order(
                    g, p, Ffit, yfit, nibble, a.orders, rng)
                acc = float((predict_local(files, xval, g, b, nibble) == yval).mean())
                rows.append(dict(grid=g, b=b, nibble=nibble, bytes=n, val=acc,
                                 pairs=p, order=order, files=files, plain=plain))
                print(f"  g{g} b={b:2d} {'nib' if nibble else 'u8 '} "
                      f"{n:5d} B  val {acc * 100:5.2f}%   "
                      f"(identity bit order {plain} B)")

    under = [r for r in rows if r["bytes"] <= a.ceiling]
    under.sort(key=lambda r: (-r["val"], r["bytes"]))
    print(f"\nbest under {a.ceiling} B on val:")
    for r in under[:8]:
        print(f"  g{r['grid']} b={r['b']} {'nib' if r['nibble'] else 'u8'} "
              f"{r['bytes']} B  {r['val'] * 100:.2f}%")

    if a.final and under:
        r = under[0]
        name = a.final
        d = emit(name, r["files"])
        rec = evaluate(
            d, name=name, split="val", save=False,
            method="oblivious table over thresholded block means")
        print(f"\nval through the harness: {summarize(rec)}")
        rec = evaluate(
            d, name=name, split="test",
            method=f"oblivious table: {r['b']} thresholded {r['grid']}x{r['grid']} "
                   f"RGB block means -> 2^{r['b']} label table",
            notes=f"selected on val; bit order chosen for compressibility "
                  f"({'nibble' if r['nibble'] else 'uint8'} table)")
        print(f"TEST: {summarize(rec)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
