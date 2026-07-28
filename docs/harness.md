# The harness — how size is measured and how to submit a model

Read this before building a Pareto point. The metric is the whole project; a
method that wins by accident of the container is not a win.

## The artifact contract

An **artifact** is a directory containing `predict.py`, which exposes:

```python
def predict(x):        # x: uint8 [N, 32, 32, 3], HWC, unnormalized
    ...                # returns: int labels [N]
```

It may read sibling files via paths relative to `__file__`, and nothing else.
The **declared runtime is CPython 3 + numpy** — see `ALLOWED_IMPORTS` in
`tinycifar/artifact.py`. Importing anything outside it (torch, sklearn, scipy)
fails the self-containment check, because a model that leans on a framework has
not accounted for the bits that framework supplies.

Train with whatever you like — torch, a GPU, hours of compute. Only what ships
in the artifact is measured. Training-time tricks are free.

## What the size number means

```
description_length = min(raw, gzip, xz)   of the canonical serialization
```

**The canonical serialization** (`artifact.serialize`) is a minimal
concatenation: for each file, sorted by name, `varint(len(name)) name
varint(len(data)) data`, after a 3-byte magic. Overhead is a handful of bytes
per file.

This is hand-rolled for a reason. `tar` pads every member to a 512-byte block
and appends a 1024-byte terminator; `zip` spends ~90 bytes of header per entry.
At a 1 KB Pareto point that container overhead would be a large fraction of the
measurement — we would be reporting the archive format, not the model.

**Why the minimum over three codecs.** All three decoders already live in the
declared runtime, so a submission could always ship whichever encoding is
smallest and expand it on the way in; charging more than that would be
fictitious. Reporting only `raw` would flatter a method that left obvious
redundancy on the table. Reporting only a compressed size would punish a method
that did its own entropy coding well — precisely the methods this project is
most interested in. The minimum is neutral between them.

All three are reported on the leaderboard regardless, because the gap between
them is diagnostic: a large `raw - xz` gap means there are bits left on the
table.

The `gzip`/`xz` figures use raw deflate/LZMA2 streams plus 4 bytes of length,
not the file-format wrappers, for the same reason — the wrapper is not part of
the model.

## Code counts

The generated `predict.py` is measured along with the weights. This is not
pedantry: without it, any amount of model could be smuggled into source as
literals, and the "smallest model" would be a decoder with the answers baked in.

The practical consequence is that at the small end **the decoder source is a
leading line item.** In a 400-byte artifact, 250 bytes of Python is most of the
budget. Write terse `predict.py`. Prefer one file over several. This is why
`experiments/baselines.py` generates cramped source — it is not sloppiness.

## What is free

The asymmetry to exploit: **anything a short piece of code can generate costs
nothing.** Downsampling, graying, a fixed DCT or Hadamard basis, a PRNG-seeded
random projection — all are a line of numpy in the artifact and reduce the
weights that must actually be shipped. The baselines already lean on this:
block-mean pooling to 4x4 grayscale cuts the weight matrix 192x for about
sixty bytes of source.

The dataset is also free — it is the signal being modeled, available to both
sides. Only the artifact is measured.

## Submitting a point

```python
from tinycifar.evaluate import evaluate, summarize

r = evaluate("artifacts/my-model",
             method="ternary micro-CNN, entropy-coded",
             notes="anything worth knowing",
             train_seconds=612.0)
print(summarize(r))
```

`evaluate` serializes the artifact, unpacks it from those bytes into an empty
temporary directory, and runs inference there in a fresh subprocess. If a model
depends on anything that did not survive that round trip, it fails here rather
than quietly scoring well. Results land in `results/<name>.json`.

Then rebuild the board:

```bash
python -m tinycifar.leaderboard      # writes LEADERBOARD.md
```

## Reusable pieces

- `tinycifar/pack.py` — uniform affine quantization at any width 1–16 bits, and
  dense bit packing. `roundtrip_error(w, bits)` tells you what a bit width costs
  before you retrain. Note that 1-bit wants `symmetric=True`.
- `tinycifar/data.py` — `load()` for uint8 HWC arrays, `load_flat()` for scaled
  and flattened.
- `tinycifar/artifact.py` — `measure()` any directory or file map; run it as
  `python -m tinycifar.artifact artifacts/foo` for a quick size read.

## Rules of the road

- Report accuracy on the full 10,000-image test set. Tune on a split of train.
- Box discipline: `nice -n 15`, cap parallelism, check `uptime` before anything
  heavy. The box is shared and has frozen before.
- Bead every experiment under `tc-s22`.
