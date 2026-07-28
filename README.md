# tiny-cifar

**Goal: the smallest models ever to run on CIFAR-10.** Optimize for **file size**
(bytes of the serialized, self-contained model artifact), *not* runtime. Build
the **size ↔ accuracy Pareto frontier** — aim for a **10 KB** point, and push
smaller. Rig prefix `tc-`. Status: exploratory (2026-07-28).

## Status

See **[LEADERBOARD.md](LEADERBOARD.md)** for the current frontier and
**[docs/harness.md](docs/harness.md)** for how size is measured and how to
submit a point. Method research is in [docs/method-survey.md](docs/method-survey.md)
(the enumerated families, plus what hlb-CIFAR10 has to teach) and
[docs/exotic-methods.md](docs/exotic-methods.md) (the non-obvious space).

```bash
python -m tinycifar.leaderboard          # rebuild the board from results/
python experiments/conv_features.py      # the current best family
python tests/test_artifact.py            # the metric's own tests
```

The frontier so far is built entirely from closed-form ridge regression — no
backprop anywhere — on features that cost nothing to ship. The single idea
carrying most of it: **a PRNG seed is four bytes no matter how much it draws**,
so random convolutional filters are free and only the classifier head is real
weight.

## Framing — this is a Minimum Description Length problem

The deliverable at each Pareto point is a self-contained artifact (quantized
weights + architecture + any decode/reconstruct code) whose **size is the metric**
and **accuracy the constraint**. This is directly analogous to the Hutter Prize
(text compression): minimize the description length of a thing that reproduces a
signal. **Cross-pollinate with the `hutter_prize` crew (shannon)** on
entropy-coding weight tensors, quantization, and MDL — much of their toolkit
transfers. (They optimize bits of enwik9; we optimize bits of a CIFAR classifier.)

## Reference model

**`tysam-code/hlb-CIFAR10`** — study its architecture, data pipeline, whitening,
and training tricks as the "great model," but **re-target the objective from
speed to size.** Its enumerated goals/leaderboard structure is our template; we
are simply *more forgiving on runtime and strict on bytes.*

## Methods — any method is fair game

- Tiny architectures: micro-CNNs, depthwise-separable, low-channel, tiny MLPs.
- **Extreme quantization:** int8/int4, ternary (TWN), binary (BNN/XNOR-Net).
- **Entropy-code the weights:** quantize → arithmetic/context-code the weight
  tensor (the Hutter angle — the biggest lever once params are fixed).
- Extreme pruning + sparse encoding (store only nonzeros + a compact mask).
- Weight sharing / hashing (HashedNet).
- **Procedural / generative weights:** ship a tiny seed + small learned
  coefficients (hypernetwork, or a fixed random basis with learned mixing) — ship
  the seed, not the weights.
- Distillation into a tiny student.
- Exploit CIFAR structure: learned/PCA basis + a tiny classifier head.

## Deliverable — the Pareto frontier

A leaderboard (table in this repo) of **best accuracy at each size target**, e.g.
~1 KB, ~10 KB, ~50 KB, ~100 KB. Size measured **rigorously**: bytes of the
serialized artifact (report raw and gzipped), self-contained to reconstruct and
run inference. Each point ships its train+eval script and a reproducible number.

## Hard constraints

- **Runtime forgiving but NOT hundreds of hours.** Cap per-experiment training
  (target minutes to low hours); report wall-clock per Pareto point.
- **Box-load discipline** — the box is shared and fragile (recent OOM freezes):
  `nice -n 15`, cap parallelism (≤2–4 heavy jobs), wrap heavy jobs in
  `systemd-run --user --scope -p MemoryMax=…`, watch `uptime`. Prefer **Modal
  GPU** for anything heavy — but **Modal spend needs Rome's explicit approval.**
- **Use subagents heavily** — parallel method exploration, ideally one subagent
  per method family; the mayor expects a fan-out, not a single serial thread.

Bead every experiment + Pareto point under `tc-` (`bd new`).
