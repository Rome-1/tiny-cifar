# Method survey: minimizing CIFAR-10 classifier artifact size in bytes

Bead: tc-b2b. Author: rissanen. Date: 2026-07-28.

**Scope.** The objective is the serialized size in bytes (raw and gzipped) of a
self-contained artifact that classifies CIFAR-10. Runtime is nearly free; bytes are
the only currency. This is an MDL problem: the artifact is a program, and its length
is the score.

**Verification convention used throughout.** Claims marked *[verified]* were read
from a source I or a subagent fetched during this session, with the URL given.
Claims marked *[UNVERIFIED RECALL]* are from model memory and have **not** been
checked against a source — treat them as hypotheses to confirm before they enter a
paper or a decision. Arithmetic I performed myself is marked *[derived]*.

---

## Part 0 — The byte-accounting frame (read this first)

Everything below only matters relative to a correct accounting model. Three points
that reshape the whole problem:

1. **The artifact includes the decoder.** A self-contained artifact is
   `decoder source/bytecode + weight payload + preprocessing constants`. At the
   ~1KB target the decoder stub is a first-order cost, not a rounding error: 40
   lines of dense Python is ~1.5KB of source, already over budget. At 1KB the
   design must be "a program so simple it barely needs a decoder," not "a network
   plus a clever codec."
2. **Container overhead dominates at small sizes.** A `.npz` is a ZIP: ~30-byte
   local header + filename + ~46-byte central-directory entry per array, plus a
   22-byte end record. A `.pt` (torch.save) carries a full pickle + ZIP directory.
   A 500-param int8 tensor saved as `.npz` can easily be 2-3x its payload.
   *[derived]* **Rule: one flat byte blob, one array, no per-tensor container
   entries.** Split tensors at load time from a shape spec that lives in the
   decoder source.
3. **Raw vs gzipped are different games and reward opposite representations.**
   If the target metric is *gzipped* size, then bit-packing your own weights is
   close to counterproductive — a bit-packed array is near maximum entropy and
   gzip cannot help, whereas a byte-per-symbol array with a skewed symbol
   distribution lets DEFLATE's Huffman stage recover most of the entropy gain for
   free and with **zero decoder bytes**. If the target is *raw* size, you must do
   the entropy coding yourself and pay for the decoder. This tension needs an
   explicit decision from the project before methods are chosen. (See Part 2c —
   this was independently flagged by the coding-theory sweep.)

---

## Part 1 — `tysam-code/hlb-CIFAR10` dissected

Sources fetched: `https://github.com/tysam-code/hlb-CIFAR10` (README) and
`https://raw.githubusercontent.com/tysam-code/hlb-CIFAR10/main/main.py`.

### Headline claims *[verified, README]*

> "Train to 94% on CIFAR-10 in <6.3 seconds on a single A100. Or ~95.79% in ~110
> seconds (or less!)"

The repo is a descendant of David Page's fast-CIFAR-10 work (the "How to Train Your
ResNet" series). It is a single-file, deliberately hackable implementation.

### Architecture, layer by layer *[verified, main.py]*

Channel plan, verbatim:

```python
depths = {'init': 32, 'block1': 64, 'block2': 256, 'block3': 512, 'num_classes': 10}
```

- **Whitening conv**: `Conv2d(3 -> 12, kernel_size=2)`, **non-trainable**,
  initialized from the eigenvectors of 2x2 image patches over
  `num_examples: 50000` training images. Output is expanded to 24 channels
  (the whitened response and its negation), then GELU.
- **ConvGroup 1**: `Conv(24->64, 3x3)` -> `MaxPool2d(2)` -> `BatchNorm` -> `GELU`
  -> `Conv(64->64, 3x3)` -> `BatchNorm` -> `GELU`
- **ConvGroup 2**: same shape with `64->256` then `256->256`
- **ConvGroup 3**: same shape with `256->512` then `512->512`
- **FastGlobalMaxPooling** (`torch.amax` over spatial dims)
- **`Linear(512 -> 10, bias=False)`** with a fixed output temperature of `1/9`
  (`scaling_factor: 1./9`)

Residual structure is not explicit modules — instead the convs inside each group get
a **Dirac-initialized identity component added to their kernels**, then the weights
are renormalized. Information passes through at init without a separate skip branch.

### Parameter count *[derived — I computed this, the repo does not state it]*

| Component | Params |
|---|---|
| whitening conv (3->12, 2x2), frozen | 144 |
| G1: 24->64 + 64->64 (3x3) | 50,688 |
| G2: 64->256 + 256->256 | 737,280 |
| G3: 256->512 + 512->512 | 3,538,944 |
| BatchNorm affine (6 layers) | 3,328 |
| Linear 512->10 | 5,120 |
| **Total** | **~4.34M** |

At fp32 that is ~17.3 MB. **This model is roughly 4 orders of magnitude larger than
our 1KB target and 3 orders larger than our 100KB target.** hlb-CIFAR10 is
therefore *not* an architecture to shrink — it is a **source of training
technique**. Note also that >80% of its parameters sit in ConvGroup 3 (the
256->512 and 512->512 convs), which is exactly the part a byte-constrained design
would never build.

### Data pipeline *[verified, main.py]*

- **Whole dataset resident on GPU in fp16** (`data.pt`), so no dataloader.
- Channel-wise mean/std normalization computed from the training set.
- **Reflect padding of 2 px** (`pad_amount: 2`), then random 32x32 crop.
- Random horizontal flip, p=0.5.
- **CutMix**, patch size up to `cutmix_size: 3`, applied only in the last
  `cutmix_epochs: 6` epochs, with proportionally blended labels.

### Training recipe *[verified, main.py]*

- `batchsize = 1024`, `train_epochs = 12.1`
- One-cycle-style LR: `percent_start: .23`, `initial_div_factor = 1e16`,
  `final_lr_ratio = .07`; separate LRs and weight decays for bias vs non-bias
  parameters (`bias_scaler = 64`)
- **Label smoothing 0.2**
- **BatchNorm momentum 0.4**
- **EMA of weights** over the last 10 epochs, updated every 5 steps, decay
  `0.95^5 * (step/total)^3`; EMA params are copied back into the live net
- **TTA at eval**: input concatenated with its horizontal flip, logits averaged 50/50

### The byte-cost classification — the actual deliverable of Part 1

This is the table that should drive our decisions.

**FREE (training-time only, zero artifact bytes) — adopt all of these aggressively:**

| Trick | Why it is free |
|---|---|
| One-cycle / warmup LR schedule | Schedule is not shipped |
| Separate bias vs non-bias LR and weight decay | Optimizer state, not shipped |
| Label smoothing (0.2) | Loss function only |
| EMA of weights | Produces *the same tensor shape* as the raw weights — you ship one set either way. Pure free accuracy. |
| CutMix / cutout, late-phase only | Augmentation only |
| Reflect-pad + random crop, random flip | Augmentation only |
| Dirac / identity initialization | Init only; the trained weights are what ship |
| GPU-resident fp16 data, memory format, tensor-core tricks | Pure speed, no bytes |
| Knowledge distillation from a big teacher (not in hlb, see Part 2g) | Teacher is discarded |
| Longer training, more epochs | Free — our budget is minutes-to-hours, so **train far longer than hlb's 12 epochs**; hlb's epoch count is a *speed* constraint we do not share |

**COSTS BYTES:**

| Item | Cost | Notes |
|---|---|---|
| Every conv/linear weight | the whole game | |
| BatchNorm affine + running stats | 4 floats per channel | Can be **folded into the preceding conv** at export: scale and bias absorb into weight and bias. Free byte savings, do it always. |
| The whitening conv | 144 params here | Data-derived, but the artifact must be self-contained, so it must be shipped. Cheap, and see Part 2h — a fixed DCT/analytic basis costs **zero** bytes instead. |
| Channel mean/std normalization constants | 6 floats | Fold into the first conv, or into the whitening layer. Should never appear as separate stored numbers. |
| Output temperature 1/9 | 0 bytes | It is a constant in the decoder; and for argmax classification a positive scalar temperature is a **no-op** — drop it entirely. |
| TTA at eval | **0 bytes** | Costs only inference time — and inference time is free for us. **Adopt TTA unconditionally**, and consider going beyond hlb's flip-only: multi-crop and small translations are all free bytes. |

**The three highest-leverage transfers from hlb to this project:** (1) EMA, (2)
aggressive late-phase augmentation with label smoothing, (3) free-at-eval TTA
expanded well past hlb's single flip. All three are pure accuracy with a zero-byte
price tag, which in a size-constrained regime is the only kind of win that is
unambiguously worth taking.

---
