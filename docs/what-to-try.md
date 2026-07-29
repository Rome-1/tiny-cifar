# Everything worth trying for CIFAR-10, ranked

A decision document, not another survey. The three research docs
([method-survey](method-survey.md), [exotic-methods](exotic-methods.md),
[cmix-and-prequential](cmix-and-prequential.md)) enumerate the space; this one
says what is actually worth doing, in what order, and what we have already
learned that rules things out.

Everything here is conditioned on what we **measured**, which is the part no
literature survey could supply.

## The six levers

There are only six ways to make the number go down. Every method is one of
these or a composition of them.

| | lever | what it changes |
|---|---|---|
| **A** | fewer parameters | how many numbers must be described |
| **B** | fewer bits per parameter | quantization, entropy coding |
| **C** | make parameters *generatable* | seeds, fixed bases, circulant structure |
| **D** | more accuracy at fixed parameters | training, distillation, augmentation |
| **E** | move bits from weights into *code* | lookup tables, decision structures |
| **F** | change what is measured | prequential / MDL framing |

C and D are where our wins came from, because both are **free in bytes**. E is
the one we have never touched and is where every prior byte-golf effort
converged. F reranks the whole board.

## What we already know (and it constrains a lot)

Measured here, not recalled:

1. **Input reduction is free and large.** 8×8 RGB beat full 3072-d pixels at 1/12
   the size. Anything a line of numpy generates costs nothing.
2. **Convolution beats a dense projection at identical seed cost** — +14 points.
   Structure matters more than the projection being learned.
3. **Free-at-inference tricks pay.** Flip TTA: +2.6 to +3.1 points for 12 bytes.
   Inference time is unbudgeted and still under-exploited.
4. **Codebooks beat uniform grids below 5 bits** (up to +17.6). Above 5 bits,
   placement stops mattering.
5. **Precision beats width at fixed bytes**, over the whole range tested. More
   random filters at low bits always lost to fewer filters at higher precision.
6. **Codebook overhead flips the ranking below ~2 KB** — per-class codebooks cost
   10·2^b centroids, a third of a 1 KB artifact.
7. **Random filters plateau at 71.33% @ 69 KB.** Doubling the bank buys <0.5
   points. That family is finished.
8. **Only artifacts under ~2,212 B can pay for themselves** in MDL terms. The
   entire region above that is dead on the second scoreboard.
9. **Selection noise is ~1.06 pp** across a 45-config sweep. Anything smaller is
   not a result.

## Tier 1 — do these

**1. Trained filters (in progress).** The single largest gap. Everything on our
frontier is closed-form ridge on random filters; µNAS reaches 86.49% @ 11.4 KB
with trained ones. First runs already show 72.60% at 6-bit, past the random-filter
plateau. Compose with: BN folding at export, per-class codebooks, TTA.
*Falsifier:* if a trained CNN at matched bytes fails to beat the random-filter
frontier by >1.5 pp, the bottleneck is the head, not the filters.

**2. The lookup-table architecture, at sub-1 KB.** Every source-byte precedent
independently converged on the same shape: project to a handful of thresholded
bits, then a compressed lookup table. Code Golf 28207 won MNIST at **101 bytes /
56.7%** with six thresholded pixels into a 64-entry table; the Number Plate and
Roboto Mono challenges rediscovered it. We have never tried it, and our 961 B
point is exactly where it belongs. Note the second-order trick from that entry:
**choose thresholds partly for table compressibility**, not accuracy alone.
*Falsifier:* if a 10-class CIFAR table needs more index bits than the entropy of
the classes it separates, it will not beat a linear head — check H(class|index)
before building it.

**3. Quantization-aware training.** We only do post-training quantization. At 2–4
bits — the regime that matters, given lever B is where the bytes are — QAT is
worth several points and costs only training time.

**4. Harder TTA.** We use one flip. Inference is free: multi-crop, small
rotations, scale jitter. Pure profit until it saturates; measure where.

**5. Distillation into the tiny student.** Train a large model, distill into the
sub-KB artifact. Costs zero artifact bytes. Standard, reliable, untried here.

## Tier 2 — strong bets, more work

**6. Circulant / tiled weight sharing.** The verified route *below* 1 bit per
weight: TBNN stores a shared circulant bit tile and regenerates the weight
matrix — 720-byte MNIST model at ~91%, 0.0009 bits/weight. This is lever C
applied to a *learned* object rather than a random one, and it is the most
promising untried idea for sub-KB CIFAR.

**7. Sparse-projection decision trees (Bonsai).** 300-byte models, ICML 2017. A
non-neural baseline that a byte-budgeted entry ought to beat before claiming
anything at sub-KB. Cheap to test, and it is lever E in a different costume.

**8. Learned free-ish stems.** The frozen 2×2 patch-whitening stem (~144 bytes)
is the best accuracy-per-byte component found anywhere in the survey. Also
untested: a **zero-byte analytic DCT stem** in place of a learned basis — no
published CIFAR-10 result exists for it.

**9. rANS / context coding on weight indices.** Our codebook artifacts barely
compress (gzip ≈ raw) because k-means indices are near-uniform in entropy. A
context model could still exploit correlation across the 10 class columns for the
same feature. Realistic gain: −650 to +150 B at 9.5 KB, and the decoder costs
600–800 B — so this only pays on large artifacts, and large artifacts are
MDL-dead. **Low priority despite being the "Hutter angle."**

**10. Seed search, done honestly.** Selecting the best of 2^k seeds encodes at
most k bits, so a search is bounded by the 32 bits the seed field already pays
for — it does not break the accounting. But ranking 2^20 seeds on *test* buys
~2.7 pp of pure noise. Search on the train split only. Cheap, real, and easy to
do wrong.

## Tier 3 — try if the above stalls

- **Hypernetworks / INR weight generation** — a small net that emits the filters.
  Cost is the hypernet itself; unproven at this scale.
- **Tensor decompositions** (Tucker, CP, tensor-train) on conv layers.
- **Structured transforms** — Hadamard, Toeplitz, Fastfood. O(n) instead of O(n²),
  and generatable, so lever C again.
- **Product quantization** of the head, rather than scalar codebooks.
- **Mixed precision across layers** — spend bits where sensitivity is highest.
  Cheap, likely worth ~half a bit/param on average.

## Ruled out, with reasons

- **Unstructured pruning.** Arithmetically unwinnable at our scale: address cost
  is H(p)/p bits per survivor — 8.08 bits at 1% density, twice the cost of the
  4-bit weight it points at. A 99%-sparse int4 net must beat a *3× larger dense
  net* to break even, and the optimal mask encoding is already known, so there is
  no headroom left.
- **Supermasks over random weights.** ~213 KB for 76.5% on Conv-6. Two orders off.
- **Binary/ternary as a *size* strategy per se.** TWN and BinaryConnect use ~13–14 M
  parameter VGG-7s: 1 bit/weight is still 1.75 MB. Ternary is the right *grid*
  (it beats binary 0.32 pp vs 2.70 pp degradation on matched VGG-7, and its zero
  level is what an entropy coder monetizes) — but it is lever B, not a
  small-model architecture.
- **PCA / learned bases as shipped objects.** PCA-200's head is 2 KB but its basis
  is 614 KB. Fixed bases only pay at zero bytes.
- **Dataset distillation.** Dominated: µNAS at 11.4 KB / 86.5% beats the best
  distillation result at 94 KB / 77.5%.
- **More random filters.** Measured plateau. Finished.

## Which scoreboard

Two live, and they rank the same artifacts nearly backwards:

- **Artifact bytes** (the charter): smallest deployable model at a given accuracy.
  Interesting range is 100 B – 100 KB.
- **MDL total** (artifact + arithmetic-coded labels): does the model pay for its
  own transmission? Baseline 4,152 B. **Only our 961 B point clears it.** Nothing
  above 2,212 B ever can.

If the MDL board is taken seriously, Tier 1 items 2, 5 and Tier 2 items 6, 7 are
the *only* ones that matter, because they all target sub-KB. That is a real
strategic fork and it should be decided deliberately rather than by drift.

## The honest summary

The remaining upside splits cleanly:

- **10–100 KB:** trained filters, and only trained filters. We know the wall
  (86.49% @ 11.4 KB) and we know why we are short of it.
- **Sub-1 KB:** lookup tables, circulant tiles, sparse trees. This is where the
  literature is empty, where the MDL board says the only real points live, and
  where we have tried the least. It is the more interesting half.

The methodological debt to clear first, in both cases: a **validation split**, so
that sweeps stop selecting on test and effects under 1.5 pp mean something.
