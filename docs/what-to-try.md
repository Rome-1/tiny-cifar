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
| **G** | **shrink the emitted source** | golf the decoder itself |

C and D are where our wins came from, because both are free in bytes. **G was
missing from the first version of this document and is the largest sub-KB lever
there is** — see below. F reranks the whole board. E, which every prior byte-golf
effort converged on, has now been measured and does not transfer.

## What we already know (and it constrains a lot)

Measured here, not recalled:

1. **Input reduction is free and large.** 8×8 RGB beat full 3072-d pixels at 1/12
   the size. Anything a line of numpy generates costs nothing.
2. **Convolution beats a dense projection at identical seed cost** — +14 points.
   Structure matters more than the projection being learned.
3. **Free-at-inference tricks pay, but they are not free.** Flip TTA: +2.6 to
   +3.1 points for 12 bytes. Inference *time* is unbudgeted; the source that
   spends it is not. Now measured per band — see item 4 below.
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

## Lever G, and why it leads

Two independent reviews measured the same thing: in the 961 B flagship,
`predict.py` is 966 B raw and the weight file is 308 B. **Two-thirds of the
artifact is source.** Every method in the surveys attacks the other third.

The asymmetry is worse than it looks, because xz takes ~35% off source and ~0%
off the weight indices (near-uniform k-means codes). Source bytes are discounted;
weight bytes are not; and there are twice as many source bytes.

Golfing the generated decoder — width-specialized unpacking instead of the
generic `unpackbits` path, `open(__file__[:-10]+"w")` instead of importing
`pathlib`, `48**-.5` instead of a 17-digit literal — takes the flagship from
**961 B to 845 B at an unchanged 42.72%**, verified through the harness. At the
local frontier slope (~1.1 points per 100 B near 1 KB) that is worth about an
accuracy point, at zero risk and no compute. `experiments/golf.py`.

## Tier 1 — do these

**1. ~~Trained filters.~~ Done, and it took the whole board above 4 KB.**
Depthwise-separable CNNs with QAT now own every frontier point from 3 KB up:
73.61% @ 5.1 KB, 77.52% @ 9.9 KB, 84.21% @ 38.9 KB, 85.37% @ 69.3 KB. The
falsifier did not fire — at 5 KB the trained net beats the entire random-filter
family, which plateaued at 71.33% @ 69 KB and is retired.

Two things the size sweep [t, s, m, l, xl] settled:

- **Width scales badly against µNAS.** Reaching 85.37% took 69 KB, six times what
  µNAS spends to beat it at 11.4 KB. Scaling this architecture is not the way to
  close that gap; the gap is architectural. Nothing above `l` is worth training.
- **Precision beats width at fixed bytes here too**, matching finding 5 for random
  filters. 4-bit on `l` gives 84.21% @ 38.9 KB; 3-bit on `xl` gives 83.26% @
  51.9 KB and is dominated. Prefer a smaller net at higher precision, always.

What is left in this family is not more parameters — it is items 4 and 5 below,
both of which cost zero artifact bytes.

**2. ~~The lookup-table architecture.~~ Measured; its own falsifier fired.**
The empirical ceiling of a b-bit thresholded partition of CIFAR-10 is **~32% on
val, peaking at b=10-11 and then declining** — a generalization ceiling, not a
selection artifact. A real shipped table reached 575 B / 29.08%. Priced in bits
it manages 2.757 bits/label against the 961 B ridge model's 2.299, so it is
barely MDL-positive. The MNIST precedent does not transfer: six thresholded
pixels reach 56.7% there because MNIST is ~92% linearly separable from raw
pixels; on CIFAR-10 six bits buys 28.6%. Kept below only as a way into the
500-700 B band. Original text follows.

*(superseded)* **The lookup-table architecture, at sub-1 KB.** Every source-byte precedent
independently converged on the same shape: project to a handful of thresholded
bits, then a compressed lookup table. Code Golf 28207 won MNIST at **101 bytes /
56.7%** with six thresholded pixels into a 64-entry table (standings verified
directly against the Stack Exchange API); the Number Plate and
Roboto Mono challenges rediscovered it. We have never tried it, and our 961 B
point is exactly where it belongs. Note the second-order trick from that entry:
**choose thresholds partly for table compressibility**, not accuracy alone.
*Falsifier:* if a 10-class CIFAR table needs more index bits than the entropy of
the classes it separates, it will not beat a linear head — check H(class|index)
before building it.

**3. Quantization-aware training — a precondition, not an enhancement.** Measured
in our own in-flight run: at 4 bits, post-training quantization gives 53.15% and
QAT gives **76.54%** — a 31.7-point gap. At 3 bits PTQ collapses outright to
18.55%. Below 6 bits, trained filters *without* QAT lose to random filters at the
same size, so anyone running that comparison without QAT will wrongly conclude
trained filters do not work. Merge this into item 1.
*Falsifier:* on the 961 B head the float ridge scores 45.51% against the 4-bit
codebook's 42.72%, so QAT can recover at most **2.79 pp** there — it is a 10 KB
lever, not a sub-KB one.

**4. ~~Harder TTA.~~ Measured, and it pays — but it was never "pure profit".**
TTA costs *source* bytes, and source is charged at the weight file's rate, so
the question is per band. Settled in
[tta-and-distillation.md](tta-and-distillation.md): saturation is at nine
spatial positions plus a flip (`flip+box1`), two-pixel shifts and edge-replicated
padding both measure worse, and the trade pays at every size except the 931 B
conv-ridge point, where the single flip already shipped is the whole effect.
Worth +2.5 pp for 58 B at 3.9 KB, +1.1 pp for 56 B at 12 KB, and +0.5 pp for
**four bytes** at 69 KB. Nothing further to chase.

**5. ~~Distillation into the tiny student.~~ Falsifier fired.** Against a
hard-label twin trained through the identical loop over the identical
augmentation cache, an `xl`-distilled arch-`t` student wins by +0.52 pp on val
(exact McNemar p = 0.355) and *loses* by 0.55 pp on test, at matched bytes. Soft
targets into the closed-form ridge head are worse at every temperature. The one
place it works is when the student's augmentation is starved — at 16 cached
crops it is worth +1.52 pp, at 64 it is worth nothing — so what it buys is a
substitute for augmentation these students already have for free. Retired.

## Tier 2 — strong bets, more work

**~~6. Circulant / tiled weight sharing.~~ Measured, and cut.** The tile carries
no signal. Best of 400 searched circulant tiles scores 21.20% on val; a *random*
binary head of the same size scores 21.64% — statistically indistinguishable. All
the accuracy came from the 11x10 mixing matrix bolted on top, which costs more
bytes than the tile saves. And simply dropping k from 6 to 3 saves 139 B and
lands at 32.50%, dominating the whole approach. The "0.0009 bits/weight" headline
is a function of matrix size, not compression: 550 weights x 0.0009 bits is half
a bit in total, which is meaningless at our scale.

**7. Decision trees — yes, but plain ones, and only for the 500-700 B band.**
Bonsai's defining trick, sparse learned projections at each node, measured
*worse* than plain axis-aligned pixel splits at every budget tested (28.91% vs
30.00% at 64 leaves, and so on up). A plain tree ships at **512 B / 28.31%** and
**680 B / 30.15%** on val. That does not threaten the flagship, but it enters a
band the conv-ridge family cannot reach at all — see below.

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

## The one genuinely open band: 500-700 B

The conv-ridge family has a **hard floor at ~694 B** — that is source plus a
minimal head, and at that size it scores 14.6%. It cannot enter the 500-700 B
band at all. A plain decision tree or an oblivious table can: ~512 B at 28.3%,
~575 B at 29.1%, against the current best-below-700 B of 26.46% at 480 B.

That is a real new Pareto point, and it is strictly below the flagship rather
than a threat to it. Build one of the two — a table over b thresholded bits *is*
an oblivious tree, and the two land within 0.5 pp of each other — not both.

## The honest summary

The remaining upside splits cleanly:

- **10–100 KB:** trained filters, and only trained filters. We know the wall
  (86.49% @ 11.4 KB) and we know why we are short of it.
- **Sub-1 KB:** **source golf first** — it is two-thirds of the artifact and
  already worth 116 B. Then a plain tree or table to open the 500-700 B band.
  The three architectures originally proposed here were measured and all land
  10-14 points *below* the existing 42.72% @ 961 B; none of them is a flagship.

The methodological debt to clear first, in both cases: a **validation split**, so
that sweeps stop selecting on test and effects under 1.5 pp mean something.
