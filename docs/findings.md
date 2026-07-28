# Findings so far

Everything below was measured on this box against the full 10,000-image CIFAR-10
test set, with the artifact re-run from its serialized bytes in a fresh process.
Sizes are description length as defined in [harness.md](harness.md).

The whole frontier to date is closed-form ridge regression. **No backprop has
been run yet.** That is worth stating plainly, because it bounds what these
numbers mean: they are the floor a few free tricks reach, not a ceiling.

## The one idea carrying most of the frontier

**A PRNG seed is four bytes no matter how much it draws.** Anything a short
piece of code can generate is free, so the only weights that cost bytes are the
ones that cannot be generated — here, the classifier head. That single asymmetry
took the frontier from 41% to 63% without a single trained filter:

| what is shipped | ≤ 10 KB accuracy |
|---|---|
| all the weights (linear on pixels) | 41.07% |
| head only, dense random projection from a seed | 47.01% |
| head only, random *conv* filters from a seed | 63.41% |

The seed is legitimate description length rather than an accounting trick only
because the harness re-runs the artifact from its bytes in a clean process — a
projection that failed to reproduce would collapse, not quietly pass.

## What worked, in order of leverage

**1. Input reduction is free, and it is large.** Block-mean pooling to 8x8 RGB
beat full 3072-d raw pixels (40.13% vs 40.10%) at one twelfth the size. At the
small end this dominates every other lever: the weight matrix shrinks by up to
192x for about sixty bytes of source.

**2. Convolution beats a dense projection at identical cost.** Both are drawn
from the same four-byte seed. Structure is worth ~14 points (47.01% → 60.83%).

**3. Flip TTA costs twelve bytes.** The head is linear, so averaging the two
feature vectors is identical to averaging the two logit vectors — one extra term
in the source, no second scoring pass. Worth +2.6 to +3.1 points. Inference time
is free for us; this is under-exploited and should be pushed further.

**4. Codebooks beat uniform grids below 5 bits.** Lloyd-max quantization wins by
up to 17.6 points at 2 bits. Ridge weights are heavy-tailed, and a min/max grid
spends its levels on outliers carrying little of the decision. Above 5 bits the
placement stops mattering and uniform catches up.

**5. Per-class codebooks, once the head is wide.** One codebook over all ten
class columns collapses as width grows (k=512 at 2 bits: 30.60%). Ten separate
codebooks cost 10·2^b float16 centroids — 160 bytes at 3 bits — and recover
+15.3 points there.

**6. Stride-1 convolution is free and mildly positive.** 3x3 stride 1 over 4x4
stride 2 is +2.0 points at k=32/6-bit, +1.0 at k=64/6-bit, for zero bytes. It
costs only training time.

**7. At tiny sizes the codebook overhead flips the ranking.** Per-class
codebooks cost 10*2^b float16 centroids — 320 bytes at 4 bits, a third of a
1 KB artifact. Below ~2 KB the single global codebook (32 bytes) wins back more
space than its worse fit costs. The sub-1 KB point uses global; everything above
~4 KB uses per-class.

## What did not work, and what that rules out

**Width is the wrong place to spend bits.** This is the most useful negative
result so far. At a fixed ~9.5 KB, a narrow head at 6 bits (k=128, 63.41%) beats
a wide head at 3 bits with per-class codebooks (k=256, 56.37%). The ordering
held across every width and bit budget tested. More random filters are not the
route to a better 10 KB point — better *features* are.

**Codebooks destroy the redundancy the entropy coder was eating.** Codebook
artifacts barely compress (gzip ≈ raw) because k-means indices are near-uniform
in entropy, while affine codes keep a skew gzip exploits. Better reconstruction,
worse residual redundancy. This is why the frontier, not the reconstruction
error, is the thing to read — and why all three codec columns stay on the board.

**Reconstruction error is not a proxy for accuracy.** Several points with lower
relative weight error scored worse. Do not tune against it.

## Where this stands against the literature

The survey puts the published bar at **µNAS, 86.49% in 11.4 KB** (int8,
structured pruning, no entropy coding). We are at 63.41% in 9.5 KB. The gap is
not mysterious — µNAS ships *trained* filters and we ship none. Closing it means
training, which is exactly the direction not yet attempted.

The survey also reports the **~1 KB, 10-class cell is empty in the literature**:
every sub-KB CIFAR-10 result it found is a 2-class relabeling where chance is
50%. Our sub-1 KB point now stands at **42.72% in 961 bytes**. If the survey's
reading of the literature holds, there is no published 10-class number at that
size to compare against — which makes it the most likely place to claim
something genuinely new, and the place where an independent check of the
literature is most worth doing before any such claim is made.

## What to try next, ranked

1. **Train the filters.** Every gain above came from free structure; the largest
   remaining lever is the one component we have never learned. A small trained
   CNN, quantized and entropy-coded, is what the 86% bar is made of. This is the
   first thing that needs real compute rather than a closed-form solve.
2. **Patch whitening.** The survey names the frozen 2x2 patch-whitening stem
   (~144 bytes) as the best accuracy-per-byte component it found anywhere.
   Cheap to test, composes with everything here.
3. **The 1 KB point.** Conv features at very small width, against a literature
   cell that is empty. **Now at 42.72% in 961 B** — a random 6-filter conv bank
with a 4-bit head — up from 36.82% for the linear model it replaced.
4. **A zero-byte analytic DCT stem** in place of a learned basis — the survey
   found no published CIFAR-10 result for this.
5. **Harder TTA.** Inference time is free and we currently use one flip.
6. **rANS on the weight indices.** The survey's arithmetic says gzip lands
   1.3–1.6x above entropy at any layout; below ~1.2 effective bits/weight a real
   entropy coder is needed. Worth it only once a method is genuinely at that
   density — the decoder itself costs 600–800 bytes.
