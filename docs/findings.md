# Findings so far

Everything below was measured on this box against the full 10,000-image CIFAR-10
test set, with the artifact re-run from its serialized bytes in a fresh process.
Sizes are description length as defined in [harness.md](harness.md).

**Read this as the record of the random-filter era.** Everything below is
closed-form ridge regression on random filters, which was the whole frontier when
it was written and is now only the sub-4 KB half of it. Trained CNNs with
quantization-aware training have since taken everything above that — see
[trained-cnn.md](trained-cnn.md) — and the source-golf result in
[what-to-try.md](what-to-try.md) has moved the sub-KB numbers down. The findings
here about quantization, free structure and free-at-inference tricks still hold
and still compose; the frontier figures quoted are superseded by
[LEADERBOARD.md](../LEADERBOARD.md).

## The one idea carrying most of the frontier

**A PRNG seed is four bytes no matter how much it draws** — true of the *sample*
it emits, and the phrasing needs that qualifier: selecting the best of 2^k seeds
encodes at most k bits, so a search is bounded by the 32 bits the seed field
already pays for. All results here use seed 1 and no experiment searches seeds;
the rule going forward is that seeds are selected on the train split only. Anything a short
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
is free for us — but the *source* is not, and the original wording here ("this
is under-exploited and should be pushed further") treated it as though it were.
Pushing it further has now been measured band by band in
[tta-and-distillation.md](tta-and-distillation.md): it saturates at nine spatial
positions, pays everywhere above 3 KB, and at this k=8 point buys nothing beyond
the single flip already shipped.

**4. Codebooks beat uniform grids below 5 bits.** Lloyd-max quantization wins by
up to 17.6 points at 2 bits. Ridge weights are heavy-tailed, and a min/max grid
spends its levels on outliers carrying little of the decision. Above 5 bits the
placement stops mattering and uniform catches up.

**5. Per-class codebooks, once the head is wide.** One codebook over all ten
class columns collapses as width grows (k=512 at 2 bits: 30.60%). Ten separate
codebooks cost 10·2^b float16 centroids — 160 bytes at 3 bits — and recover
+15.3 points there.

**6. Stride-1 convolution is free and mildly positive.** 3x3 stride 1 over 4x4
stride 2 is +2.0 points at k=32/6-bit for zero bytes. The +1.0 measured at
k=64/6-bit is **below the selection noise floor** (see the caveat below) and
should be treated as unresolved, not as a result.

**7. At tiny sizes the codebook overhead flips the ranking.** Per-class
codebooks cost 10*2^b float16 centroids — 320 bytes at 4 bits, a third of a
1 KB artifact. Below ~2 KB the single global codebook (32 bytes) wins back more
space than its worse fit costs. The sub-1 KB point uses global; everything above
~4 KB uses per-class.

## A caveat that applies to every number above

Every configuration in this document was scored on the test set, and the best
was reported. `harness.md` says to tune on a split of train; no experiment did.
An adversarial review measured what that costs: with a per-config binomial
standard error of 0.482 pp at n=10,000, the expected inflation of the maximum is
**+1.06 pp** over 45 configurations, and **~+1.23 pp** now that the board holds
164. A fixed validation split (`data.load_dev()`, `split="val"`) now exists and
is used going forward; existing rows are kept and annotated rather than
rescored, since each individual row remains an unbiased estimate — it is only
the *maximum* that is inflated.

So **63.41% is realistically about 62.4% as an unbiased estimate**, and any
effect below roughly 1.5 pp reported here is not distinguishable from selection
noise. The large effects — conv over dense (+14), TTA (+2.6 to +3.1), codebooks
at low bits (up to +17.6), per-class codebooks (+15.3) — survive this comfortably.
The small ones do not, and are flagged where they appear.

The fix — a fixed validation split carved from train, with test touched once per
method family — **now exists** as `data.load_dev()` and `split="val"`, and the
trained-CNN work uses it. The rows above predate it, so treat them as a ranking
with a ~1 pp fog rather than a set of point estimates.

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
structured pruning, no entropy coding). At the time of writing we were at 63.41%
in 9.5 KB, and the diagnosis was that µNAS ships *trained* filters and we shipped
none. That has since been acted on: **77.52% in 9,949 B**, closing the gap from
23 points to 9.

The survey also reports the **~1 KB, 10-class cell is empty in the literature**:
every sub-KB CIFAR-10 result it found is a 2-class relabeling where chance is
50%. Our sub-1 KB point now stands at **42.72% in 961 bytes**. If the survey's
reading of the literature holds, there is no published 10-class number at that
size to compare against — which makes it the most likely place to claim
something genuinely new, and the place where an independent check of the
literature is most worth doing before any such claim is made.

That check has since been done, and it matters how narrowly this claim is read.
At **~1 KB** it survives. Stretched to "under 10 KB" it does not: Müksch et al.
2020 report 60.4% in 5.39 KB on 10-class CIFAR-10, full test set. See
[sota-datapoints.md](sota-datapoints.md) and the frontier caveats in the README.

## The MDL track — pricing accuracy in bits

Added after the frontier above was built. A classifier is a compressor of labels, so artifact bytes
and accuracy *can* be added once accuracy is priced in bits:

    two-part total = artifact bytes + arithmetic-coded label bytes

Sending the 10,000 test labels with no model at all costs 4,152 bytes. That is
the number every model has to beat, and most of ours do not:

| artifact | accuracy | bits/label | labels | total | verdict |
|---|---:|---:|---:|---:|---|
| 961 B | 42.72% | 2.361 | 2,951 B | **3,912 B** | pays, +241 B |
| 1,713 B | 50.78% | 2.087 | 2,609 B | 4,322 B | costs 170 B more |
| 3,441 B | 56.77% | 1.833 | 2,292 B | 5,733 B | costs 1,580 B more |
| 9,503 B | 63.41% | 1.574 | 1,968 B | 11,471 B | costs 7,319 B more |

*(Corrected.* The first version of this table paired the 961 B artifact's size
with a different model's accuracy — the audit refit with per-class codebooks
while that artifact ships a global one. The audit now refuses to price an
artifact its refit does not reproduce. The conclusion is unchanged; the margin
shrinks from +318 B to +241 B, and golfing the source to 845 B restores it to
about +356 B.)

**Only the smallest artifact on the board is MDL-positive.** The 9.5 KB model —
the best point by the leaderboard's own ranking — costs seven kilobytes more
than transmitting the answers outright. No artifact above **2,184 B** can pay
for itself at any accuracy these features reach.

The two tracks rank the same artifacts in nearly opposite orders. That is not a
contradiction; they answer different questions. The leaderboard asks for the
smallest deployable model at a given accuracy, which is a real engineering
question and the one the rig was chartered on. The MDL total asks whether the
model is worth its own transmission, which is the Hutter question. Both stay.

Cross-entropy is measured on the *quantized* weights actually shipped, with a
temperature fitted on the training set. That step is not cosmetic: ridge margins
are not logits, and softmax of raw margins is nearly uniform, which would price
even a good model at ~3.3 bits/label.

### Prequential coding — and why it loses here

The full Hutter reframing ships no weights at all. The decoder holds the images,
decodes label i, updates its model on (x_i, y_i), and decodes label i+1
(Dawid's prequential principle). What is transmitted is the learning algorithm.

    cold  program 2,479 B + labels 4,541 B = 7,020 B
    warm  program 2,748 B + labels 2,376 B = 5,124 B

The warm learner codes labels better than any artifact we have — 1.901
bits/label — and still loses, because it pays 2.7 KB of fixed program to do it.
Cold, learning from scratch on 10,000 examples, does worse than sending the
labels raw. The lesson is about amortization: prequential machinery is a fixed
cost spread over symbols, and 10,000 labels is too few to spread it over. It
should win on the 50,000-label training stream.

Both variants are verified by decoding back with a real arithmetic coder. A
codelength that does not round-trip is not a codelength.

### The load-bearing assumption

The prequential track only works because the training set is free to both sides.
If that is granted, then a decoder can *re-derive* weights instead of receiving
them, and the artifact track's size axis is partly an artifact of forbidding the
model to look at data it could have had. This is worth stating plainly because
it is the assumption the whole comparison rests on, and it is a choice rather
than a fact. The harness now enforces the strict reading — artifacts cannot read
the dataset at all — and the prequential track is where the other reading lives.


## What to try next, ranked

0. **Go down, not up.** The MDL audit says the whole region above ~2.2 KB is
   dead in two-part terms. The 961 B point is the only one that pays, and it has
   had the least attention.
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
