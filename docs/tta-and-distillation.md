# TTA and distillation, priced in bytes

Two levers were supposed to buy accuracy without buying weights. One of them
does. This document reports both, band by band, and retires the other.

It also corrects a claim this repo made in two places. `findings.md` item 3 and
`what-to-try.md` item 4 both describe TTA as free — "pure profit until it
saturates". That is wrong as stated, and the error is not small. **TTA costs
source bytes.** Every extra transform is more code in `predict.py`, and
`predict.py` is charged at the same rate as the weight file. At 69 KB a hundred
bytes of source is noise; at 470 B it is a fifth of the artifact. The question
is not "does TTA help" but "does the gain beat the source bytes, *at this
size*". A transform that pays at 69 KB and loses at 931 B is the expected
result, and it is what happened.

Distillation is the lever that really is byte-free: it changes what the student
is trained against and leaves the artifact's shape, weight count and decoder
untouched. It is also the one that did not survive its falsifier.

Everything below is selected on the 5,000-image validation split from
`tinycifar.data.load_dev()`. Test is scored once per declared artifact, through
`tinycifar/evaluate.py`, in a fresh subprocess from the artifact's own bytes.

## Two noise floors, and which is which

The repo's stated floor is **1.06 pp**, which prices the inflation of the
*maximum* of an unpaired sweep. It is the right floor for "is this configuration
better than that one" when the two were trained separately.

It is the wrong floor for TTA. Two TTA schedules run the *same weights* over the
*same 5,000 images*; only the aggregation differs. That is a paired comparison,
and the exact McNemar test on the disagreement counts is what settles it. Both
numbers are reported for every row below, and neither is allowed to stand in for
the other: a row at +0.9 pp with p = 0.005 is a real effect that is nonetheless
smaller than the repo's unpaired floor, and both facts are true at once.

## Lever 1 — harder TTA

`experiments/tta_cnn.py` execs an existing artifact's own `predict.py` to get the
exact forward pass that ships, scores each candidate view set on validation, and
re-emits `predict.py` with the winning aggregation so the byte cost is measured
rather than estimated. The weight file is never touched.
`experiments/tta_ridge.py` does the same for the conv-ridge family, where the
head has to be refit because the views are summed in feature space.
`experiments/tta_table.py` does it for the oblivious table, which has no scores
to average and must vote.

Schedules are named `flip+<shifts>`: `cross1` is the four one-pixel neighbours
plus the identity, `diag1` the four diagonals, `box1` all nine of {-1,0,1}²,
`box2` all twenty-five of {-2..2}², `cross2` the one- and two-pixel cross.
Shifts are `np.roll`, which wraps at the border.

### The table (470 B) — `experiments/tta_table.py`

The hardest band, and the only one where TTA changes the *shape* of the decoder.
The table emits a label, not a score, so combining T views costs an `np.eye` and
a vote: +77 bytes before a single shift is added. It also had no flip TTA to
begin with, so here even the first transform is an open question.

| schedule | views | val | bytes | Δ bytes | paired | p |
|---|---:|---:|---:|---:|---|---:|
| incumbent (1 view) | 1 | 29.10% | 470 | — | — | — |
| `2px` | 5 | 30.22% | 547 | +77 | +160/−104 | 0.0007 |
| `1px+flip` | 10 | 30.26% | 559 | +89 | +293/−235 | 0.013 |
| **`4px+flip`** | 10 | **30.62%** | **560** | **+90** | +365/−289 | 0.003 |
| `2px9+flip` | 18 | 30.50% | 571 | +101 | +367/−297 | 0.007 |
| `4px9+flip` | 18 | 28.32% | 571 | +101 | +430/−469 | 0.21 |
| flip-augmented refit, `2px9+flip` | 18 | 30.84% | 586 | +116 | +398/−311 | 0.001 |

Two controls matter here. Refitting the table's cell labels without augmentation
reproduces the shipped `w` file **byte for byte**, which is the check that this
module's idea of the decoder matches the one that shipped. And refitting *with*
flip augmentation — the zero-source-byte lever TTA is competing against in this
band — buys nothing on its own: 29.02% against the incumbent's 29.10%, while
costing 17 bytes because the refit table compresses slightly worse. Layered on
top of TTA it is worth +0.34 pp for +15 B, which is inside the noise. Train-time
augmentation is not the answer here; inference-time voting is.

Whether +90 bytes for +1.5 pp pays depends entirely on what else those bytes
could buy, and the answer is *nothing*. A depth sweep of the same family
(`experiments/oblivious_table.py --grids 8 --depth 12 --bmin 8`) shows it
saturates at b = 9:

| table | bytes | val |
|---|---:|---:|
| b = 8 | 363 | 28.18% |
| **b = 9** | **472** | **29.10%** |
| b = 10 | 673 | 28.32% |
| b = 11 | 1,081 | 28.60% |
| b = 12 | 1,878 | 27.88% |

Doubling the table costs 200 bytes and *loses* 0.8 points. So in this band TTA is
the only lever that converts bytes into accuracy at all, and the resulting point
is Pareto-nondominated: **560 B / 30.80% on test**, against 470 B / 28.25%.

Priced against the joint frontier's local slope instead — 470 B → 931 B is 3.4 pp
per 100 B — the 90 bytes "should" have bought 3.0 points and bought 2.6. Both
readings are true; the second is the one that says this band is thin, not that
TTA failed in it.

### The conv-ridge family (931 B and 3.9 KB) — `experiments/tta_ridge.py`

The head is linear, so T views cost one matmul rather than T: the views are
summed in feature space and the head is fit against the same sum. This is why
flip TTA was ~12 bytes here rather than a second scoring pass, and it is why
shifts are cheap — the only new source is the shift list itself.

*A parametrization bug worth recording.* Summing T views scales the features by
about T, which weakens the effective ridge penalty by T². A first version of this
module fit against the view *mean* at a fixed lambda and shipped an unrescaled
bias row; the emitted artifact scored **2.8 points below its own incumbent on
test** while looking better on validation. The fix is to accumulate the Gram
matrix once per schedule and solve it at every lambda in a grid, picking the best
on validation, so each schedule is seen at its own best regularization. At T = 2
and lambda = 1e2 this reproduces `golf.py` exactly, which is the check that the
incumbent is inside the grid rather than outside it.

**k = 8, 4-bit — the 931 B point.** Nothing beyond the incumbent flip pays, and
most schedules are worse:

| schedule | views | val | bytes | Δ bytes | paired | p |
|---|---:|---:|---:|---:|---|---:|
| `none` | 1 | 42.00% | 920 | −12 | +479/−633 | <0.001 |
| **`flip`** (incumbent) | 2 | **45.08%** | 932 | — | — | — |
| `cross1` | 5 | 43.48% | 958 | +26 | +437/−517 | 0.011 |
| `flip+cross1` | 10 | 44.06% | 965 | +33 | +440/−491 | 0.10 |
| `flip+diag1` | 10 | 43.96% | 972 | +40 | +465/−521 | 0.080 |
| `flip+box1` | 18 | 45.08% | 981 | +49 | +579/−579 | 1.00 |
| `flip+cross2` | 18 | 44.14% | 979 | +47 | +444/−491 | 0.13 |
| `flip+cross1e` | 10 | 45.24% | 1,006 | +74 | +457/−449 | 0.82 |

The single flip is worth **+3.08 pp for 12 bytes** — the incumbent is a bargain
and is confirmed. Everything past it is flat or negative. No artifact was
emitted for this band.

**k = 32, 8-bit — the 3.9 KB point.** The same family, four times as wide, and
the answer flips:

| schedule | views | val | bytes | Δ bytes | paired | p |
|---|---:|---:|---:|---:|---|---:|
| `flip` (incumbent) | 2 | 58.48% | 3,940 | — | — | — |
| `flip+cross1` | 10 | 60.10% | 3,989 | +49 | +307/−226 | <0.001 |
| `flip+diag1` | 10 | 59.20% | 3,986 | +46 | +345/−309 | 0.17 |
| **`flip+box1`** | 18 | **61.00%** | 3,998 | +58 | +439/−313 | <0.001 |

**+2.52 pp for 58 bytes.** That is the largest TTA gain measured anywhere here,
and the size dependence within one family is the interesting part: with eight
filters the head sees a 72-dimensional feature and has no capacity left to
exploit extra views; with thirty-two it sees 288 and does.

Emitted: **`g-k32-p4s2-8b-tta`, 3,998 B, 61.37% on test**, against 3,938 B /
57.33%. Note that this artifact differs from its incumbent in two ways, not one
— the schedule *and* the lambda picked from the grid — so the +4.04 pp on test
is not all TTA. The TTA-only effect, at matched lambda selection, is the
+2.52 pp on validation.

### The trained CNNs (4 KB – 69 KB) — `experiments/tta_cnn.py`

Here the weights never move: only `predict.py` is re-emitted, so the byte cost is
the whole cost and the accuracy change is attributable to nothing else.

**`gc-cnntsmall-4b-qat`, 5,224 B** — the band where every schedule was tried:

| schedule | views | val | bytes | Δ bytes | paired | p |
|---|---:|---:|---:|---:|---|---:|
| `none` | 1 | 72.90% | 5,215 | −9 | +128/−210 | <0.001 |
| `flip` (incumbent) | 2 | 74.54% | 5,224 | — | — | — |
| `flip+cross1` | 10 | 75.14% | 5,263 | +39 | +108/−78 | 0.033 |
| `flip+diag1` | 10 | 75.44% | 5,262 | +38 | +179/−134 | 0.013 |
| **`flip+box1`** | 18 | **75.46%** | 5,273 | +49 | +154/−108 | 0.005 |
| `flip+cross2` | 18 | 75.08% | 5,270 | +46 | +114/−87 | 0.066 |
| `flip+box2` | 50 | 75.24% | 5,303 | +79 | +178/−143 | 0.058 |
| `flip+cross1e` | 10 | 74.76% | 5,289 | +65 | +89/−78 | 0.44 |
| `flip+box1e` | 18 | 75.08% | 5,301 | +77 | +126/−99 | 0.083 |
| `flip+box2e` | 50 | 75.36% | 5,331 | +107 | +155/−114 | 0.015 |

Emitted: **`gc-cnntsmall-4b-qat-tta`, 5,273 B, 74.67% on test**, against
5,224 B / 73.61%.

**The other three bands**, swept over the four schedules that survived above:

| artifact | bytes | schedule | val | Δ val | Δ bytes | p | test |
|---|---:|---|---:|---:|---:|---:|---:|
| `gc-cnntkd64-3b-qat` | 4,115 | `flip` | 70.88% | — | — | — | 71.26% |
| | 4,163 | `flip+box1` | 71.40% | +0.52 | +48 | 0.13 | **72.26%** |
| `gc-cnntkd64-4b-qat` | 5,231 | `flip` | 75.06% | — | — | — | 74.76% |
| | 5,280 | `flip+box1` | 75.42% | +0.36 | +49 | 0.31 | **75.65%** |
| `gc-cnnm-4b-qat` | 12,411 | `flip` | 80.68% | — | — | — | 80.74% |
| | 12,525 | `flip+cross1` | 81.64% | +0.96 | +114 | <0.001 | |
| | 12,435 | `flip+diag1` | 81.36% | +0.68 | +24 | 0.038 | |
| | 12,467 | `flip+box1` | 81.74% | +1.06 | +56 | <0.001 | **81.68%** |
| `gc-cnnlqat-4b-qat` | 39,798 | `flip` | 84.46% | — | — | — | 84.21% |
| | 39,839 | `flip+cross1` | 85.12% | +0.66 | +41 | 0.002 | **85.07%** |
| | 39,849 | `flip+diag1` | 85.10% | +0.64 | +51 | 0.024 | |
| | 39,853 | `flip+box1` | 85.04% | +0.58 | +55 | 0.026 | |
| `gc-cnnxlqat-4b-qat` | 71,006 | `none` | 84.34% | −0.96 | **+7** | 0.001 | |
| | 71,006 | `flip` | 85.30% | — | — | — | 85.37% |
| | 71,010 | `flip+cross1` | 85.78% | +0.48 | **+4** | 0.052 | **85.86%** |
| | 71,060 | `flip+box1` | 85.78% | +0.48 | +54 | 0.090 | |
| | 71,102 | `flip+box2` | 85.58% | +0.28 | +96 | 0.39 | |

At 69 KB the byte cost has stopped meaning anything: `flip+cross1` costs **four
bytes** over the incumbent, and *removing* TTA entirely costs seven bytes more
than keeping it, because xz would rather have the repetition. Above ~10 KB the
only question left is whether the gain is positive.

The gains are 0.4–1.1 pp everywhere and the byte costs are 4–56 B. Against the
local frontier slope — 1.27 pp per 100 B between the 3.9 KB and 5.2 KB points,
0.10 between 5.2 and 12.4 KB, and effectively zero above that — every one of
these trades pays several times over.

### Where it saturates, and where the wrap does not matter

Two negative results, both measured rather than assumed.

**Saturation is at radius 1.** `box2` (fifty views, {-2..2}²) is never better than
`box1` (eighteen views, {-1,0,1}²) at any size tested, and at 5 KB it is 0.22 pp
*worse* while costing 30 more bytes. Nine spatial positions is the whole of it.
Two-pixel shifts of a 32×32 image after three max-pools are simply a different
image.

**Edge padding does not pay for itself.** `np.roll` wraps a row of pixels from
the bottom of the image to the top, which is wrong, and an edge-replicating pad
is right and costs about 40 more bytes of source. At 5 KB, `flip+box1e` scores
75.08% against `flip+box1`'s 75.46% at 28 more bytes — worse on both axes. The
wrap is not a compromise; it is the better transform here. The likely reason is
that the training-time random crop pads with zeros, so a replicated edge is no
more familiar to the network than a wrapped one and costs bytes to produce.

## Lever 2 — distillation

`experiments/distill.py`. The teacher is the `xl` network already on disk,
`artifacts/_folded-xlqat.npz`, 85.50% float on validation. It is never shipped,
so whatever it is worth is worth zero artifact bytes.

### The closed-form ridge head

Ridge fits a linear map to targets and soft targets are still targets, so the
one-hot matrix is simply replaced by the teacher's probabilities. The Gram matrix
does not depend on the targets, so this is one extra solve per target set: the
cheapest possible test of the idea.

| targets | val (k=8, 4-bit, 931 B) | Δ vs hard | paired | p |
|---|---:|---:|---|---:|
| hard one-hot | 45.08% | — | — | — |
| teacher soft, T = 0.5 | 43.36% | −1.72 | +444/−530 | 0.006 |
| teacher soft, T = 1 | 43.44% | −1.64 | +449/−531 | 0.010 |
| teacher soft, T = 2 | 43.80% | −1.28 | +445/−509 | 0.041 |
| teacher soft, T = 4 | 42.48% | −2.60 | +455/−585 | <0.001 |
| half soft, T = 0.5 | 45.16% | +0.08 | +447/−443 | 0.92 |
| half soft, T = 1 | 44.74% | −0.34 | +394/−411 | 0.57 |
| half soft, T = 2 | 46.22% | +1.14 | +357/−300 | 0.029 |
| half soft, T = 4 | 44.84% | −0.24 | +294/−306 | 0.65 |

Pure soft targets are worse at every temperature, consistently and significantly.
The one positive cell is the 50/50 mix at T = 2, and it is the maximum of eight
configurations whose three neighbours in the same family are −0.34, +0.08 and
−0.24 — a flat family with one outlier, which is what noise looks like. **The
ridge answer is no.** It is not surprising: a linear map onto a 72-dimensional
random-feature space is capacity-bound, not target-bound, and the teacher's
extra information has nowhere to go.

### The trained CNN, against a matched twin

The student trains on random crops, so the teacher has to be asked about the
*same* crop or its logits describe a different image. Doing that live costs an
`xl` forward pass per step, which is most of the training budget. Instead K
augmented views per image are drawn once, the teacher scores them once, and
training samples from that cache. The hard-label twin trains through the
identical loop over the identical cache with the identical seed — the only
difference between the two runs is the loss. That is what makes the comparison
answer the falsifier rather than a confound.

Arch `t` (7,056 parameters), 30 epochs, 8 epochs of QAT at 4 bits, α = 0.9,
T = 4 — one setting, not tuned:

| cache | arm | float val | quantized val | bytes | test |
|---|---|---:|---:|---:|---:|
| K = 16 | hard-label twin | 73.98% | 72.18% | 5,300 | — |
| K = 16 | **distilled** | 75.68% | **73.70%** | 5,290 | 73.68% |
| K = 64 | hard-label twin | 75.18% | 74.54% | 5,289 | **75.31%** |
| K = 64 | **distilled** | 76.16% | **75.06%** | 5,302 | 74.76% |

At K = 16 distillation is worth **+1.52 pp** at matched bytes, exact McNemar
p = 0.012 — above the floor, resolved, a real effect.

At K = 64 it is worth **+0.52 pp**, p = 0.355 — below the floor, unresolved. And
on the test set the ordering **reverses**: the hard-label twin scores 75.31%
against the distilled student's 74.76%, at thirteen fewer bytes.

**The falsifier fires.** A distilled student does not beat its hard-label twin by
more than 1.06 pp at matched bytes, and its validation edge does not survive to
test. Distillation is retired as a lever for this frontier.

The K = 16 row says why, and it is the more useful result. The gain is not
"distillation adds knowledge"; it is "distillation substitutes for augmentation
diversity". Starve the student of crops and the teacher's soft labels recover
most of what was lost. Give it enough crops and there is nothing left to
recover. On this problem the crops are free and the teacher costs twenty minutes
of forward passes, so the crops win.

Arch `m` was not run. With the teacher at 85.50% float and arch `m` already at
82.24%, the teacher gap there is 3.3 points against arch `t`'s 9.2, so the
mechanism the K = 16 row identifies would have less room, not more. Running it
would have been sampling until the answer changed.

## Composition, and one artifact that is not what it looks like

Nine new points on this repo's own Pareto set, all scored on the full 10,000
test images through the harness. Each is marked with what produced it.

| bytes | test | artifact | from |
|---:|---:|---|---|
| 560 | 30.80% | `obt-g8-b9-tta` | TTA (`4px+flip`, 10 views) |
| 3,998 | 61.37% | `g-k32-p4s2-8b-tta` | TTA (`flip+box1`) + lambda on val |
| 4,115 | 71.26% | `gc-cnntkd64-3b-qat` | distilled student, 3-bit QAT |
| 4,163 | 72.26% | `gc-cnntkd64-3b-qat-tta` | the same, plus `flip+box1` |
| 5,231 | 74.76% | `gc-cnntkd64-4b-qat` | distilled student, 4-bit QAT |
| 5,280 | 75.65% | `gc-cnntkd64-4b-qat-tta` | the same, plus `flip+box1` |
| 12,467 | 81.68% | `gc-cnnm-4b-qat-tta` | TTA (`flip+box1`) |
| 39,839 | 85.07% | `gc-cnnlqat-4b-qat-tta` | TTA (`flip+cross1`) |
| 71,010 | 85.86% | `gc-cnnxlqat-4b-qat-tta` | TTA (`flip+cross1`) |

Two more artifacts were emitted and scored but are dominated:
`gc-cnntsmall-4b-qat-tta` (5,273 B / 74.67%) and `cnnthardtwin64-4b-qat`
(5,289 B / 75.31%), the distillation control.

The 3.9 KB row is the largest single improvement: **+4.04 points for 60 bytes.**

**Four of those rows say `distilled student`, and the accuracy is not the
distillation's.** The matched twin settles that: at 5,289 B it scores 75.31% on
test against the distilled student's 74.76% at 5,302 B. The four `cnntkd64` rows
are better than the shipped `cnntsmall` incumbents because they are a different
training run, not because of the loss. They are named for what produced them and
kept because they are genuinely the best artifacts at those sizes; nothing about
distillation should be read into them.

The twin makes that concrete in an uncomfortable way. It scores 75.31% on test
against the shipped incumbent's 73.61%, on identical architecture, epochs and
seed, with a validation accuracy of 74.54% — **identical** to the incumbent's. A
1.7-point test gap under an equal validation score is not a finding about the
64-view cache. It is what the repo's own caveat is for: differences of this size
between separately trained runs are fog.

TTA and distillation also partly overlap. On the non-distilled arch `t`
artifact, `flip+box1` is worth +0.92 pp on validation; on the distilled one,
+0.36 pp. Both are buying some of the same shift-invariance, so their gains do
not add.

## The verdict

**Keep TTA. It pays at every size measured but one, and the reason differs by
band.** Above 10 KB the source cost rounds to nothing — four bytes at 69 KB — so
any positive gain is worth taking. Between 4 and 40 KB it costs 40–60 bytes and
buys 0.5 to 2.5 points, which beats the local frontier slope by several times.
At 470 B it costs 90 bytes for 2.6 points, which *loses* against the joint
frontier's slope but wins outright, because the table family has nothing else to
spend bytes on: its own depth sweep saturates at b = 9.

The one band where it does not pay is **931 B**, the conv-ridge point, where the
incumbent single flip is already the whole of the effect (+3.08 pp for 12 bytes)
and every schedule beyond it is flat or negative. That the same family at 3.9 KB
gains +2.52 pp from the same schedule is the sharpest statement of the size
dependence: the lever is not a property of the transform, it is a property of
how much capacity the head has left to use it.

Saturation is at radius 1 — nine spatial positions with a flip — and it is the
same at every size. `flip+box1` wins below 12 KB and `flip+cross1` above it, and
the two are within 0.1 pp of each other everywhere; either is defensible. What
is settled is that two-pixel shifts and edge padding both measured worse, so
there is nothing further out to chase.

**Retire distillation.** Its falsifier fired: +0.52 pp over a matched twin at
K = 64, p = 0.355, and the test ordering reversed. What it does buy is a
substitute for augmentation the students already have for free. Item 5 of
Tier 1 in `what-to-try.md` is closed.

Item 4 — harder TTA — is closed too, but as a success: the schedule is settled at
`flip+box1`, the saturation point is measured, and the remaining upside from
this lever is zero. Both items should come off the list.

## Reproducing

```bash
# TTA, by family. --emit scores the named schedule on test, once.
python experiments/tta_cnn.py gc-cnntsmall-4b-qat            # all schedules
python experiments/tta_cnn.py gc-cnnm-4b-qat gc-cnnlqat-4b-qat \
    --scheds flip flip+cross1 flip+diag1 flip+box1
python experiments/tta_cnn.py gc-cnnxlqat-4b-qat --scheds flip flip+cross1 \
    --emit flip+cross1 --suffix=-tta
python experiments/tta_ridge.py --k 8  --bits 4              # 931 B
python experiments/tta_ridge.py --k 32 --bits 8 --scheds flip flip+box1 \
    --final flip+box1 --name=g-k32-p4s2-8b-tta               # 3.9 KB
python experiments/tta_table.py --emit "4px,flip,shipped" --name obt-g8-b9-tta

# the byte-for-byte alternative in the 470 B band
python experiments/oblivious_table.py --grids 8 --depth 12 --bmin 8 --orders 12

# distillation: ridge in seconds, CNN in about ten minutes per arm
python experiments/distill.py --mode ridge --k 8 --bits 4
python experiments/distill.py --arch t --tag kd64      --views 64 --alpha 0.9
python experiments/distill.py --arch t --tag hardtwin64 --views 64 --alpha 0 --no-eval
python experiments/distill.py --mode compare \
    --compare cnnthardtwin64-4b-qat cnntkd64-4b-qat

# compose: golf the decoder, then add TTA
python experiments/golf_cnn.py cnntkd64-4b-qat
python experiments/tta_cnn.py gc-cnntkd64-4b-qat --scheds flip flip+box1 \
    --emit flip+box1 --suffix=-tta
```

The teacher's 64-view logit cache is written to
`artifacts/_teacher-xl-v64.npz` on first use and reused after, which is what
makes the distilled run and its twin see identical crops.
