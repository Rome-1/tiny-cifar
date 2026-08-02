# Public CIFAR-10 size↔accuracy datapoints

A sourced set of published points for plotting a Pareto frontier of bytes against
accuracy. Every row was checked against a primary source during this pass unless
it says otherwise. The point of the document is the *byte* column, and the byte
column is the one the literature is worst at reporting — read the caveats before
plotting anything.

## How to read this — why cross-paper byte comparisons are treacherous

The papers do not agree on what a "model size" is, and most of them never say.
Four distinct things get called the same number:

1. **A reported compressed-file size.** MIRACLE and Entropy Penalized
   Reparameterization actually encode the weights and report the length of the
   resulting bitstream. This is the only figure that means what we mean by bytes.
2. **params × bits/param, computed by the authors.** µNAS states its convention
   outright — "8 bits = 1 byte per parameter" — so its 11.4 KB is a parameter
   count wearing a byte costume. Faithful, but it charges nothing for the
   architecture, the quantization scale factors, or the interpreter.
3. **A deployed artifact size.** MLPerf Tiny's 96 KB is the size of a `.tflite`
   file, which *does* include graph structure and metadata. It is therefore not
   comparable, byte for byte, to a params×8-bit figure of the same magnitude.
4. **Nothing at all.** ResNet, WideResNet, DenseNet, LEMONADE, TWN, BinaryConnect,
   BNN and airbench report parameter counts and never mention bytes. Any byte
   figure for them is an assumption about a precision they did not commit to.

Three further traps, all live in this table:

- **Architecture is free everywhere in the literature and is not free here.** Our
  harness measures `predict.py` along with the weights. Every published number
  below excludes its own decoder, its layer graph, and its runtime. A 11.4 KB
  µNAS model is 11.4 KB of weights plus a TFLite Micro interpreter nobody counts.
  This is the single largest systematic bias in the table, and it runs against us.
- **Sub-kilobyte CIFAR-10 results are usually not 10-class.** Confirmed below.
- **Test-time augmentation and extra training data** move accuracy by points and
  are flagged per row.

Units: where a paper states "101 KB" the string is preserved as reported. Where a
byte count is derived here, KB = 1024 B and the arithmetic is shown.

---

## Band 1 — sub-100 KB, MCU-class

### µNAS (Liberis, Dudziak, Lane, 2020) — the number to beat

**VERIFIED** at <https://ar5iv.labs.arxiv.org/html/2010.14246>, Table 2.

| Task | Acc. | Model size | RAM | MACs |
|---|---|---|---|---|
| CIFAR-10, **10-class** | **86.49%** | **11.4 KB** | 15.4 KB | 384 K |
| CIFAR-10, **2-class** | 77.49% | **685 B** | 909 B | 41.2 K |

Three corrections to how this result has been passed around:

- The survey's "86.49% @ 11.4 KB" **holds**. The 11.4 K figure is the *model size*
  column; 15.4 KB is peak RAM, a different quantity. Do not swap them.
- The byte figure is the authors' own params×int8 arithmetic, not a measured file:
  *"to compute the storage requirement, µNAS simply counts the number of parameters
  of a neural network at 8 bits = 1 byte per parameter."* Call it reported, but know
  what it is made of.
- **The claim that Appendix B excludes sparsity-mask storage did not survive
  checking.** No such sentence was found. What Appendix B does say is that for
  unstructured-sparse models "latency and peak memory usage constraints would no
  longer be accurate and are not included." The mask-storage claim in
  `method-survey.md` should be treated as unsupported until someone finds the line.
  The 11.4 KB dense row is unaffected either way.

### The 2-class question — confirmed

The survey's reading is **correct**. µNAS: *"We also consider 'binary' versions of
Chars74K and CIFAR-10, which have images partitioned into two subclasses."*
SpArSe (<https://ar5iv.labs.arxiv.org/html/1905.12107>): *"we also report on binary
versions of these datasets, meaning that the classes are split into two groups and
re-labeled."* Every sub-kilobyte CIFAR result located — µNAS 685 B / 77.49%,
SpArSe 0.78 KB / 73.84% (Table 3), SpArSe 487 params / 73.08% (Table 2) — is on the
2-class task, where **chance is 50%, not 10%**. No 10-class sub-*kilobyte* CIFAR-10
number was found.

Both quotes above were re-fetched and re-confirmed independently in the falsification
pass of 2026-08-02 (µNAS <https://ar5iv.labs.arxiv.org/html/2010.14246>, SpArSe
<https://ar5iv.labs.arxiv.org/html/1905.12107>). SpArSe reports **no** 10-class
CIFAR-10 result in any table. That part of this document stands.

**What did not survive that pass: the claim that the 10-class cell under 10 KB is
empty. It is not. See the next section.**

### Müksch et al. 2020 — the sub-10 KB 10-class cell is NOT empty

**VERIFIED** at <https://ar5iv.labs.arxiv.org/html/2005.04968> (Table 7) and
<https://arxiv.org/abs/2005.04968> (metadata). Müksch, Olausson, Wilhelm &
Andreadis, *Quantitative Analysis of Image Classification Techniques for
Memory-Constrained Devices*, arXiv 11 May 2020. arXiv preprint, 9 pp., derived from
an Edinburgh MSc thesis — **not peer-reviewed**, but public and specific.

It is unambiguously **10-class CIFAR-10 on the full 10,000-image test set**: *"The
CIFAR-10 data set consists of 60,000 32×32 3-channel colour images, divided into 10
classes… It comes split into 50,000 images for training and 10,000 for testing."* The
held-out validation set is carved out of **train** (*"we join the training batches
into one set and then extract a hold-out validation set consisting of exactly 1000
images of each class"*), and model selection is done on it: *"The validation set is
used to set, for example, hyper-parameters and to select the best models for each
algorithm… This ensures that the final test set accuracy reported for each model is
an unbiased estimate of its generalisation accuracy."* That is a cleaner protocol
than most of our own early rows.

Table 7, caption verbatim: *"Test set accuracies for methods described in Section 2
for different memory size budgets. Actual model size given in square brackets."*
The sub-10 KB entries:

| method | accuracy | stated size |
|---|---:|---:|
| Direct Convolution (3-channel) | **60.4%** | **5.39 KB** |
| Direct Convolution (3-channel) | 62.9% | 8.65 KB |
| FastGRNN (channel-major) | 48.2% | 7.57 KB |
| FastGRNN (row-major) | 47.1% | 7.57 KB |
| Multi-FastGRNN | 44.7% | 7.94 KB |
| Bonsai | 14.9% | 7.88 KB |

Direct Convolution is not theirs — it is Gural & Murmann (ICML 2019), reimplemented
here (*"Direct Convolution neural network proposed in [7]"*, *"Memory used to store
the pixels of an input feature map is progressively replaced with the activations of
the layer"*). Gural & Murmann's own paper reports **MNIST only, no CIFAR-10**; this
paper is what puts the method on 10-class CIFAR-10.

**Byte-accounting caveats, and why they do not rescue the empty-cell claim:**

- The paper **never defines what the KB figure measures** — weights only, or weights
  plus the activation buffer. It inherits Gural & Murmann's framing, whose metric is
  total inference SRAM (weights *and* activations, reported separately and summed).
  This ambiguity runs in the *counterexample's* favor, not ours: if 5.39 KB includes
  activations, the weights alone are **smaller** than 5.39 KB. Under either reading
  the model is under 10 KB.
- **Precision is never stated** for the CNN rows. If the weights are fp32, an int8
  version would be roughly 4× smaller again.
- As with every other published row here, the figure **excludes architecture and
  inference code**, which our harness charges us for.

So the honest statement is not "the cell is empty" — it is that **we dominate it**.
Our 61.37% in 3.9 KB beats their 60.4% in 5.39 KB on both axes; our 75.65% in 5.2 KB
beats it by 15 points at less size; our 50.78% in 1.7 KB beats FastGRNN's 48.2% in
7.57 KB on both axes; our 43.79% in 931 B beats Bonsai's 14.9% in 7.88 KB. That is a
comparison rather than an absence, and it is the stronger thing to say.

### µNAS's LEMONADE row — a published 10-class entry printed at 10 K

Also found in the falsification pass, and also fatal to "the cell is empty" as
literally worded. µNAS's **own Table 2**, in the 10-class CIFAR-10 block, prints a
comparison row: **LEMONADE, ≈91.77%, model size 10 K** (RAM and MACs "unk."). At
µNAS's stated 1 byte/parameter convention that is 10,000 B — under 10 KB.

**That row is not supported by LEMONADE's own paper.** Verified by extracting the
text of <https://arxiv.org/pdf/1804.09081>: Table 1 spans 0.5 M–13.1 M params, and
Table 2 (*"Comparison between LEMONADE (SS-I), Random Search, NASNet, MobileNet and
MobileNet V2 on CIFAR-10 for different model sizes"*) bottoms out at **NASNet 38 K /
12.0%** and **LEMONADE 47 K / 8.9%**. There is no 10 K row anywhere. What the paper
does say is that the search's Pareto front covers *"model parameters, ranging from
10 000 to 10 000 000"* — but that is Figure 2, whose axis is labeled **"Validation
error"**, on the 5,000-image validation split (*"The training set is split up in a
training (45.000) and a validation (5.000) set"*), not the 10,000-image test set.
Figure 5 *is* test error (*"Performance on CIFAR-10 test data"*) and its y-axis does
reach 10⁴, but no numeric value is printed for that point, and a Pareto front cannot
put 10 K params at a *lower* error than the 47 K / 8.9% row in the same paper.

Verdict: treat µNAS's LEMONADE 10 K row as **unsupported by its cited source** — a
textbook case of this document's own rule, *never take a size from a rival's
comparison row*. But it exists in print, in the very paper we cite as our 11.4 KB
anchor, and any reader who checks will find it. A universal negative over the
literature cannot survive it.

### Two negative results, both useful

- **MCUNet reports no CIFAR-10 at all**, and says why: *"We did not use datasets
  like CIFAR since it is a small dataset with a limited image resolution (32×32),
  which cannot accurately represent the benchmark model size or accuracy in real-life
  cases."* (<https://ar5iv.labs.arxiv.org/html/2007.10319>, §4.1.) Stop citing it.
- **No citable TFLite-Micro CIFAR-10 example exists.** The `tensorflow/tflite-micro`
  examples are `micro_speech` and `person_detection`. The MicroNet Challenge track
  was **CIFAR-100**, not CIFAR-10. The 100 KB–1 MB band has thinner sourcing than
  it looks like it should.

## Band 2 — 100 KB to 1 MB

The strongest rows in the whole document, because these papers actually encode
their weights and report the bitstream length.

- **Entropy Penalized Reparameterization**, VGG-16: **101 KB at 90.0%** (10.0%
  error). Table 1, <https://ar5iv.labs.arxiv.org/html/1906.06624>. Reported bytes.
- **MIRACLE**, VGG-16: **135 KB at 90.0%**, and **384 KB at 93.43%** (6.57% error).
  Table 1, <https://ar5iv.labs.arxiv.org/html/1810.00440>. Reported bytes.
  *Discrepancy worth knowing:* EPR's comparison table cites MIRACLE at **168 KB**
  for the same 10.0% operating point that MIRACLE's own table calls 135 KB. Cite
  each paper for itself; never take a size from a rival's comparison row.

**Deep Compression reports no CIFAR-10** — *"two on MNIST and two on ImageNet"*
(<https://ar5iv.labs.arxiv.org/html/1510.00149>). The recollection was right; there
is no Deep Compression CIFAR row to put in the table.

**Binary and ternary nets are not small.** This is counterintuitive and it matters:
TWN, BinaryConnect and BNN all use a VGG-7-shaped net whose fully-connected layer
alone is 8.4 M parameters. One bit per weight on 14 M weights is still 1.75 MB.
Extreme quantization buys a constant factor; it does not buy a small model. None of
these papers reports a parameter count or a byte size — the counts below are derived
here from their stated layer specs.

## Band 3 — 1–50 MB, the upper anchors

Standard references, all parameter counts as printed in the source papers, all byte
figures derived at fp32 because none of these papers ships a quantized artifact.

- ResNet-20/32/56/110 — He et al., Table 6, <https://ar5iv.labs.arxiv.org/html/1512.03385>.
  Note ResNet-110's headline 6.43% error is *best of 5 runs*; the mean is 6.61 ± 0.16.
- WRN-28-10 — 4.00% error, 36.5 M params, Table 5,
  <https://ar5iv.labs.arxiv.org/html/1605.07146> (moderate augmentation, no dropout).
- DenseNet-BC — Table 2, <https://ar5iv.labs.arxiv.org/html/1608.06993>.
- **hlb-CIFAR10** (<https://github.com/tysam-code/hlb-CIFAR10>): the README claims
  *94% in <6.3 s on a single A100* and *~95.79% in ~110 s*. **The parameter count is
  stated nowhere** — not in the README, not in `main.py`. The survey's ~4.34 M was
  re-derived independently this pass by summing the layers in `main.py`
  (`SpeedyConvNet`: a 2×2 whitening conv 3→12, then ConvGroups at 24/64/64,
  64/256/256, 256/512/512 with BatchNorm, and a bias-free Linear(512,10)) →
  **≈ 4,337,180 params**, hence ≈ 17.3 MB at fp32. The two derivations agree, so the
  survey's number stands — but it is a derivation twice over, not a published fact.
- **airbench94** (<https://ar5iv.labs.arxiv.org/html/2404.00498>): **1.97 M params
  (reported)**, 94.01% — **and the headline uses TTA** (horizontal flip plus two
  crops). Without TTA it is 93.2%. The survey's "~2 MB" was a size guess; the paper
  gives params, so use 1.97 M and derive.

## Band 4 — simple non-deep baselines, the low end

- **Logistic regression on raw pixels, 41.13%** (unwhitened) / **37.32%** (whitened) —
  Krizhevsky 2009 TR, Fig. 3.1, <https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf>.
  Whitening *hurts* a raw-pixel linear classifier. 3072×10 + 10 = 30,730 params.
- **Linear SVM on raw pixels, 49.88%** — Abouelnaga et al., Table II,
  <https://arxiv.org/abs/1611.04905>. Same 30,730-parameter shape, eight points
  better than logistic regression. This is the best value-per-byte non-deep row
  found and the more useful low-end anchor.
- **k-NN, and why the 38.6% figure is weak.** CS231n does state *"this classifier
  only achieves 38.6% on CIFAR-10"* for 1-NN with L1 (35.4% with L2),
  <https://cs231n.github.io/classification/> — so the number is real but the source
  is course notes, not peer review. The peer-reviewable substitute is Abouelnaga
  Table II: **kNN on raw pixels 33.86%**, **kNN + PCA-30 41.78%**. Also: CS231n
  gives **no** accuracy for k=7 or k=10; any such figure in circulation is
  unsupported. Mark the whole k-NN family weak.
- **Coates, Lee & Ng 2011**, Table 1, <https://proceedings.mlr.press/v15/coates11a/coates11a.pdf>:
  K-means (triangle) 1600 features → **77.9%**; 4000 features → **79.6%**;
  K-means (hard) 1600 → 68.6%. Configuration: ZCA whitening, 6-pixel receptive field,
  **stride 1**, 4-quadrant pooling, linear L2-SVM. Feature dimension is 4×K.

---

## Fairness flags — where a comparison to us would be dishonest

| Row | Why it is not a fair comparison |
|---|---|
| µNAS 685 B, SpArSe 0.78 KB | **2-class relabeling; chance is 50%.** Plot in a separate series or not at all. |
| 1-NN / kNN / kNN+PCA | **Transductive.** The model *is* the training set: 50,000 × 3072 = 153.6 MB of pixels must be shipped. The PCA variant still stores 50,000 projected vectors. |
| airbench94 (94.01%) | **TTA** (flip + 2 crops). No-TTA figure is 93.2%. Our harness allows TTA too, so this one is fair *if* the TTA is disclosed on both sides — which is why it is flagged rather than excluded. |
| LEMONADE | Trained with **mixup and Cutout** (§A.3.4). From scratch at 32×32, so otherwise comparable. |
| ViT-H/14 99.50% | **Pretrained on JFT-300M**, ~303 M extra images. An upper bound on the plot's y-axis, nothing more. |
| Every row without a reported byte count | **Architecture, graph and runtime excluded.** Our artifact pays for its own source; theirs do not. |
| MobileNet / ShuffleNet / SqueezeNet "93–98% CIFAR-10" figures in circulation | ImageNet-pretrained and fine-tuned on **upsampled 224×224** input. Excluded from this document entirely. |
| µNAS unstructured-sparse rows | The paper states latency and peak-memory constraints "are not included" for these. Only the dense 11.4 KB row is used here. |
| LEMONADE @ 10 K, as printed in µNAS Table 2 | **Unsupported by the cited source.** LEMONADE's own tables bottom out at 47 K params / 91.1%. Record it, do not plot it as a rival. |
| Müksch et al. rows (Direct Conv, FastGRNN, Bonsai) | **Size basis undefined** — may be weights-only or weights+activations, and precision is never stated. The ambiguity runs *for* them, not us: either reading leaves them under 10 KB. Also **not peer-reviewed** (arXiv preprint from an MSc thesis). Fair to plot, with the caveat that, like every other published row, it excludes inference code. |

---

## The table

Sorted by bytes ascending. `R` = byte figure reported by the authors; `D` = derived
here, with the arithmetic in the footnote. Accuracy is top-1 on the full 10,000-image
CIFAR-10 test set unless flagged.

| model | accuracy | params | bytes | bytes source | year | citation | verified? |
|---|---:|---:|---:|---|---:|---|---|
| µNAS (2-class CIFAR) ⚠ | 77.49% | — | 685 | R (params×int8) | 2020 | [arXiv:2010.14246](https://ar5iv.labs.arxiv.org/html/2010.14246) T2 | verified |
| SpArSe (2-class CIFAR) ⚠ | 73.84% | — | ~799 (0.78 KB) | R | 2019 | [arXiv:1905.12107](https://ar5iv.labs.arxiv.org/html/1905.12107) T3 | verified |
| **Direct Convolution CNN (Müksch et al.)** | **60.40%** | — | **5,519 (5.39 KB)** | R ("actual model size"; basis undefined) | 2020 | [arXiv:2005.04968](https://ar5iv.labs.arxiv.org/html/2005.04968) T7 | verified |
| FastGRNN (channel-major, Müksch et al.) ⚠ | 48.20% | — | 7,752 (7.57 KB) | R (basis undefined) | 2020 | [arXiv:2005.04968](https://ar5iv.labs.arxiv.org/html/2005.04968) T7 | verified |
| Bonsai (Müksch et al.) ⚠ | 14.90% | — | 8,069 (7.88 KB) | R (basis undefined) | 2020 | [arXiv:2005.04968](https://ar5iv.labs.arxiv.org/html/2005.04968) T7 | verified |
| Direct Convolution CNN (Müksch et al.) | 62.90% | — | 8,857 (8.65 KB) | R (basis undefined) | 2020 | [arXiv:2005.04968](https://ar5iv.labs.arxiv.org/html/2005.04968) T7 | verified |
| LEMONADE, as cited by µNAS ⚠ unsupported | ≈91.77% | 10 K | 10,000 | R (µNAS's row, not LEMONADE's) | 2020 | [arXiv:2010.14246](https://ar5iv.labs.arxiv.org/html/2010.14246) T2 | verified as printed; **contradicted by the cited source** |
| **µNAS** | **86.49%** | 11.4 K | **11,400** | R (params×int8) | 2020 | [arXiv:2010.14246](https://ar5iv.labs.arxiv.org/html/2010.14246) T2 | verified |
| Logistic regression on raw pixels | 41.13% | 30,730 | 30,730 | D — int8 [1] | 2009 | [Krizhevsky TR](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf) Fig. 3.1 | verified |
| **Linear SVM on raw pixels** | **49.88%** | 30,730 | **30,730** | D — int8 [1] | 2016 | [arXiv:1611.04905](https://arxiv.org/abs/1611.04905) T2 | verified |
| LEMONADE-47K | 91.10% | 47 K | 47,000 | D — int8 [2] | 2018 | [arXiv:1804.09081](https://ar5iv.labs.arxiv.org/html/1804.09081) T2 | verified (params) |
| hls4ml MLPerf-Tiny IC | 83.50% | 58,115 | 58,115 | D — int8 lower bound [3] | 2022 | [arXiv:2206.11791](https://ar5iv.labs.arxiv.org/html/2206.11791) T1 | verified (params) |
| MLPerf Tiny ResNet (TFLite int8) | 85% target | not stated | 98,304 (96 KB) | R (deployed `.tflite`) | 2021 | [arXiv:2106.07597](https://ar5iv.labs.arxiv.org/html/2106.07597) T1 | verified |
| **Entropy Penalized Reparam., VGG-16** | **90.00%** | — | **103,424 (101 KB)** | R (encoded bitstream) | 2019 | [arXiv:1906.06624](https://ar5iv.labs.arxiv.org/html/1906.06624) T1 | verified |
| MIRACLE, VGG-16 | 90.00% | — | 138,240 (135 KB) | R (encoded bitstream) | 2018 | [arXiv:1810.00440](https://ar5iv.labs.arxiv.org/html/1810.00440) T1 | verified |
| LEMONADE-190K | 94.50% | 190 K | 190,000 | D — int8 [2] | 2018 | [arXiv:1804.09081](https://ar5iv.labs.arxiv.org/html/1804.09081) T2 | verified (params) |
| Coates K-means (triangle) 1600 + SVM | 77.90% | 248,464 | 248,464 | D — int8 [4] | 2011 | [Coates AISTATS](https://proceedings.mlr.press/v15/coates11a/coates11a.pdf) T1 | verified |
| MIRACLE, VGG-16 (low-compression) | 93.43% | — | 393,216 (384 KB) | R (encoded bitstream) | 2018 | [arXiv:1810.00440](https://ar5iv.labs.arxiv.org/html/1810.00440) T1 | verified |
| Coates K-means (triangle) 4000 + SVM | 79.60% | 603,664 | 603,664 | D — int8 [4] | 2011 | [Coates AISTATS](https://proceedings.mlr.press/v15/coates11a/coates11a.pdf) T1 | verified |
| ResNet-20 | 91.25% | 0.27 M | 1,080,000 | D — fp32 [5] | 2015 | [arXiv:1512.03385](https://ar5iv.labs.arxiv.org/html/1512.03385) T6 | verified |
| BinaryConnect (stochastic), VGG-7-ish | 91.73% | ~14.02 M (derived) | ~1,752,752 | D — 1 bit/weight [6] | 2015 | [arXiv:1511.00363](https://ar5iv.labs.arxiv.org/html/1511.00363) T2 | acc verified; params derived |
| DenseNet-BC (k=12, L=100) | 95.49% | 0.8 M | 3,200,000 | D — fp32 [5] | 2016 | [arXiv:1608.06993](https://ar5iv.labs.arxiv.org/html/1608.06993) T2 | verified |
| TWN ternary, VGG-7 | 92.56% | ~12.97 M (derived) | ~3,243,360 | D — 2 bits/weight [7] | 2016 | [arXiv:1605.04711](https://ar5iv.labs.arxiv.org/html/1605.04711) T2 | acc verified; params derived |
| ResNet-56 | 93.03% | 0.85 M | 3,400,000 | D — fp32 [5] | 2015 | [arXiv:1512.03385](https://ar5iv.labs.arxiv.org/html/1512.03385) T6 | verified |
| kNN + PCA-30 ⚠ transductive | 41.78% | — | ~6,368,640 | D — fp32 store [8] | 2016 | [arXiv:1611.04905](https://arxiv.org/abs/1611.04905) T2 | verified |
| ResNet-110 (best of 5) | 93.57% | 1.7 M | 6,800,000 | D — fp32 [5] | 2015 | [arXiv:1512.03385](https://ar5iv.labs.arxiv.org/html/1512.03385) T6 | verified |
| airbench94 ⚠ TTA | 94.01% (93.2% no TTA) | 1.97 M | 7,880,000 | D — fp32 [5] | 2024 | [arXiv:2404.00498](https://ar5iv.labs.arxiv.org/html/2404.00498) | verified |
| hlb-CIFAR10 | ~94% | ~4.34 M (derived) | ~17,348,720 | D — fp32 [9] | 2023 | [github tysam-code](https://github.com/tysam-code/hlb-CIFAR10) | acc verified; params derived |
| DenseNet-BC (k=40, L=190) | 96.54% | 25.6 M | 102,400,000 | D — fp32 [5] | 2016 | [arXiv:1608.06993](https://ar5iv.labs.arxiv.org/html/1608.06993) T2 | verified |
| WRN-28-10 | 96.00% | 36.5 M | 146,000,000 | D — fp32 [5] | 2016 | [arXiv:1605.07146](https://ar5iv.labs.arxiv.org/html/1605.07146) T5 | verified |
| 1-NN, L1, raw pixels ⚠ transductive, weak source | 38.60% | — | 153,600,000 | D — training set [10] | 2016 | [CS231n notes](https://cs231n.github.io/classification/) | verified (course notes, not peer-reviewed) |
| ViT-H/14 ⚠ JFT-300M pretraining | 99.50% | 632 M | 2,528,000,000 | D — fp32 [5] | 2020 | [arXiv:2010.11929](https://ar5iv.labs.arxiv.org/html/2010.11929) T2 | verified |

⚠ marks a row flagged in the fairness table above.

### Derivations

1. `3072 × 10 + 10 = 30,730` params × 1 B (int8) = **30,730 B**. Neither source
   quantizes; int8 is our assumption and is generous to the baseline.
2. LEMONADE reports **parameters only, never bytes**. `47,000 × 1 B = 47,000 B`,
   `190,000 × 1 B = 190,000 B` at an assumed int8. The authors never quantized, so
   these are hypothetical artifacts, not things that exist. At fp32 they would be
   188 KB and 760 KB.
3. `58,115 × 1 B = 58,115 B` at int8. The paper says **8–12 bit** and gives no byte
   figure, so 58,115 B is a lower bound; at 12 bits it is 87,173 B.
4. Coates: dictionary `1600 × (6·6·3=108) = 172,800`, SVM head `6400 × 10 = 64,000`,
   ZCA whitening matrix `108 × 108 = 11,664`; total `248,464` × 1 B = **248,464 B**.
   For 4000 features: `4000×108 = 432,000` + `16000×10 = 160,000` + `11,664` =
   **603,664 B**. At fp32, ×4. The paper reports no size of any kind.
5. `params × 4 B` (fp32). None of these papers quantizes or reports bytes.
6. BinaryConnect's stated net is `(2×128 C3)-MP2-(2×256 C3)-MP2-(2×512 C3)-MP2-(2×1024 FC)-10`.
   Convs: `3456 + 147456 + 294912 + 589824 + 1179648 + 2359296 = 4,574,592`.
   After three 2× pools the map is 4×4×512 = 8192, so FC1 = `8192×1024 = 8,388,608`,
   FC2 = `1024×1024 = 1,048,576`, FC3 = `1024×10 = 10,240`. Total **≈14.02 M**.
   × 1 bit / 8 = **1,752,752 B**. Params are our arithmetic; the paper gives none.
7. TWN's VGG-7 is `2×(128-C3) + MP2 + 2×(256-C3) + MP2 + 2×(512-C3) + MP2 + 1024-FC`.
   Same convs (4,574,592) with a single FC `8192×1024 = 8,388,608` and `1024×10 =
   10,240` → **≈12.97 M**. Ternary packs at 2 bits: `12,973,440 × 2 / 8 =
   3,243,360 B`. An ideal entropy coder at log₂3 = 1.585 bits would reach ~2.57 MB.
   The paper reports neither params nor bytes, only a generic "up to 16×" claim.
8. `50,000 × 30 × 4 B = 6,000,000` for the stored projected training set, plus the
   PCA basis `30 × 3072 × 4 B = 368,640`. Excludes labels. Transductive.
9. `4,337,180 × 4 B = 17,348,720 B`. Param count derived from `main.py`, not stated.
10. `50,000 × 32 × 32 × 3 = 153,600,000 B` of uint8 training pixels. The training set
    *is* the model.

---

## What this says about where we stand

µNAS at 86.49% in 11.4 KB is the strongest 10-class published point near our size,
and it still beats us there.

**Below 10 KB the literature is thin, but it is not empty — an earlier version of
this section said it was, and that was wrong.** Müksch et al. put five 10-class,
full-test-set CIFAR-10 points in the 5–9 KB band (arXiv:2005.04968, Table 7). The
correct claim is dominance, not absence:

| our artifact | their best comparable |
|---|---|
| 61.37% in 3.9 KB | 60.4% in 5.39 KB — beaten on **both** axes |
| 75.65% in 5.2 KB | 60.4% in 5.39 KB — +15.3 points at less size |
| 50.78% in 1.7 KB | FastGRNN 48.2% in 7.57 KB — beaten on both axes |
| 43.79% in 931 B | Bonsai 14.9% in 7.88 KB — beaten on both axes |

That is the headline the plot should make, and it is a better one than the empty
cell was: a measured comparison beats an unfalsifiable absence. Two asymmetries run
in opposite directions and both belong in any caption. Against us: every published
byte count here, Müksch's included, excludes the architecture and runtime that our
harness charges for, and Müksch's figure may already include activations. For us:
the sub-10 KB published points are dominated on both axes, and the 30.7 KB
linear-SVM row is dominated outright.
