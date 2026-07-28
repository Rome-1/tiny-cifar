# Exotic methods: the non-obvious space for a byte-minimal CIFAR-10 classifier

Author: rissanen. Date: 2026-07-28. Companion to `docs/method-survey.md`.

**Scope.** This document deliberately *excludes* the mainstream families (tiny/depthwise
CNNs, int8/int4/ternary/binary quantization, Deep-Compression clustering+Huffman,
magnitude pruning, HashedNet, vanilla distillation, PCA+head) — another agent owns those.
Everything here is the weird stuff: procedural weights, supermasks over seeded PRNGs,
hypernetworks, tensor decompositions, structured transforms, coding-theoretic training
objectives, and dataset distillation. Several of the most-hyped ideas in this space **do
not survive the byte arithmetic**, and saying so precisely is half the value of this
document.

**Verification convention.** *[verified]* = read from a source fetched during this
session, URL given. *[UNVERIFIED RECALL]* = model memory, not checked. *[derived]* =
arithmetic I performed. *[figure-read]* = digitized off a plot, ±0.5.

---

## Part 0 — Three accounting facts that decide everything below

### 0.1 The information budget is fixed and small

The entire supervised signal in CIFAR-10 is 50,000 labels x log2(10) = **166 kbit =
20.8 kB** *[derived]*. That is a hard ceiling on how much label information any artifact
could possibly need to memorize, and generalizing classifiers need far less.

Blier & Ollivier, *The Description Length of Deep Learning Models* (arXiv 1802.07044,
NeurIPS 2018) *[verified, Table 1]* measured this directly. Codelength of the CIFAR-10
**labels given the images**:

| Coding scheme | Codelength | Accuracy |
|---|---|---|
| Uniform | 166 kbit | 10% |
| float32 two-part code | >428 **M**bit | — |
| Variational | 89.0 kbit | 66.5% |
| **Prequential (VGGb)** | **45.3 kbit = 5.66 kB** | **93.3%** |

This is the single most important number in this document. It says the label information
needed for **93.3% accuracy is ~5.7 kB**. Prequential coding is not a two-part code — it
does not hand you a static artifact — but it establishes that a ~6 kB artifact at ~90% is
not information-theoretically absurd. It should be the project's north star, and it is
roughly 10x below the current published state of the art in bytes (see 0.3).

**Recommendation:** report a *secondary* MDL score alongside artifact bytes:
`L = |artifact| + L(train labels | artifact, train images)`. That is Rissanen's two-part
code and it is the only metric under which "ship a bigger model" and "ship a lookup table"
are penalized consistently.

### 0.2 The free/paid boundary must be declared before any method is chosen

Three self-consistent regimes, which price methods *completely differently*:

- **Regime A — strict self-containment.** The artifact runs with no external data. Any
  dataset-derived constant (whitening matrix, k-means dictionary, class means) must be
  shipped in full.
- **Regime B-unlabeled — training *images* are a shared public resource** available to
  both encoder and decoder; labels are not. Then **any deterministic function of the
  unlabeled training images costs zero bytes**: ZCA/PCA bases, k-means dictionaries,
  patch statistics, self-supervised features. You pay only for the label-dependent part.
- **Regime B-labeled — the full labeled training set is free.** This is **degenerate and
  must be rejected**. Under it, a ~200-byte program that reads the training set, trains
  hlb-CIFAR10, and classifies scores 94% at 200 bytes. The benchmark ceases to exist.

**My view: run Regime A as the headline metric, and report Regime B-unlabeled as a clearly
labeled second column.** Regime A is the one that is unambiguously defensible and
comparable to the TinyML/compression literature. Regime B-unlabeled is scientifically
interesting (it isolates "how many bits is the *supervision* worth") and is the natural
setting for the Blier–Ollivier anchor above, but it is a different game and must never be
plotted on the same axis without a label.

Practical consequence in Regime A: use **patch-level** whitening, not full ZCA. hlb's 2x2
patch-eigenvector conv is 3x12x2x2 = 144 params. A full 3072x3072 ZCA matrix is 9.4M
floats and is unshippable *[derived]*.

### 0.3 The frontier we must beat (verified anchors, 10-class CIFAR-10)

| Artifact | Bytes | Accuracy | Source |
|---|---|---|---|
| µNAS (8-bit, **not** entropy coded) | **11.4 kB** | **86.5%** | arXiv 2010.14246 Table 2 *[verified]* |
| LilNetX ResNet-20-4 | 66 kB | 91.5% | arXiv 2204.02965 Table 1 *[verified]* |
| LilNetX VGG-16 (extreme) | 76 kB | 90.0% | same |
| Oktay EPR VGG-16 | 101 kB | 90.0% | arXiv 1906.06624 *[verified]* |
| MIRACLE VGG-16 (corrected) | 168 kB | 90.0% | arXiv 1810.00440 + OpenReview correction *[verified]* |
| Bao et al., best dataset distillation *by bits* | 94 kB | 77.5% | arXiv 2507.17221 (ICCV'25) *[verified]* |

Two corrections to widely-repeated folklore, both verified this session:

- **SpArSe's famous "CIFAR-10 at 3.2 kB" is a *binary* (2-class) result** — the paper says
  so in Section 3, chance is 50%, and the 3.2 kB figure is not in any table. Do not cite
  it. Actual: 73.84% on CIFAR10-**binary** at 0.78 kB model size *[verified, NeurIPS PDF]*.
- **MCUNet reports no CIFAR-10 at all** *[verified]*.

**The open seam** (independently identified by two of my verification threads): the
NAS/TinyML side reaches ~11 kB by shrinking the architecture and never entropy-codes; the
compression side reaches ~66 kB by coding a large architecture and never starts from a
tiny one. **Nobody has arithmetic-coded a µNAS-scale network.** 11,400 params at an
achievable 2-3 bits/param is **3-4 kB at ~86%** — which would be a 3x improvement on the
published frontier from composition alone.

---

## Part 1 — Supermasks over seeded random networks

### 1.1 Mechanism

Fix a PRNG seed; the decoder regenerates the entire weight tensor from it, so the weight
*values* cost zero bytes. Training learns only which weights survive — a binary mask.
Ramanujan et al.'s edge-popup keeps the top-k% of weights by a learned score; the shipped
artifact is `seed + mask`. Diffenderfer & Kailkhura's biprop additionally binarizes
(sign only) and adds one fp scalar per layer.

### 1.2 The literature, verified

**Zhou et al. 2019, "Deconstructing Lottery Tickets" (arXiv 1905.01067)** *[verified,
Table 1]*. CIFAR-10, best supermask (learned mask + Dynamic Weight Rescaling, signed
constant): **Conv2 66.0, Conv4 72.5, Conv6 76.5**, versus trained baselines 69.2 / 75.4 /
78.3. Params: Conv2 4.3M, Conv4 2.4M, Conv6 2.3M. The abstract's "41% on CIFAR-10" is the
*heuristic*, no-training-at-all case. Note Ramanujan et al. cite Zhou as "65.4%", which
understates Zhou's own best by 11 points.

**Ramanujan et al. 2020, edge-popup (arXiv 1911.13299)** *[verified — and an important
negative]*: **the paper states no CIFAR-10 accuracy numerically anywhere.** All CIFAR-10
results are curves. Figure-3 peaks *[figure-read]*: Conv2 ~77.5, Conv4 ~86, Conv6 ~88,
Conv8 ~88.5, against learned-dense baselines of ~79.5 / 86.5 / 88.5 / 88. The commonly
cited "77.7 / 85.8 / 88.1" triple appears only in *Slot Machines'* Table 2. Verbatim on
sparsity: "The best accuracy occurs when k ∈ [30, 70]... the number of possible
subnetworks is maximized when k ≈ 0.5." ImageNet (Table 2, k=30%): Wide ResNet-50 with
Signed Kaiming Constant reaches **73.3%** at 20.6M params — matching a *trained* ResNet-34.
Signed Kaiming Constant (weights uniform over {−σ, +σ}) is worth 5-6 points over Kaiming
Normal, and it is what makes the "weights are free" story clean: one bit of sign per
weight, itself PRNG-generated.

**Diffenderfer & Kailkhura 2021 (arXiv 2103.09377)** *[verified]*. Correct title is
"Multi-Prize Lottery Ticket Hypothesis: Finding Accurate Binary Neural Networks by Pruning
A Randomly Weighted Network." CIFAR-10: MPT-1/32 ResNet-18 at 80% pruned → **94.66%** (2.2M
params); **VGG-Small at 95% pruned → 91.48% at 0.23M surviving params** (Appendix Table 8).
Shipped object per Algorithm 1: binary sign B = sign(W) (free, from seed) x binary mask M
x one fp per-layer scalar α. **Caveat:** the "+BN" variants learn BatchNorm parameters and
are worth up to 7 points on MPT-1/1 — the headline numbers are not strictly
weight-training-free. The paper's "32x memory saving" prices 1-bit weights only and
**never states bits/weight including the mask**.

**Chijiwa et al. 2021, IteRand (arXiv 2106.09269)** *[verified]*. Conv6 88.80 at 50%
sparsity, ResNet18 92.61. **But the paper explicitly states that saving an IteRand model
requires R seeds and R masks — strictly worse than plain seed+mask.** Rule it out for us.

**Aladago & Torresani, Slot Machines (arXiv 2101.06475)** *[verified]*: select one of K
random values per connection. CIFAR-10 CONV-6 88.4, VGG-19 91.7. Costs log2(K) ≈ 3
bits/connection at K=8 — 3x the mask cost for ~0.4 points over edge-popup. Bad trade.

**García-Arias et al., Hidden-Fold Networks (arXiv 2111.12330, BMVC 2021)** *[verified]* —
the paper closest to our framing, verbatim: "it is only necessary to store the supermask
and the seed for generating the random signs... the supermask only needs one bit per
weight." CIFAR-**100**: HFN-ResNet152 78.02% at **2.46 MB, 38.5x smaller** than a matched
ResNet50. No CIFAR-10.

**Gaier & Ha, WANN (arXiv 1906.04358)** *[verified]*: **no CIFAR-10 anywhere** — the string
"CIFAR" does not appear in the paper. Best MNIST is 91.9% on *16x16 deskewed* MNIST with
1849 connections. Interesting as philosophy, useless as a CIFAR-10 datapoint.

**Theory.** Malach et al. (arXiv 2002.00585) prove a random net of width poly(...) and
depth 2l contains a subnetwork ε-approximating any target of width n, depth l. Pensia et
al. (arXiv 2006.07990) sharpen this to a width factor of only **O(log(dl))** and show it is
essentially optimal. *[verified]* — and note the widely quoted O(d⁴l²) is Pensia's
characterization of Malach's bound, not Malach's own theorem.

### 1.3 The byte arithmetic — and why the hype does not survive it

The mask is *not* 1 bit/weight in general. At keep-fraction p and i.i.d. positions, an
optimal entropy coder spends **H(p) = −p·log2(p) − (1−p)·log2(1−p)** bits per weight
*[derived]*:

| p (kept) | H(p) bits/weight | surviving params per bit | random weights affordable at 1 kB | at 10 kB |
|---|---|---|---|---|
| 0.50 | 1.0000 | 0.500 | 8,192 | 81,920 |
| 0.30 | 0.8813 | 0.340 | 9,295 | 92,955 |
| 0.10 | 0.4690 | 0.213 | 17,467 | 174,671 |
| 0.05 | 0.2864 | 0.175 | 28,604 | 286,037 |
| 0.02 | 0.1414 | 0.141 | 57,918 | 579,183 |
| 0.01 | 0.0808 | 0.124 | 101,395 | 1,013,948 |

**Now price the published operating points** *[derived]*:

| Setting | Mask bits | Bytes | Accuracy | Verdict |
|---|---|---|---|---|
| edge-popup Conv4, 2.4M weights, k=50% | 2.4 Mbit | **293 kB** | ~86% *[figure-read]* | Dominated |
| Zhou Conv6 supermask, 2.3M, p~0.5 | 2.3 Mbit | **281 kB** | 76.5% | Badly dominated |
| MPT VGG-Small, 4.6M weights, 95% pruned | 1.32 Mbit | **161 kB** | 91.48% | Same order as LilNetX |
| **LilNetX ResNet-20-4 (comparison)** | — | **66 kB** | **91.5%** | **Wins** |

**Conclusion, stated plainly: the supermask idea does not currently beat entropy-coded
trained weights, and at its most-cited operating point (k=50%) it loses by 4x.** The
reason is an information-theoretic identity that the "free weights" framing obscures:

> A supermask over n random weights and a quantized weight vector both transmit exactly
> the same commodity — bits of *learned* information. n·H(p) bits is the entire channel.
> Random weights are free because they carry **zero task information**; freeing them frees
> no capacity.

The supermask therefore wins only if *selection from a random ensemble* is a more efficient
parameterization of good functions than *direct quantized specification*. Pensia's O(log dl)
width overhead is the price of selection. At p=0.5, 1 bit/weight x (2-5x width overhead) ≈
2-5 bits per effective parameter — no better than int4 on a trained net, and worse than the
~1-2 effective bits/param that entropy-penalized training achieves (Part 5).

**Where it is still live.** The MPT row is the tell: pushing to p=0.05 improved the bits
picture by 1.8x over p=0.5. And **nobody has jointly swept (width, p) at a fixed bit budget
with an entropy coder on the mask** — every paper fixes the architecture and sweeps p for
accuracy alone. That sweep is cheap and is the one genuinely open question here. My prior
is that it closes some of the 4x gap but does not overturn LilNetX; I would not stake the
1 kB point on it.

**Two free improvements if we do run it:** (i) freeze BatchNorm at PyTorch defaults as
Ramanujan does, so it costs zero bytes; (ii) use a *structured* mask (whole channels or
kxk kernels) — a channel mask over C channels costs H(p)·C bits instead of H(p)·C·k² and
is far cheaper, at some accuracy cost.

**Composition:** supermasks compose with structured transforms (mask a circulant layer's
generator), with weight-tying, and with free front-ends. They do **not** compose with
quantization (there is nothing left to quantize) — they are an *alternative* to it, which
is exactly why the head-to-head above is the right comparison.

---

## Part 2 — Hypernetworks and implicit representations of the weight tensor

### 2.1 Mechanism

Ship a small network that *outputs* the large network's weights. A coordinate-based
version (SIREN/Fourier-feature MLP mapping (layer, filter, i, j) → weight) is the INR
framing; a learned-embedding version (HyperNetworks) gives each layer a learned latent
vector that a shared MLP expands into a filter bank. Cost = the hypernet's own parameters
plus any per-layer latents.

### 2.2 Verified numbers

| Method | CIFAR-10 | Compression | Notes |
|---|---|---|---|
| **HyperNetworks** (arXiv 1609.09106) | 92.77% (base 94.34) | **15.1x** (0.148M vs 2.236M, WRN 40-2) | Ratio is my arithmetic on their table *[derived]*; they print none |
| **NeRN** (arXiv 2212.13554) | ResNet20 90.39% @0.36 MB (base 91.69 @1.03 MB) | 2.86x | Excludes the permutation index (authors' own 4-6%) and all BN/FC params |
| **SBS** (arXiv 2509.07373), NeRN + random Fourier features | ResNet20 90.50%, ResNet56 92.51% | 5.0x | Best INR-of-weights ratio found |
| **Denil et al.** (arXiv 1306.0543) | "predicting 75% of parameters has negligible effect" | 4x, not 20x | The dictionary U is excluded from the count |

*[all verified]*

### 2.3 The arithmetic, and the verdict

NeRN at 2.86x is **worse than plain int8 post-training quantization** (4x at ~0 loss), and
the paper never compares against quantization. HyperNetworks' 15.1x on 0.148M params is
592 kB at fp32, ~120 kB even with int8+entropy coding — behind LilNetX's 66 kB at higher
accuracy *[derived]*.

**Verdict: does not pencil out.** The empirical ceiling for hypernetwork/INR weight
representation is 3-15x; entropy-penalized training delivers 500-800x on the same
architectures. This is a 50x gap, not a tuning gap.

The mechanism lesson is worth keeping even though the method is not: HyperNetworks gets
15x because its coordinate is a *learned* per-layer embedding; NeRN and SBS get 2-5x
because their coordinate is a *fixed index*, so all capacity must live in a shared MLP
fitting a signal with no genuine smoothness. **The INR framing is what costs the
compression, not what buys it.** HyperNetworks' own Appendix A.1 reports that a
coordinate-based hypernet reached only 93.5% vs 98.5% on MNIST and they abandoned it
*[verified]*.

Also: **"Neural Network Compression via Implicit Neural Representations" does not exist**
under that title, and "Hyper-Compression" (arXiv 2409.00592) has **no CIFAR-10 experiment**
and is a structured VQ, not a hypernetwork — secondary summaries claiming otherwise are
wrong *[verified]*.

A hypernet *is* worth one narrow shot: at our scale the target net has ~5-15k params, and a
500-param hypernet would be a 10-30x win if it trains. That is untested territory (all the
literature targets 0.5M-25M param nets) but the base rates above make it a low-priority bet.

---

## Part 3 — Zero-byte structure: what a short program can generate for free

This is the central asymmetry of the project and deserves an explicit inventory.

### 3.1 The free list

Everything below is generated by decoder code, costing only the (shared, amortized) source
bytes of the generator — typically 20-80 bytes of Python each:

| Free object | Cost | Use |
|---|---|---|
| Any PRNG stream (Gaussian, uniform, signed-constant, Rademacher) | 0-4 B for the seed; **0 B if the seed is fixed to a literal in code** | Random weights, random projections, random codebooks, permutations, dropout masks |
| DCT / DFT / Hadamard / Haar bases | 0 B | Structured transforms, frequency-domain filters |
| Morlet/Gabor wavelet filter banks (scattering) | 0 B | Fixed convolutional front-end |
| Gaussian-derivative basis (structured receptive fields) | 0 B | Fixed filter basis |
| Identity / Dirac initialization | 0 B | Residual-free skip behavior |
| Permutation matrices | 0 B (PRNG) | Fastfood/ACDC-style transforms |
| Circulant / Toeplitz / Hankel structure | Stores O(n), not O(n²) | Any dense layer |
| Analytic quantization grids (uniform, power-of-two/log) | 0 B | vs. a learned codebook, which costs bytes |
| **A random VQ codebook** | **0 B** | See 3.3 — the sleeper idea |
| Any deterministic function of the *unlabeled* train set (Regime B only) | 0 B | ZCA, k-means dictionaries, patch statistics |
| Test-time augmentation, multi-crop, flip-ensembling | 0 B | Free accuracy, runtime is forgiving |
| Weight tying / recurrent application of one block | Reuses stored bytes | Free *depth* |

### 3.2 Structured transforms — verified numbers

| Paper | CIFAR-10 | Stored → Generated | Factor |
|---|---|---|---|
| Cheng et al., circulant (arXiv 1502.03436) | 16.71% err vs 15.60% dense | d floats + d sign bits → full dxd circulant | 4x on CIFAR; AlexNet 233.2→20.5 MB (11.4x) |
| Deep Fried Convnets (arXiv 1412.7149) | **none** | 3 diagonals + permutation → Hadamard never materialized | ImageNet 3.6x |
| ACDC (arXiv 1511.05946) | **none** | 2 diagonals → fixed DCT | CaffeNet 41M FC params → 165,888 |
| Sindhwani Toeplitz-like (arXiv 1510.01722) | **none** (MNIST + speech) | 2 nxr generators → dense via displacement | 3.5x |
| **Harmonic Nets** (arXiv 1812.03205) | **3.84% err @24.4M; 4.25% @12.3M** | learned 1x1 combination coefficients → **fixed DCT spatial basis** | ~3x at +0.34 pts |
| Structured Receptive Fields (arXiv 1605.02971) | RFNiN 86.31% | learned α → fixed Gaussian-derivative basis | Loses on full ImageNet; authors say so |
| **Scattering transform** (arXiv 1412.8659, 1703.08961) | **82.3% with a linear/SVM head and ZERO learned filters**; 93.1% hybrid | fixed Morlet wavelets, J=2 → 243 channels at 8x8 | — |

*[all verified]*. **Flag: Deep Fried, ACDC, Sindhwani, Kim-Tucker and Lebedev-CP have no
CIFAR-10 result at all.** Do not let a survey table interpolate one.

The cleanest evidence that structure does real work rather than merely exploiting
over-parameterization is Cheng's parameter-matched control: at the same 20.7 MB budget,
Reduced-AlexNet gets 65.3% top-1 error while Circulant CNN 2 gets 43.2% — **22 points at
identical bytes** *[verified]*.

**Harmonic Networks is the single best structured-transform Pareto point in this
literature** and is directly applicable: replace every kxk conv kernel with a learned
linear combination over a fixed DCT basis, truncated to the low-frequency subset. At k=3
the DCT basis has 9 elements, so full-rank truncation buys nothing — but truncating to the
4 lowest-frequency bases is a hard 2.25x on every conv layer with a well-motivated
inductive bias, and Harmonic Nets report >20% compression at *no* accuracy loss from
exactly this truncation *[verified, arXiv 2001.06570]*.

### 3.3 The sleeper: a PRNG codebook is free *and* asymptotically optimal

Deep Compression ships a learned k-means codebook plus indices. **The codebook can be
drawn from a PRNG instead, costing zero bytes.** The naive objection is that a random
codebook is worse than a learned one — but Shannon's rate-distortion achievability theorem
is *proved by random coding*: for a memoryless Gaussian source, a codebook of 2^(nR)
i.i.d. Gaussian codewords achieves the rate-distortion bound as block length n grows.
Trained weight tensors are close to Gaussian/Laplacian.

Concrete design *[derived]*: block weights into groups of d=8; generate 2^b random
codewords per block-type from a fixed seed; ship only the b-bit index per block, so the
rate is b/d bits per weight with a **zero-byte codebook**. At b=16, d=8 → 2 bits/weight,
codebook is 65,536 x 8 floats materialized at decode time (runtime is free for us). This
is *strictly better than Deep Compression* at equal rate because the codebook cost vanishes,
and it is one nearest-neighbor search per block to encode.

This is exactly the mechanism underneath **MIRACLE** (Havasi et al., arXiv 1810.00440),
which draws candidate weight vectors from a shared PRNG prior and ships an index — a
bits-back / relative-entropy-coding scheme where the model costs KL(q‖p) bits rather than
the weights' entropy. Verified numbers: LeNet-5 **1.52 kB (1110x) @ 0.96% err**; VGG-16 /
CIFAR-10 **417 kB @ 6.57%** and **168 kB @ 10.0%** (these are the OpenReview-*corrected*
figures; the arXiv PDF's 384/135 kB are superseded — Oktay and LilNetX both cite the
corrected pair) *[verified]*.

**This is the most under-exploited idea on the list**: theoretically principled, verified in
the literature, zero-byte codebook, and it composes with everything.

### 3.4 Free depth via weight tying

Applying one conv block R times costs zero additional bytes and adds real capacity. No
CIFAR-10 byte-frontier paper does this. Cheap to test, obviously safe. *[UNVERIFIED RECALL
that this helps at this scale — but it cannot hurt bytes.]*

---

## Part 4 — Tensor decompositions

### 4.1 Verified numbers

| Paper | CIFAR-10 | Whole-net compression |
|---|---|---|
| Novikov TT (arXiv 1509.06569) | 76.87% | **1.24x** — the famous 200,000x is a *single FC layer of VGG on ImageNet* |
| **Wang, Tensor Ring (arXiv 1802.09052, Table 6)** | **92.7% @243x**, **90.3% @521x**, **83.7% @1217x** (WRN-28) | up to 1217x |
| Garipov (arXiv 1611.03214) | 89.4% | 82.9x |
| Hawkins & Zhang (arXiv 1905.10478) | ResNet-110 90.4% | only 7.4x (the abstract's 137x is MNIST FC-only) |
| Kim Tucker (1511.06530), Lebedev CP (1412.6553) | **no CIFAR-10** | — |

*[all verified]*

### 4.2 Arithmetic

Tensor Ring is the only verified >100x whole-network CIFAR-10 result with credible
accuracy, and the extreme rows are the interesting ones *[derived]*:

- r=2: 36.2M / 1217 = **29,745 params @ 83.7%** → 30 kB at 8-bit, **~15 kB at 4-bit**,
  plausibly **10-12 kB entropy-coded**.
- r=6: 36.2M / 521 = **69,482 params @ 90.3%** → ~34 kB at 4-bit.

**Caveat that must be stated:** 1217x is partly a statement about the 36.2M-param WRN-28-10
baseline. Every *conv-dominated, already parameter-efficient* backbone in this literature
tops out at 5-20x. The right test is whether TRN factorization still buys anything on a net
that is already at 15k params — my prior is that it buys little, because the redundancy it
exploits has already been removed.

Still: TRN r=2 at ~29.7k params / 83.7% is a legitimate 10-15 kB candidate and it is
cheap to reproduce. It does not reach 1 kB.

**Composition:** TRN cores are ordinary tensors, so they quantize and entropy-code
normally. TRN + int4 + arithmetic coding is an unexplored and arithmetically promising
stack.

---

## Part 5 — Coding-theoretic training: make the model compressible, don't just compress it

### 5.1 The head-to-head evidence (this is the strongest verified result in the document)

**DeepCABAC's Table I (arXiv 1907.11900)** applies the *identical* context-adaptive
arithmetic coder to a normally-trained net and to the same architecture after
compressibility-oriented training *[verified]*:

| Model | Post-hoc on dense | Same coder, trained-compressible | Gain |
|---|---|---|---|
| Small-VGG16 / CIFAR-10 | 17.0x | **52.6x** | **3.1x** |
| LeNet5 / MNIST | 39.7x | 115x | 2.9x |
| VGG16 / ImageNet | 25.3x | 63.3x | 2.5x |
| ResNet50 / ImageNet | 9.9x | 19.5x | 2.0x |

Accuracy cost is near zero (91.54 → 91.35 on Small-VGG16).

**LilNetX Table 1 (arXiv 2204.02965)** is the byte-level head-to-head on VGG-16/CIFAR-10
(60 MB uncompressed @ 6.6% error) *[verified]*:

| Method | Size | Ratio | Error |
|---|---|---|---|
| DeepCABAC (post-hoc coder) | 960 kB | 62x | 9.0% |
| BC-GHS (Bayesian, Louizos) | 525 kB | 116x | 9.2% |
| MIRACLE (bits-back) | 168 kB | 452x | 10.0% |
| Oktay EPR (entropy-penalized) | 101 kB | 590x | 10.0% |
| **LilNetX** | **129 kB** | 465x | **7.4%** |
| **LilNetX (extreme)** | **76 kB** | **800x** | 10.0% |

**Answer to the question posed: yes, decisively.** At matched accuracy, entropy-penalized
training gets **7-13x better compression than the best post-hoc arithmetic coder**
(590-800x vs 62x). Holding the coder fixed and changing only the training, the gain is a
cleaner **2-3x**. A rate-aware *coder* buys tens of percent (Choi et al. arXiv 1612.01543:
+31% on LeNet over Deep Compression); a rate-aware *objective* buys multiples.

Supporting verified anchors: Ullrich Soft Weight-Sharing (1702.04008) WRN-16-4/CIFAR-10 45x
at a 2-point accuracy cost; Louizos Bayesian Compression (1705.08665) VGG/CIFAR-10 95-116x;
Molchanov variational dropout (1701.05369) VGG-like CIFAR-10 at 48x fewer weights matching
its own baseline — but note Molchanov reports *sparsity*, Ullrich/Louizos report *CR under
Han's CSR+Huffman format*, and MIRACLE/Oktay/LilNetX report *actual bytes*. **These are not
commensurable**; putting 280x next to 1110x on one axis is a category error.

**Negative finding worth knowing:** Oktay et al. contains **no** with/without-entropy-penalty
ablation, and Wiedemann's ECT never benchmarks post-hoc coding of the same net. A
controlled "entropy penalty on vs off, same architecture, same coder, at the ~10 kB scale"
ablation **does not exist in the literature**. We would have to run it — and it would be
publishable on its own.

### 5.2 Exploiting structure *within* conv filters

| Method | Mechanism | Result |
|---|---|---|
| **CNNpack** (NeurIPS 2016) | DCT each filter, cluster centers + residuals, threshold, Huffman | AlexNet **39x, 41.6/19.2 top-1/5** (better than the 41.8 original); VGG-16 46x; LeNet 32x |
| **FreshNets** (arXiv 1506.04449) | DCT then hash frequency coefficients into shared buckets, frequency-sensitive allocation | CIFAR-10: 14.91% err uncompressed → **21.42% @ 1/16 size**, **30.79% @ 1/64**. Beats "just zero the high frequencies" by 9 points at 1/16 |
| Frequency-Domain Dynamic Pruning (NeurIPS 2018) | 2-D DCT per kernel, band-adaptive dynamic prune-and-splice | LeNet-5 **150x (2.8k params) @ 99.08%**; ResNet-110/CIFAR-10 8.4x above reference accuracy |
| ILWP (arXiv 1907.06835) | Predict kernels from the *adjacent layer's* kernels, code the residual | MobileNet/CIFAR-10 70.2 kB → 33.4 kB |

*[all verified]*

**A gap my verification thread identified and I think is real:** nobody entropy-codes a conv
filter with a **2-D spatial context model** — predicting coefficient (i,j) from already-coded
(i−1,j) and (i,j−1) *inside the same kxk kernel*, JPEG-DC-prediction style. CNNpack/
FreshNets/FDNP transform then drop-or-share; DeepCABAC context-codes in raster order with
adaptive probabilities but no spatial predictor; ILWP predicts across layers. The obvious
objection — k=3 makes a 2-D context template nearly degenerate — is probably why, and we
should preempt it rather than pitch this as a headline idea. It is a Hutter-crew-flavored
refinement worth maybe 5-15%, not a new frontier.

---

## Part 6 — The dataset-as-prior angle, and dataset distillation

### 6.1 The arithmetic correction that changes the conclusion

**The common claim that "10 images/class at 32x32x3 quantized is a few kB" is wrong by two
orders of magnitude** *[derived]*:

| Budget | Values | 8-bit | 4-bit | 16x16, 8-bit |
|---|---|---|---|---|
| IPC = 1 (10 images) | 30,720 | **30.7 kB** | 15.4 kB | 7.7 kB |
| IPC = 10 (100 images) | 307,200 | **307 kB** | 154 kB | 76.8 kB |
| IPC = 50 | 1,536,000 | 1.54 MB | 768 kB | 384 kB |

A CIFAR-10 image is 3,072 values. IPC=10 is a **307 kB** artifact, not "a few kB". The
50-60% figure everyone quotes for IPC=10 is therefore a 300 kB / ~52% point, which is
catastrophically off the frontier.

### 6.2 Verified accuracies (ConvNetD3 throughout; full-dataset reference 84.8%)

| Method | IPC 1 | IPC 10 | IPC 50 |
|---|---|---|---|
| DC | 28.3 | 44.9 | 53.9 |
| DSA | 28.8 | 52.1 | 60.6 |
| DM | 26.0 | 48.9 | 63.0 |
| MTT (arXiv 2203.11932) | 46.3 | 65.3 | 71.5 |
| FRePo (arXiv 2206.00719) | 46.8 | 65.5 | 71.7 |
| DATM (arXiv 2310.05773) | 46.9 | 66.8 | 76.1 |
| **NCFM (CVPR'25, arXiv 2502.20653)** | **49.5** | **71.8** | **77.4** |
| Random real subset (DC paper protocol) | 14.4 | 26.0 | 43.4 |
| Herding (DC paper protocol) | 21.5 | 31.6 | 40.4 |
| K-Center | 21.5 | 14.7 | 27.0 |
| Forgetting | 13.5 | 23.3 | 23.3 |
| Random real subset (**IDC** protocol) | — | **37.2** | **56.5** |

*MTT, DATM, NCFM, FRePo verified from primary PDFs. DC and all four coreset baselines
verified from the DC paper's own Table 1 (arXiv 2006.05929), which also confirms the
84.8±0.1 whole-dataset reference. DSA/DM cross-verified from three agreeing secondary
tables, not primary-fetched — high confidence, flagged.*

**Two cautions from that table.** Herding beats random at IPC 1 and 10 but *loses* at IPC 50
(40.4 vs 43.4), and K-Center is non-monotone (21.5 → 14.7 → 27.0) — the classical coreset
baselines are weak, not a real frontier. More importantly, **the training recipe moves the
random-real baseline by ~11 points** (26.0 under DC's protocol vs 37.2 under IDC's at the
same IPC=10), which is larger than many published method-vs-method gaps. Any frontier row we
publish must state its protocol.

**Storage-matched factorized methods** (these already are bytes-axis numbers, at fp32):

| Method | IPC-1 budget | IPC-10 budget | IPC-50 budget |
|---|---|---|---|
| IDC (arXiv 2205.14959) | — | 67.5 | 74.5 |
| HaBa (arXiv 2210.16774) | 48.3 | 69.9 | 74.0 |
| **RTP/LinBa (arXiv 2206.02916)** | **66.4** | 71.2 | 73.6 |

RTP's IPC-1 result is the standout: **+20.1 points over MTT at identical storage**, achieved
by learning K bases at 16x16x3 plus addressing matrices. *[verified]*

**Quantization of distilled sets exists and the field is already crowded.** QuADD (CVPR
2026, arXiv 2603.02411) does differentiable quantization inside the distillation loop:
CIFAR-10 at 9 bits/sub-pixel gives 44.9 / 65.4 / 74.8 vs DATM's unquantized 46.1 / 65.7 /
75.1 — a **10.6x storage reduction at ~0.5 points**. AutoPalette (2411.11329) and a
post-training-quantization variant (2603.13346, down to 2-bit) also exist *[verified,
HTML-level]*.

**Someone has already done the bytes axis.** Bao et al., *Dataset Distillation as Data
Compression: A Rate-Utility Perspective*, **ICCV 2025, arXiv 2507.17221** *[verified]*.
Defines `bpc(S) = #bits(S)/K` — total bits to losslessly encode samples + labels + decoder
parameters — and makes exactly our critique of IPC. Reports **CIFAR-10 77.5% at 94 kB**,
up to 170x over vanilla distillation. Its Figure 3(a) is *the* accuracy-vs-bits curve for
this literature, covering TM/IDC/HaBa/RTP/FreD/FRePo/NSD/SPEED/HMN/DDiF — but it is a
figure, not a table, so we would have to digitize it.

Also relevant: **PoDD**, "Distilling Datasets Into Less Than One Image" (arXiv 2403.12040):
CIFAR-10 at 0.3 / 0.5 / 1.0 IPC → 42.3 / 49.5 / 59.1 *[HTML-level, not line-verified]*. At
0.3 IPC = 9,216 values = **9.2 kB at 8-bit for 42.3%** — the best sub-10 kB DD point I
found.

### 6.3 Verdict: dataset distillation is dominated, and it is not close

Putting the best DD points on our axis against the existing frontier *[derived]*:

| Artifact | Bytes | Accuracy |
|---|---|---|
| PoDD 0.3 IPC, 8-bit | 9.2 kB | 42.3% |
| RTP IPC-1 budget, 8-bit | 30.7 kB | ~66% |
| Bao et al. best-by-bits | 94 kB | 77.5% |
| **µNAS** | **11.4 kB** | **86.5%** |

**µNAS at 11.4 kB / 86.5% dominates every dataset-distillation point on both axes** — 8x
fewer bytes and 9 points more accuracy than Bao's rate-optimized result. The reason is
structural, not a tuning gap: distilled images are a *dense, unstructured, non-shared*
parameterization at 3,072 values per image, whereas a conv net's parameters are reused
across every spatial position and every input. DD spends its bytes on the wrong thing.

**A second, independent problem.** DD-Ranking (arXiv 2505.13300) re-evaluates under a fair
protocol where synthetic data is compared to an *equal-size random real subset under
identical label type and augmentation* *[verified]*. On CIFAR-10, **RDED scores +2.4 / +1.1
/ −1.6 and SRe2L scores −0.3 / −5.7 / −6.5** — the decoupled/recover-relabel family
provides zero or negative value over random real images. Their verbatim conclusion: the
gains are "predominantly attributable to knowledge distillation from soft labels, rather
than any inherent improvement in the informativeness of the distilled data." The
trajectory-matching family (DATM +30.8/+35.1/+23.9, MTT +27.6/+30.9/+20.5) survives this
test; the newer efficient methods do not.

**Third: the protocols do not compare.** Three papers report three different full-dataset
"100%" lines for the same ConvNet-3 (82.24 / 84.8 / 88.1). MTT/HaBa/RTP use ZCA whitening;
DATM/NCFM do not. DATM and FRePo learn soft labels; FRePo evaluates on a *wider Conv-BN*
net because "the KRR component does not behave well when the feature dimension is low."
Any frontier plot mixing these has a meaningless y-axis.

### 6.4 Is shipping distilled data legitimate MDL? My view

**Yes, with three conditions.** An artifact is a program; a program that contains 10
synthetic images, an architecture spec, a hyperparameter block, and a fixed RNG seed, and
that trains for ten minutes at decode time before classifying, is a perfectly ordinary
program. Runtime is forgiving by project rule. There is nothing about pixels-as-parameters
that is less legitimate than floats-as-parameters.

The conditions:

1. **Everything is counted**: distilled pixels *at their actual bit depth*, the learned
   student learning rate (MTT learns one), learned soft labels (DATM, FRePo), the
   architecture spec, the augmentation policy, and the seed. Bao et al.'s `bpc` is the
   right accounting and we should adopt its definition.
2. **No reference to the real training set at decode time.** "Distill, then fine-tune on
   CIFAR-10" imports the labels for free and is Regime B-labeled — degenerate.
3. **Determinism.** A fixed seed, or report the mean and the variance honestly. A method
   whose accuracy depends on a lucky decode-time training run is not a 94-kB artifact
   achieving 77.5%; it is a distribution.

Under those conditions DD is legitimate — and, as shown above, uncompetitive. I would not
spend project time on it except as a baseline row on the frontier plot, which is worth
having precisely because it is a natural idea that people will ask about.

**One unclaimed seam, if we ever want a paper out of this.** The rate-distortion framing of
DD is taken (Bao et al.) and bit-depth quantization of distilled sets is taken (QuADD,
AutoPalette). But **nothing frames dataset distillation in terms of MDL, a two-part code,
or a prequential codelength** — Bao et al. are rate-*utility*, with no model-description
term. Computing the prequential description length of the labels that a distilled set
substitutes for, and comparing it to the distilled set's own bit budget against the
Blier–Ollivier 45.3 kbit anchor, is genuinely open and is a natural side-product of the
accounting this project has to do anyway.

### 6.5 The genuinely interesting version: Regime B-unlabeled

The unlabeled-training-images-are-free regime is where the "dataset as prior" idea actually
pays. Verified anchors *[verified]*:

| Front-end | CIFAR-10 | Front-end storage | Head |
|---|---|---|---|
| 1-NN raw pixels, L1 / L2 | **38.6% / 35.4%** | 0 (Regime B only) | 0 |
| Linear SVM, raw pixels | 42.3% | 0 | 30,720 params |
| Random Fourier features, n=16,384 | 62.4% | **seed only** | 163,840 params |
| Random Gaussian conv dict (1600) + soft threshold | 73.2% | **seed only** | ~131k params |
| **Random *patch* dict (1600)** | **79.1%** | 1600 indices ≈ 6.4 kB | ~131k params |
| K-means 1600 (Coates & Ng) | 77.9% | 691 kB learned | — |
| K-means 4000 (Coates & Ng) | **79.6%** | 1.73 MB learned | 163,840 params |
| **Scattering (fixed Morlet) + SVM** | **82.3%** | **0 bytes, no learned filters** | large |

Two things jump out. First, **random patches at 79.1% essentially match learned k-means at
79.6%** while storing 1600 integer indices instead of 1.73 MB of floats — and under Regime
B those indices are free too. Second, and decisively: **the front-end is not the
bottleneck, the head is.** Every competitive configuration above carries a 0.5-2 MB linear
classifier. Zeroing the front-end moves maybe 20% of the bytes.

Note also the correction: Coates' "4000 features" means K=4000 *centroids*; the pooled
vector is 16,000-dim, so the head is 160,000 params ≈ 640 kB, not 40k *[verified]*.

**That observation is the actual design principle for our 1 kB point**: take a zero-byte
front-end and then attack the head — pool aggressively, project to low dimension with a
free random or DCT projection, and quantize what remains. That is Bet #1 in Part 7.

One caution: Arora et al. (arXiv 1904.11955) found random-feature approximation of the CNTK
badly underperforms the exact kernel (90.65% vs 95.70% on CIFAR-2 at depth 21)
*[verified]*. Cheap randomization is not free accuracy.

---

## Part 7 — Ranked shortlist

Ordering is by expected bytes-per-accuracy given the arithmetic above, with implementation
cost as a tiebreak. All five compose; the top two compose especially well.

### For a strong 1 kB point

**#1 — Zero-byte front-end + aggressively pooled features + entropy-coded micro-head.**

Take a fixed, code-generated front-end (scattering with J=2 Morlet wavelets, or 2-3 layers
of fixed low-frequency DCT filters — both 0 bytes), spatially average-pool hard to D ≈
150-250 dimensions, and train a single quantized linear head of D x 10 weights with an
entropy penalty (LilNetX-style) so it codes at ~2-3 bits/weight. Arithmetic: 200 x 10 =
2,000 params at 3 bits = **750 B**, leaving ~270 B for the decoder if the metric charges
for it. Expected accuracy is the big unknown — the verified anchors bracket it between
linear-on-raw-pixels (42.3%) and scattering+full-head (82.3%), and the pooling to 200 dims
is where most of that gap gets spent; I would predict **50-60%** and would want that
measured before anything else. **Why it wins:** every byte purchased is label-dependent
information, and the forward pass is short enough that the decoder source stays under
budget. **What would make it fail:** (a) hard pooling destroys the spatial information that
gives scattering its 82.3%, collapsing accuracy toward the raw-pixel linear baseline; (b)
the decoder source for a scattering transform is bigger than I think — a Morlet filter bank
generator plus a two-layer modulus cascade may be 600+ gzipped bytes, which at a 1 kB total
budget is fatal. Mitigation for (b): use a DCT/Haar front-end instead, whose generator is
three lines.

**#2 — PRNG-codebook vector quantization over a micro-CNN.**

Train a ~3-6k-parameter conv net, then encode it with a zero-byte random codebook: block
weights into groups of d=8, generate 2^b i.i.d. Gaussian codewords from a fixed seed, ship
only the b-bit index per block. Rate = b/d bits/weight with **no codebook bytes**. At b=16,
d=8 → 2 bits/weight → a 4,000-param net costs **1,000 B**. Justification is Shannon's
random-coding proof of the rate-distortion theorem plus MIRACLE's verified 1.52 kB LeNet-5
at 0.96% error. **Why it wins:** it strictly dominates Deep-Compression-style clustering at
equal rate, because the codebook cost goes to zero; and it is orthogonal to architecture, so
it stacks on #1's head or #4's factorization. **What would make it fail:** a random codebook
at small block length d=8 is far from the asymptotic regime, and the nearest-neighbor
quantization error may cost more accuracy than the bytes it saves versus a learned 16-entry
codebook (which is only ~64 B anyway at this scale — check that the codebook you are
eliminating was actually expensive before celebrating).

**#3 — Structured (circulant / Hadamard-diagonal) layers so a dense layer costs O(n).**

Replace the classifier head and any 1x1 mixing with circulant matrices (store n floats, the
full nxn matrix is generated by cyclic shift) or ACDC/Fastfood blocks (store 2-3 diagonals,
the Hadamard/DCT transform is free). Cheng's parameter-matched control — 22 points of
ImageNet top-1 at identical bytes — is the strongest evidence in this whole survey that
structure beats naive shrinking. **What would make it fail:** every verified structured-
transform result operates on layers with 10⁴-10⁷ parameters, where the O(n²)→O(n) saving is
enormous. At n=200 the saving is 200x in principle but the layer only had 40,000 params to
begin with, and circulant structure is a *severe* constraint on a small layer that has no
redundancy to give up. This is the bet I am least confident transfers to our scale.

**#4 — Weight-tied recurrent micro-CNN.**

Apply a single small conv block R=4-8 times with shared weights, plus per-iteration
BatchNorm folded away. Depth is free; only the block's parameters are shipped. Zero
literature risk (this is just ALBERT/DEQ-style tying), zero byte cost, and it composes with
every other bet. **What would make it fail:** shared-weight iteration at 1 kB scale may
simply not train, and the accuracy gain over an untied net of equal parameter count may be
small enough to be noise.

**#5 — Supermask over a seeded random net, with a jointly swept (width, keep-fraction) at
fixed bit budget.**

I rank this fifth, not first, and Part 1.3 is the argument: at the published k≈50%
operating point the supermask costs 293 kB for ~86%, versus LilNetX's 66 kB for 91.5%. The
"weights are free" framing conceals that n·H(p) bits is the entire learned-information
channel, and Pensia's O(log dl) width overhead is the price of buying capacity by selection
rather than specification. The one thing nobody has done — sweeping width and p *jointly*
under a fixed entropy-coded bit budget, rather than fixing the architecture and sweeping p
for accuracy — is cheap and worth one experiment. Use Signed Kaiming Constant weights
(worth 5-6 ImageNet points over Kaiming Normal), freeze BatchNorm at defaults so it costs
zero bytes, and consider a channel-structured mask at H(p)·C bits instead of H(p)·C·k².
**What would make it fail:** the most likely outcome, honestly, is that it closes some of
the 4x gap and still loses to entropy-penalized training — in which case the finding is
still worth writing down, since the supermask-as-compression claim is widely repeated and
has never been priced.

### For a strong 10 kB point

**#1 — Entropy-penalized training of a µNAS-scale architecture (the unclaimed seam).**

This is the highest-expected-value experiment in the document. µNAS reaches 11.4 kB at
86.5% with 8-bit weights and **no entropy coding at all**; LilNetX/EPR-style training
reaches 2-3 effective bits/param on much larger nets. Composing them puts 11,400 params at
~2.5 bits into **~3.5 kB at ~86%**, a 3x improvement on the published frontier. The
DeepCABAC control (3.1x on Small-VGG16/CIFAR-10 from training alone, same coder) and the
LilNetX table (590-800x vs 62x post-hoc at matched accuracy) both say the gain is real and
large. **What would make it fail:** all the verified evidence for entropy-penalized training
comes from heavily over-parameterized nets, where the penalty is removing redundancy that a
µNAS-searched architecture has already removed. If compressibility and parameter-efficiency
are the same resource, the gains do not stack and we get 11.4 kB → 8 kB, not 3.5 kB. This
is *the* risk and it is also *the* reason the experiment is publishable either way — the
controlled "entropy penalty on vs off, same tiny architecture, same coder" ablation does
not exist in the literature.

**#2 — Tensor-ring factorization of a small backbone + int4 + arithmetic coding.**

Wang's TRN reaches 83.7% at 29,745 params (r=2) and 90.3% at 69,482 (r=6) on WRN-28. At
4 bits those are ~15 kB and ~34 kB, and entropy coding should take the r=2 point to
**10-12 kB at ~84%**. TRN cores are ordinary tensors, so the quantize-and-code stack applies
unchanged, and nobody has run it. **What would make it fail:** the 1217x is partly a
statement about a 36.2M-param baseline; on a backbone that is already parameter-efficient,
every conv-dominated result in this literature collapses to 5-20x. Test by applying TRN to a
~200k-param net first and checking whether the compression factor survives.

**#3 — Fixed-DCT-basis convolutions with learned coefficients (Harmonic Networks),
truncated.**

Every kxk kernel becomes a learned linear combination over a fixed, code-generated DCT
basis, truncated to the low-frequency subset. Harmonic Nets verify 4.25% error at 12.3M
params (~3x) and >20% compression at *no* accuracy loss from truncation alone. At k=3 the
basis has 9 elements, so truncating to 4 is a hard 2.25x per conv layer with a defensible
inductive bias. **What would make it fail:** at k=3 the DCT basis is small enough that
truncation is nearly equivalent to just using smaller kernels, in which case this is a
reparameterization with no byte win. Verify on k=5 kernels where the basis has 25 elements
and truncation is meaningful.

**#4 — Bits-back / relative-entropy coding of the weight posterior (MIRACLE-style).**

Train a variational posterior over weights and transmit a *sample* from it using a shared
PRNG prior, paying KL(q‖p) bits rather than the weights' entropy. This is the theoretically
correct MDL answer — it is Hinton & van Camp's 1993 objective with a coder that actually
achieves it — and MIRACLE verifies 1.52 kB LeNet-5 at 0.96% error and 168 kB VGG-16 at 10%
CIFAR-10 error. **What would make it fail:** MIRACLE's encoder is expensive and its block
decomposition is fiddly; and at 10 kB the KL budget per parameter is ~7 bits for an 11k-param
net, which is a regime where the bits-back advantage over straightforward
quantize-plus-entropy-code may be small. Its real strength is at extreme ratios on large
nets, which is not our operating point.

**#5 — Frequency-domain hashing of conv coefficients (FreshNets) as a composition layer.**

DCT each filter, then hash frequency coefficients into shared buckets with a
frequency-sensitive allocation (more buckets for low frequencies). Verified: CIFAR-10
21.42% error at 1/16 size and 30.79% at 1/64, and — the useful part — it beats the naive
"zero the high frequencies" baseline by 9 points at 1/16, so the mechanism is doing real
work rather than just low-pass filtering. **What would make it fail:** FreshNets' baseline
is a 1.2M-param 5-layer conv net; 1/64 of that is ~19k params, which is roughly where we
already are. Compressing *our* 11k-param net by another 16x is a completely different ask
and there is no evidence it survives.

### Things I looked at and am recommending *against*

- **Hypernetworks and INRs of the weight tensor.** 3-15x verified ceiling against 500-800x
  for entropy-penalized training. Not a tuning gap. (Part 2.)
- **Dataset distillation as a frontier method.** Dominated on both axes by µNAS by 8x in
  bytes and 9 points in accuracy, for a structural reason. Keep it as one baseline row.
  (Part 6.3.)
- **IteRand.** The paper itself says storing the model needs R seeds and R masks. (Part 1.2.)
- **Slot Machines.** 3 bits/connection for ~0.4 points over a 1-bit supermask. (Part 1.2.)
- **2-D spatial context coding inside kxk filters.** Genuinely unclaimed, but k=3 makes the
  context template nearly degenerate; worth 5-15%, not a frontier move. (Part 5.2.)

---

## Appendix — citations used, with verification status

Verified from a fetched primary source this session: 1905.01067, 1911.13299, 2103.09377,
1906.04358, 2002.00585, 2006.07990, 2106.09269, 2101.06475, 2111.12330, 1502.03436,
1412.7149, 1511.05946, 1510.01722, 1812.03205, 2001.06570, 1605.02971, 1412.8659,
1703.08961, 1609.09106, 2212.13554, 2509.07373, 1306.0543, 1509.06569, 1802.09052,
1611.03214, 1905.10478, 1511.06530, 1412.6553, 1702.04008, 1705.08665, 1701.05369,
1810.00440 (+ OpenReview correction), 1906.06624, 1905.08318, 1907.11900, 2204.02965,
1612.01543, 1806.08342, 2106.08295, CNNpack (NeurIPS 2016), 1506.04449, 1907.06835,
1905.12107, 2010.14246, 2203.11932, 2206.00719, 2310.05773, 2502.20653, 2205.14959,
2210.16774, 2206.02916, 2505.13300, 2507.17221, 1802.07044, 2103.03872, 2309.10668,
Coates & Ng (PMLR v15), Saxe et al. (ICML 2011), Le et al. (PMLR v28), 1904.11955.

Read at HTML/abstract level only, **not line-verified**: 2603.02411 (QuADD), 2603.13346,
2411.11329, 2403.12040 (PoDD), 2603.03808, 2604.18135, Ko et al. DATE 2017.

Verified in a second pass: DC's Table 1 (arXiv 2006.05929) for DC itself and all four
coreset baselines plus the 84.8±0.1 whole-dataset reference; **1-NN on raw CIFAR-10 pixels,
38.6% (L1) and 35.4% (L2)** — but the source is the Stanford CS231n course notes
(cs231n.github.io/classification/), not a peer-reviewed paper, and it is k=1 only; CS231n
states **no test accuracy for cross-validated k**.

**Nearest-class-mean on raw CIFAR-10 pixels: searched and genuinely absent.** No citable
number exists. The ~27-28% figure in circulation is folklore with no source — do not use it.
It is a one-line computation if we want the anchor.

Unverified and flagged in-text: DSA/DM primary tables (cross-verified from three
agreeing secondary tables); µNAS's LEMONADE comparison row, which appears to
be off by ~10x on the size axis when checked against LEMONADE's own Figure 3; TinBiNN;
NCFM's soft-label status; RDED's final CVPR camera-ready numbers.
