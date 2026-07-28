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

## Part 1.5 — The published byte-vs-accuracy frontier (what we are actually racing)

Before the method survey, the anchor points. All are 10-class CIFAR-10 unless
marked. Byte figures marked *[C-est]* are int8 conversions of published *parameter*
counts — the papers report params, not bytes, so these are assumptions to validate.

| Artifact size | Model | Accuracy | Source |
|---|---|---|---|
| ~0.5-0.9 KB | µNAS / SpArSe sub-KB results | 73-77% — but **2-class binary CIFAR** | arXiv:2010.14246 T2/T4; arXiv:1905.12107 T2 *[verified]* |
| **~1 KB** | **nothing published** | — | **empty cell** |
| **11.4 KB** | **µNAS, 11.4K params, int8** | **86.49%** | arXiv:2010.14246 Table 2 *[verified]* |
| ~47 KB *[C-est]* | LEMONADE, 47K params | **91.1%** | arXiv:1804.09081 Table 2 *[verified]* |
| ~53 KB *[C-est]* | YYNet-Small-16ch, 52,882 params | 89.46% | arXiv:2310.16148 T4 *[verified]* |
| ~58 KB | hls4ml MLPerf-IC, 8-12 bit | 83.5% | arXiv:2206.11791 T1 *[verified]* |
| 96 KB | MLPerf Tiny ResNet (TFLite int8) | 86.5% | arXiv:2106.07597 *[verified]* |
| ~101 KB | Entropy Penalized Reparam., VGG-16 | 90.0% (10.0% err) | arXiv:1906.06624 *[verified]* |
| ~135 KB | MIRACLE, VGG-16 | 90.0% | arXiv:1810.00440 *[verified]* |
| ~190 KB *[C-est]* | LEMONADE, 190K params | 94.5% | arXiv:1804.09081 *[verified]* |
| ~2 MB | airbench94 | 94.01% | arXiv:2404.00498v2 *[verified]* |

**Three readings of this table:**

1. **µNAS at 11.4 KB / 86.49% is the number to beat at the 10KB point**, and it is
   strong. Note it is 8-bit dense with *structured* (channel) pruning and no entropy
   coding — every byte-side technique in Part 2 is unexploited by it.
2. **MLPerf Tiny's 96 KB / 86.5% is far off the frontier** — µNAS matches it in
   1/8 the bytes. Do not use it as a target.
3. **The ~1 KB 10-class cell is genuinely empty.** Every sub-kilobyte CIFAR result
   in the literature (SpArSe, µNAS Tables 2 and 4, Bonsai) is on a **2-class**
   relabeling where chance is 50%. This is the project's clearest opening — and it
   also means there is no baseline to beat there, only one to establish.

A caution that applies to the whole table: µNAS Appendix B states its sparse-model
sizes do **not** account for mask storage *[verified]*, so its sparse rows are
optimistic in real bytes. Its dense 11.4 KB row is not affected.

---

## Part 2 — Method survey, ranked by bytes

### (a) Tiny architectures / micro-CNNs / depthwise-separable

**Mechanism.** Reduce parameter count structurally: depthwise-separable convolutions
factor a KxKxCxC' conv into KxKxC + CxC'; channel-pruning and NAS search the width
and depth directly. Convolution's weight sharing is the byte-efficiency mechanism
that matters — it decouples representational width from parameter count, which is
why every point on the verified frontier above is a convnet.

**Datapoints.** LEMONADE Table 2, all trained identically at native 32x32 from
scratch *[verified]*: 38K→88.0%, 40K→88.5%, 47K→**91.1%**, 68K→88.5%,
190K→94.5%, 3.4M→96.4%. µNAS *[verified]*: **86.49% @ 11.4K params**, 384K MACs,
via structured DPF channel pruning + TFLite post-training int8.

**Linear/kNN floors** *[verified, Krizhevsky 2009 TR Fig. 3.1 and arXiv:1611.04905]*:
multinomial logistic regression on **unwhitened** raw pixels **41.13%**; on
**whitened** raw pixels **37.32%** — whitening *hurts* a raw-pixel linear
classifier, which is counterintuitive and directly relevant. 1-NN L1 38.6%, kNN
33.86%, SVM 49.88%, 1-hidden-layer (1000 units) 49.78%. **kNN + PCA-30 = 41.78%**,
beating raw-pixel kNN by ~8 pp on 30 dimensions.

**The linear baseline is not on the frontier.** A 3072x10 logistic regression is
30,730 params = **30.7 KB at int8 for 41%** *[derived]*. µNAS gets 86.5% in 11.4 KB.
Even at 4 bits it loses by 45 accuracy points at 1.5x the bytes. Keep it only as a
sanity floor.

**Scaling law, and the fact that governs the whole project.** arXiv:2603.07365
*[verified]* fits CIFAR-**100** error ~ N^(-0.156) for plain ConvNets and
N^(-0.106) for MobileNetV2 over 22K-19.8M params. **At exponent 0.156, halving the
error costs 2^(1/0.156) ~= 88x the parameters** *[derived]*. Accuracy bought with
width is ruinously expensive in bytes. Wins must come from *changing the
description*, not scaling it — which is the MDL thesis of the project restated.
Note this fit does not extend below 22K params; **extrapolating it to the 1-3K
regime is unjustified** and any 1KB accuracy forecast built on it should be treated
as a guess.

**Traps.** MicroNets (arXiv:2010.11267) reports **no CIFAR-10 at all**; MCUNet's
abstract has none either *[verified]*. MobileNetV2 / ShuffleNet / SqueezeNet
"CIFAR-10 93-98%" figures in circulation are **ImageNet-pretrained and fine-tuned at
224x224 upsampled** *[verified, arXiv:2505.03303]* — not comparable to from-scratch
32x32 training and must not enter our frontier table.

**Difficulty.** Medium-high (NAS is expensive: µNAS spent 23 GPU-days). But a hand-
designed narrow convnet with a good stem gets most of the way for a fraction of the
compute.

**Composes with.** Everything. This is the substrate; all other families are
transforms applied to it.

---

### (b) Extreme quantization: int8 / int4 / ternary / binary

**Mechanism.** Restrict weights to a small level set. Binary {-1,+1} (BinaryConnect,
BNN) keeps an fp shadow copy during training and discards it. Ternary (TWN) adds a
zero level with a learned per-layer threshold and one scale; TTQ learns two
independent scales {-W_n, 0, +W_p}. Mixed-precision (DNAS, Bayesian Bits) allocates
bitwidth per layer.

**CIFAR-10 datapoints** *[all verified]*:

| Method | Setting | CIFAR-10 | vs FP baseline |
|---|---|---|---|
| BinaryConnect (arXiv:1511.00363) | VGG-like, deterministic | 9.90% err | 10.64% (BC *helps*) |
| BinaryConnect, stochastic | same | **8.27% err** | 10.64% |
| BNN (arXiv:1602.02830) | weights+activations binary | 10.15% err | — (no matched FP ablation) |
| **TWN (arXiv:1605.04711)** | **VGG-7 ternary** | **92.56%** | **92.88% FP; binary 90.18%** |
| TTQ (arXiv:1612.01064) | ResNet-20 | 8.87% err | 8.23% FP (**-0.64pp**) |
| TTQ | ResNet-32 / 44 / 56 | 7.63 / 7.02 / 6.44% err | 7.67 / 7.18 / 6.80% (TTQ *wins*) |
| DNAS (arXiv:1812.00090) | ResNet-20 mixed-prec | 92.00% @ 16.6x compression | 92.35% FP |
| DNAS | vs uniform 2-bit TTQ | **+1.59pp at similar compression** | — |
| Tiled Bit Networks (arXiv:2407.12075) | ResNet-18, 0.256 b/w | **93.1%** | 93.1% FP32 |
| TBN | ResNet-18, 0.069 b/w | 91.2% | 93.1% FP32 |

**The two conclusions that matter:**

1. **Ternary is the sweet spot, not binary.** On matched VGG-7, ternary costs 0.32pp
   where binary costs 2.70pp — the zero level does almost all the work. And ternary's
   zero level is what an entropy coder monetizes (see (c)), whereas a balanced binary
   net is already at maximum entropy and incompressible.
2. **The "quantization is free" literature does not transfer to our regime.** TTQ
   *beats* FP on ResNet-32/44/56 but *loses* on ResNet-20 — the smallest model
   tested, ~270K params. Soft weight-sharing shows the same sign flip (0.09pp cost on
   LeNet-5, 2.02pp on ResNet). **Compression regularizes only overparameterized
   models. At 1-10 KB nothing is overparameterized, so assume every quantization step
   costs accuracy.** One search-snippet source reports CIFAR-10 drops of 19.4pp at
   8-bit and 67.1pp at 4-bit for tiny nets *[UNVERIFIED — snippet only,
   mdpi.com/2079-9292/14/1/14, contradicts µNAS's clean int8; must be checked]*. The
   safe read: sub-8-bit requires real quantization-aware training, and the tiny
   regime is the hardest place to quantize, not the easiest.

**int8 is uninteresting for us.** PTQ within 1-2% of FP32 is well established
(arXiv:1806.08342, arXiv:2004.09602 *[verified, general-CNN claims not CIFAR-specific]*)
but it is a 4x win when we need 100x. Treat int8 as the *baseline representation*,
not a technique.

**Corrections.** HAQ is **arXiv:1811.08886**, not 1811.11721. XNOR-Net
(arXiv:1603.05279) reports **ImageNet only, no CIFAR-10** — its transferable idea is
per-channel scaling factors α, which cost one fp16 per output channel and are not
free. DoReFa-Net's abstract has no CIFAR-10 number; do not cite one.

**Difficulty.** Low for PTQ int8, medium for ternary QAT (straight-through
estimator, threshold/scale schedules), high for learned mixed precision.

**Composes with.** Entropy coding (strongly — see (c)); structured pruning (freely);
distillation (freely); *badly* with unstructured sparsity (see (d)).

---

### (c) Entropy coding the quantized weights

**Mechanism.** Treat the quantized level sequence as a symbol stream and code it at
its empirical entropy with arithmetic / range / rANS coding, or cluster weights and
Huffman-code the cluster indices (Deep Compression). The gain over fixed-width
packing is `log2(k) - H` bits per weight; the price is a codebook, a frequency
table, and decoder code.

**Measured, this session, by the coding sweep** *[measured, N=4000 symbols,
including container overhead — scripts are in the session scratchpad]*:

| distribution | H | byte+gzip | bitpack+gzip | bitpack raw |
|---|---|---|---|---|
| k=2, 50/50 | 1.000 | 1.458 | 1.046 | 1.000 |
| k=3 ternary, 60% zero | 1.371 | 1.950 | 1.558 | 2.000 |
| k=3 ternary, 80% zero | 0.922 | 1.402 | 1.204 | 2.000 |
| k=4 Laplacian | 1.802 | 2.476 | 2.000 | 2.000 |
| k=16 Laplacian | 3.518 | 4.238 | 3.710 | 4.000 |

**This corrects the project's working assumption.** "gzip on a bit-packed array is
useless" is true only for *near-uniform* distributions (k=2 at 50/50: gzip actually
*expands* 1.000 → 1.046 b/w). For **skewed** distributions bit-packing plus gzip
beat byte-per-weight plus gzip in every case tested, because the skew survives
packing as byte-level runs that LZ77 finds. The correct general statement is:

> **gzip lands 1.3-1.6x above entropy regardless of layout, and the gap *widens* as
> the distribution gets more skewed**, because DEFLATE's Huffman stage cannot assign
> a symbol fewer than 1 bit. Measured: p(zero)=0.50 → 1.29x above H; 0.80 → 1.40x;
> 0.95 → 1.50x; 0.98 → 1.63x.

**Actionable threshold: if the target is below ~1.2 effective bits/weight, gzip
cannot get there at any layout — you need rANS or arithmetic coding. Above ~2
bits/weight, gzip on bit-packed is within ~10% of entropy and not worth replacing.**

**Fixed overheads, measured** *[measured]*:

- **gzip container is exactly 20 bytes on empty input**; raw DEFLATE (`wbits=-15`)
  is **2 bytes**. Dropping the gzip wrapper is a free 18-byte win.
- **`.npz` costs ~199 B for one entry and ~177 B per additional array.** A 10-tensor
  model spends **1.8 KB on container metadata** before a single weight. Never use
  npz; ship one contiguous bitstream and keep shapes in the decoder.
- **Below ~1000 params, general-purpose compression is counterproductive**:
  sparse-ternary at N=500 gzips to 2.064 b/w, *worse* than raw 2-bit packing.
  (N=2000 → 1.512; N=10000 → 1.310; N=200000 → 1.215.)

**Codebook cost at tiny scale — the answer is yes, it dominates** *[measured]*. A
k-entry codebook (fp16 centroids + 16-bit counts) is 4k bytes. As a fraction of the
entropy payload: at N=2000, k=16 costs 7.2%, k=32 costs 11.2%, **k=256 costs 54.2%**
— a 256-entry codebook costs more than half of the weights it indexes. Deep
Compression's 8-bit conv codebooks are a large-model luxury. **At 1 KB use k<=4; at
10 KB use k<=16.** Soft weight-sharing (arXiv:1702.04008) independently supports
this: initialized with 17 mixture components, it typically converges to **~6
significant ones** *[verified]*.

**Worked 1 KB budget** *[measured/derived]*, N=2000 ternary weights at H=0.922:

```
entropy payload                    230 B
+ rANS state flush (4 B)           234
+ static frequency table (3x2 B)   240
+ centroid table (3x fp16)         246
+ per-layer scales (6x fp16)       258
+ shape metadata (6 layers x 4 B)  282 B   -> 1.130 effective bits/weight
```

Fixed overhead is **52 B = 18% of the artifact**. gzip on the byte array would give
370 B, so hand-rolled rANS is **~24% smaller**. But the line item *not* in this
budget is the decoder: a bit-unpacking loop is ~150 B of Python source, a full rANS
decoder ~600-800 B. **If the decoder source counts toward the artifact, that cost
alone can invert the rANS-vs-gzip conclusion at 1 KB.** This is a project decision
that must be made before methods are chosen (see Part 3).

**Literature** *[all verified]*:
- **Deep Compression (arXiv:1510.00149)**: k-means, 256 clusters (8b) conv / 32
  clusters (5b) FC. Huffman's *marginal* contribution over pruning+quantization is
  **18%-58% across the per-network tables** (LeNet-300-100 32→40x, LeNet-5 33→39x,
  AlexNet 27→35x, VGG-16 31→49x) — cite the tables, not the paper's "20-30%" prose.
  Final rates: AlexNet 4.0 bits/weight **+ 3.2 bits/index**. Note the index bits
  nearly equal the weight bits — sparse encoding is not free, which is (d)'s thesis.
- **DeepCABAC (arXiv:1905.08318)**: context-adaptive binary arithmetic coding, the
  entropy coder adopted into **MPEG-7 Part 17 NNR**. VGG16/ImageNet 63.6x with no
  accuracy loss = **~0.50 bits/weight**.
- **Entropy Penalized Reparameterization (arXiv:1906.06624)**: learned probability
  model + entropy penalty *during training*, arithmetic coding after. **VGG-16 on
  CIFAR-10: 60 MB → 101 KB (590x) at 10.0% error.** LeNet5-Caffe/MNIST → **2.84 KB**.
  This is the most directly relevant compression paper in the survey — training the
  weights to be codeable is strictly better than coding whatever training produced.
- **MIRACLE (arXiv:1810.00440)**: bits-back coding of samples from a variational
  posterior. VGG-16/CIFAR-10 **384 KB @ 6.57% error**, or **135 KB @ 10.0%**.
- **CERWU (arXiv:2505.18758)**: rate-constrained quantization + entropy coding,
  ResNets on CIFAR-10, reports **Pareto fronts of minimum bits/weight** at 95% and
  99% of original accuracy, and explicitly counts per-layer grid overhead. **This is
  the closest published analogue to our deliverable and should be read in full.**

**Difficulty.** Low (gzip), medium (rANS encoder+decoder ~200 lines), high (learned
entropy model trained jointly, i.e. the EPR approach).

**Composes with.** Ternary/low-bit quantization (its input), pruning (which supplies
the skew it monetizes), weight clustering. Composes *poorly* with binary at balanced
sign, which has no entropy to recover.

---

### (d) Extreme pruning + compact sparse encoding — and why it loses

**Mechanism.** Zero out most weights (magnitude, lottery-ticket rewinding, movement
pruning) and store only the survivors plus an addressing structure.

**Reachable sparsity** *[verified, Frankle & Carbin arXiv:1803.03635 Fig. 2]*:
Conv-2 (4.3M) within 1pp at Pm=8.8%; Conv-4 (2.4M) at 9.2%; Conv-6 (1.7M) at 15.1%;
VGG-19 (20M) exceeds initial accuracy at Pm>=1.5%; ResNet-18 (271K) at Pm>=11.8%.
**Note the direction: the smaller the net, the *less* sparsity it tolerates.**
Sparsity is a symptom of overparameterization, and tiny nets have none to give.

**The mask cost analysis — this is the decisive part** *[derived]*. For an N-param
layer with k survivors at density p = k/N:

| Scheme | Bits | N=10,000, k=500 |
|---|---|---|
| (i) dense bitmask | N | **1,250 B** |
| (ii) index list (COO) | k·ceil(log2 N) = 500·14 | **875 B** |
| (iii) Golomb-Rice delta, M=16 | ~500·(4+1+20/16) | **~391 B** |
| (iv) **information floor** log2 C(N,k) | 2,858 bits | **357.3 B** |
| — arithmetic-coded bitmask | N·H(0.05) = 2,864 bits | 358.0 B |
| *weight payload at 4 b/w* | 500·4 | *250 B* |

Two things fall out. **First, an arithmetic-coded bitmask attains the combinatorial
floor** (358.0 vs 357.3 B, a 0.2% gap) — so entropy-coding the mask is not an
approximation to the optimum, it *is* the optimum, and there is no cleverer encoding
left to find. A raw bitmask is 3.5x over; an index list 2.4x over. **Second, the
mask costs more than the weights it addresses** (357 B of addressing for 250 B of
values).

The scale-free way to see it: address cost per *surviving* weight is **H(p)/p bits**,
independent of N *[derived]*:

| p | H(p) | **H(p)/p bits per survivor** |
|---|---|---|
| 0.5 | 1.000 | 2.00 |
| 0.1 | 0.469 | 4.69 |
| 0.05 | 0.286 | 5.73 |
| 0.01 | 0.081 | **8.08** |
| 0.001 | 0.011 | **11.41** |

At 1% density you pay 8.08 bits of address for every weight you keep — twice the
cost of the 4-bit weight it points at.

**Break-even rule** *[derived]*: an unstructured-sparse net with k survivors at b
bits/weight costs the same as a **dense net with k·(1 + H(p)/(p·b))** parameters:

| b | p=0.5 | p=0.1 | p=0.05 | p=0.01 |
|---|---|---|---|---|
| 8 | 1.25x | 1.59x | 1.72x | **2.01x** |
| 4 | 1.50x | 2.17x | 2.43x | **3.02x** |
| 2 | 2.00x | 3.35x | 3.87x | **5.04x** |

A 99%-sparse net at 4 bits must beat a dense net with **3x as many parameters** just
to break even. Combined with the LTH finding that small nets tolerate *less*
sparsity, and the scaling law saying 3x params is worth ~15-18% error reduction, the
trade is a clear loss. Concretely *[derived]*: LTH's best Conv-6 ticket keeps 257K of
1.7M weights; at 4 bits + optimal mask that is **~259 KB** for Conv-6-level accuracy,
while LEMONADE gets **94.5% at ~190 KB and 91.1% at ~47 KB**. The pruned artifact is
bigger *and* worse.

**Note the perverse interaction: quantization and unstructured sparsity fight each
other.** Every bit you shave off the weights makes the address relatively more
expensive. The two techniques you most want to stack are the two that stack worst.

**Verdict: do not pursue unstructured pruning.** Hoefler et al.'s survey
(arXiv:2102.00554 §2.2) reaches the same place from the systems side: *"No sparse
storage scheme offers benefits for less than 10% sparsity"* *[verified]*.

**Two escape hatches that do work:**

1. **Structured sparsity is free.** Dropping whole channels or filters changes the
   *shape*, which costs a few bytes of architecture descriptor and zero mask. This is
   exactly why µNAS holds the 11.4 KB record.
2. **Seeded pseudorandom masks are free.** LFSR-pruning (arXiv:1911.04468)
   *[verified]* generates indices from a stored seed at inference — *"we no longer
   need to store the sparse weight addresses"* — reporting **1.51x-2.94x memory
   reduction** vs 4-8-bit indexed pruning. The price is that you cannot *choose*
   which weights survive, so magnitude selection is gone. **At tiny scale, where
   H(p)/p dominates, seeded random sparsity is the only unstructured sparsity worth
   considering, and it is under-explored** — the paper reports MNIST only, no usable
   CIFAR-10 number.
3. N:M / block sparsity amortizes the mask across a group and is a middle path.

**Prefer dense ternary over sparse-with-indices.** A zero *symbol* in an entropy
coder is nearly free; a zero *address* is not. This single substitution captures most
of pruning's benefit at none of its cost.

---

### (e) Weight sharing / hashing

**Mechanism.** Force many virtual weights to share few real values. HashedNets uses a
hash function on the index pair plus a random sign to decorrelate collisions;
FreshNets does the same in the DCT frequency domain; circulant projections replace an
FC weight matrix with a single vector plus a sign diagonal, applied via FFT.

**The valuable property: the hash costs zero storage** — it is recomputed from
indices *[verified, arXiv:1504.04788]*. At 1 KB, where every table is expensive, a
zero-byte structure is worth a lot.

**Datapoints** *[all verified]*:
- **HashedNets has no CIFAR-10 results** — MNIST and 7 MNIST variants only. Its
  degradation from 8x to 64x compression is modest on easy tasks (MNIST 1.45→2.79%)
  and severe on hard ones (rot-MNIST 11.17→18.04%, +3.5pp). That gradient is a
  warning for CIFAR-10.
- **FreshNets (arXiv:1506.04449)** *does* have CIFAR-10: 1/16 compression →
  **21.42% error**; 1/64 → **30.79% error** (69.2% accuracy). Sobering calibration
  for aggressive hashing.
- **Circulant projections (arXiv:1502.03436)**: CIFAR-10 CNN, 4x smaller for
  15.60% → 16.71% error (+1.11pp).
- **Tiled Bit Networks (arXiv:2407.12075)** is the strongest result in this family
  and is really structural sharing expressed as a bitwidth: ResNet-18 on CIFAR-10 at
  **0.256 bits/weight → 93.1%** (matching FP32), **0.069 b/w → 91.2%**. At 0.069 b/w
  an 11M-param ResNet-18 is ~95 KB — directly relevant to the 100 KB point, though it
  inherits ResNet-18's structural prior rather than shrinking the architecture.

**Composition caution:** weight sharing and value quantization are **partially
redundant** — both reduce the number of distinct values, and HashedNets over an
already-ternary net has little left to share. TBN is the exception because it shares
*tiles* (structural, not value-level), so it does stack with value quantization.

**Difficulty.** Medium (HashedNets/FreshNets), medium-high (TBN).

---

### (f) Procedural / seed weights — mechanically elegant, empirically dominated

**Mechanism.** Ship a PRNG seed that regenerates a large frozen random tensor, plus a
small number of learned coefficients that mix or mask it. Variants: random-subspace
training (Li et al.), supermasks over random weights, BatchNorm-only training,
hypernetworks, low-rank adapters over a random base.

**Intrinsic dimension (Li et al., arXiv:1804.08838, Table 1)** *[verified]*:

| Dataset | Net | D | d_int90 | D/d |
|---|---|---|---|---|
| MNIST | FC | 199,210 | 750 | 265x |
| MNIST | LeNet | 44,426 | 290 | 153x |
| **CIFAR-10** | **FC** | 656,810 | **9,000** | **73x** |
| **CIFAR-10** | **LeNet** | 62,006 | **2,900** | **21x** |
| ImageNet | SqueezeNet | 1,248,424 | >500K | <2.5x |

The paper **explicitly proposes the seed artifact**: store the seed for θ0, the seed
for the projection P, and the d coefficients — *"compression by a factor of 260x from
793 kB to only 3.2 kB"*, for the **MNIST FC** case *[verified]*.

**Byte verdict for CIFAR-10** *[derived]*: 2,900 coefficients = 11.6 KB fp32,
**2.9 KB at int8**, ~1.5 KB at int4, plus two 4-byte seeds. But d_int90 buys 90% of
the *host's* baseline, and LeNet on CIFAR-10 sits around 58-63% *[UNVERIFIED RECALL —
weak search snippets only]*, so **~2.9 KB buys roughly 53-57%**, against µNAS's
86.5% at 11.4 KB. **And note the trend: D/d collapses as the task gets harder** —
265x on MNIST, 21x on CIFAR-10, <2.5x on ImageNet. The trick degrades exactly where
it is needed.

**Supermasks over random weights — the mask is not cheaper than the weights.**
Zhou et al. (arXiv:1905.01067) *[verified]*, CIFAR-10, untrained random weights +
learned supermask with dynamic weight rescaling: Conv-2 **66.0%**, Conv-4 **72.5%**,
Conv-6 **76.5%**, vs trained baselines 69.2 / 75.4 / 78.3%. Heuristic masks only
reach 37-41%. Ramanujan et al. (arXiv:1911.13299) *[verified]* find optimal
**k in [30,70]%**, peaking near 50% because that maximizes the number of candidate
subnetworks.

That optimum is precisely the worst case for compression. **At p=0.5, H(p)=1 bit per
parameter — an entropy-coded supermask is exactly 1 bit/weight of the *full* net and
incompressible.** Conv-6 has 1.7M params, so a Conv-6 supermask is **~213 KB for
76.5%** *[derived]*; Conv-2 is ~538 KB for 66.0%. Coates' K-means + linear SVM gets
77.9% with no backprop at all. **Supermasks only ever looked good against a 32-bit
float baseline.** The honest framing is that a 50% supermask is a 1-bit-per-weight
code — the same budget as a binary sign network — and there is no published evidence
it beats one.

**BatchNorm-only training (arXiv:2003.00152)** *[verified]* — the cleanest seed
method, since the artifact really is seed + γ,β:

| Net | trainable BN params | BN-only acc | full-train baseline |
|---|---|---|---|
| ResNet-14 | 1.12K | 48% | ~91% |
| ResNet-110 | 8.29K | **69.5%** | ~93.3% |
| ResNet-866 | 64.7K | **82%** | ~93% |

8.3 KB → 69.5% against µNAS's 11.4 KB → 86.5%; 64.7 KB → 82% against LEMONADE's
47 KB → 91.1%. **Dominated by 13-17 points at every budget checked** *[derived]*.

**Weight Agnostic Neural Networks (arXiv:1906.04358): no CIFAR-10 number was
found; do not cite one.** *[not verified either way]*

**Verdict on (f): every procedural/seed variant verified here is dominated by a
well-designed small dense net at equal bytes.** The mechanism is sound and the byte
accounting is honest — the accuracy per coefficient is simply worse. The failure mode
is structural and worth stating once: **the artifact scales with the *host* network's
parameter count, not with the degrees of freedom actually learned**, so the trick
only pays when the host is enormous and the task is easy. CIFAR-10 at 1-10 KB is
neither.

**The one unexhausted variant** is seeded random *sparsity* over a **small** dense
net (LFSR-style, see (d)): zero mask bytes, and unlike supermasks it does not require
carrying a huge random host. That is the piece of this family worth an experiment.

---

### (g) Distillation into a tiny student — free in bytes, but a weak lever

**Mechanism.** Train the small student against the teacher's softened logits rather
than (or alongside) hard labels. The teacher is discarded, so **KD is the only family
here that costs exactly zero artifact bytes** and composes with everything.

**Datapoints** *[all verified]*:
- **Hinton et al. (arXiv:1503.02531) contains no CIFAR results whatsoever** — MNIST,
  speech, and JFT only. Cite it for the method, never for CIFAR evidence.
- The standard CRD/ReviewKD benchmark is **CIFAR-100** with students >=0.27M params.
  Vanilla-KD lift there is **+0.83 to +4.33 pp**, largest for architecturally
  mismatched students (wrn-40-2→ShuffleNetV1 +4.33) and smallest for well-matched
  ones (resnet32x4→resnet8x4 +0.83).
- **TAKD (arXiv:1902.03393) Table 1, CIFAR-10**: plain CNN 70.16 → **72.57 (+2.41)**;
  ResNet 88.52 → **88.65 (+0.13)**.
- **Cho & Hariharan (arXiv:1910.01348)**: WRN16-1 student error *increases*
  monotonically as the teacher grows (7.681 → 7.733 → 8.028% for WRN16-3/4/8). The
  capacity gap is real and it bites hardest on small students.
- **arXiv:2605.31191**: CIFAR-10 KD gains for ResNet-18 students are statistically
  indistinguishable from zero. And the finding that should reorder our priorities:
  **swapping the ImageNet stem for a CIFAR stem (3x3/s1, no maxpool) is worth +5.50
  to +7.15 pp — "more than 25x the largest KD gain" — with the largest effect on the
  smallest model.**
- **Ba & Caruana (arXiv:1312.6184)**: the shallow CIFAR-10 mimic reaches 14.2% error
  only by spending ~70M params. **Distillation rescues a *wide* student, not a
  *small* one.**

**Gap:** there is **no published CIFAR-10 KD result with a student under ~100K
parameters, and none with an MLP student** *[verified by search]*. Anything we
produce in the 1-50 KB regime is new — and there is no baseline to cite, only one to
establish.

**Verdict.** Budget KD as a **+1 to +3 pp finishing move at zero byte cost**, most
likely at the low end. Take it — free is free — but do not build the frontier on it,
and get the input stem right first, which the evidence says is 5-25x the leverage.

**Difficulty.** Low.

---

### (h) Exploiting CIFAR structure — highest leverage, with one expensive trap

**Mechanism.** Replace learned early layers with a fixed transform (PCA, DCT,
whitening, random projections, patch dictionaries) and train only a small head.

**Datapoints** *[all verified]*:

| Method | Features | Accuracy | Source |
|---|---|---|---|
| kNN + PCA-30 | 30 | 41.78% | arXiv:1611.04905 |
| LogReg + PCA-200 | 200 | 41.04% | arXiv:1611.04905 |
| Random Fourier + NCM (RanDumb) | 1K / 25K | 41.6% / 55.6% | arXiv:2402.08823 |
| Random cosine features + linear | 4,096 | 74.3% | arXiv:1602.05310 |
| K-means (hard) 1600 + linear SVM | 6,400 | 68.6% | Coates et al. 2011 T1 |
| **K-means (triangle) 1600 + SVM** | 6,400 | **77.9%** | Coates T1 |
| **K-means (triangle) 4000 + SVM** | 16,000 | **79.6%** | Coates T1 |
| Random *Gaussian* patches, 1 layer + linear | — | **78.6%** | arXiv:2101.07528 |
| SimplePatch (data patches) + linear | 10K | **85.6%** | arXiv:2101.07528 |
| SMT + kNN | 384/patch | 81.1% | arXiv:2209.15261 |
| Myrtle10-Gaussian kernel + flips | — | 89.8% | arXiv:2003.02237 |

Coates' configuration throughout: whitening, 6x6 receptive field, **stride 1**
(stride 2 costs >=3pp), 4-quadrant pooling.

**The trap, and the sharpest single insight in this survey.** *A learned basis is not
free, and at our budgets the basis dominates the head by two orders of magnitude*
*[derived]*:

- LogReg on PCA-200: head = 2,000 params = 2 KB. But the PCA basis is 200x3072 =
  **614 KB at int8**. The artifact is 616 KB for 41%.
- Coates K-means-1600: dictionary 1600x108 = **173 KB**, plus a 6400x10 head =
  **64 KB**. **~237 KB for 77.9%** — over budget, and beaten by µNAS at 11.4 KB.
  The 4000-feature version is ~592 KB for 79.6%.
- RanDumb: the projection is seeded and therefore free, but the readout needs class
  means plus a shared covariance ~1M values. ~1 MB for 41.6%.

The general form *[derived]*: **a linear readout over F features costs 10·F values.
A budget of B bytes at b bits/weight affords only F ≈ 8B/(10b) features — at 1 KB
and 4 bits, F ≈ 205.** Random and fixed feature banks are byte-hostile precisely
because the readout scales linearly with feature count, and their accuracy needs
thousands of features. This is the byte-level argument for why **convolution wins**:
weight sharing decouples representational width from parameter count.

**So the fixed-basis idea only pays when the basis costs zero (or near-zero) bytes.**
That means analytic (DCT), seeded-random, or *tiny and data-derived*. There is
exactly one verified instance of the third, and it is excellent:

**Frozen patch-whitening stem.** airbench (arXiv:2404.00498v2) *[verified]* calls its
frozen patch-whitening first layer **"the single most impactful feature"**; adding it
**more than doubles training speed** to 94%. In hlb-CIFAR10 the same layer is
`Conv2d(3→12, k=2)`, i.e. **144 real parameters ≈ 144 bytes at int8**, expanded to 24
channels by negation *[verified from hlb source, derived byte count]*. (One survey
thread reported airbench's stem as 24 filters of 3x3x3 = 648 values; hlb's is
unambiguously 2x2. Resolve against airbench's own source before quoting a number for
airbench specifically.) Corroborating evidence that the *front-end* is where the
accuracy-per-byte is: SMT's ablation shows whitening alone is worth **~+4pp**
(71%→75%); PCA-30 beats raw pixels by 8pp for kNN; and replacing learned patches with
Gaussian noise costs **8.1 pp** *[verified]* — so a small amount of data-derived
structure genuinely matters and cannot be entirely seeded away.

**The most promising untested idea in this survey: a fixed DCT (zero-byte) basis in
place of a learned PCA basis.** For natural images PCA converges to the DCT
asymptotically, so a zero-byte analytic basis may recover most of what a 614 KB
learned basis provides. **No CIFAR-10 result exists for a fixed DCT basis + tiny
head** — everything found uses DCT *inside* a trained deep net *[verified by
search]*. Also missing: any published nearest-class-mean-on-raw-pixels CIFAR-10
number.

**Difficulty.** Low (DCT/whitening stem), medium (dictionary learning).

---

## Part 3 — Ranked plan

### 3.0 One decision needed before anything else

**Does the decoder source count toward the artifact?** At 100 KB it is noise; at
1 KB a bit-unpacker (~150 B) versus an rANS decoder (~600-800 B) is 15-80% of the
entire budget, and it flips the method ranking. Related sub-decisions: is the score
raw bytes, gzipped bytes, or both reported separately (they reward opposite
representations — see Part 2c); and does the frozen whitening stem have to ship, or
may it be recomputed from the training set (it must ship, if "self-contained" means
what it says). **Recommend: score = raw bytes of a single self-contained file
including decoder, with gzipped size reported alongside.** That is the honest MDL
reading and it is the only version of the metric that cannot be gamed.

### 3.1 The ~10 KB point — what most plausibly wins

**Target to beat: µNAS, 86.49% at 11.4 KB** (8-bit dense, structurally pruned, no
entropy coding). Every byte-side technique in Part 2 is unexploited by it, which is
where the headroom is.

**Recommended stack:**

1. Small dense convnet, hlb/airbench-shaped but ~50-70x narrower, with a **frozen
   2x2 patch-whitening stem (~144 B)** and a CIFAR-appropriate stem geometry
   (3x3/s1, no maxpool — worth +5.5 to +7.2 pp per arXiv:2605.31191).
2. **Structured channel pruning only.** Zero mask bytes. No unstructured sparsity.
3. **Ternary QAT** (TWN/TTQ-style, learned per-layer scales) — not binary. Fold
   BatchNorm into the preceding conv at export.
4. **rANS entropy coding** of the ternary stream with k=3 and a static frequency
   table, biased toward zero during training (an EPR-style entropy penalty on the
   weights is the principled version and is the single most relevant published
   technique, arXiv:1906.06624).
5. **Free training tricks from Part 1**, all of them: EMA, label smoothing, CutMix,
   long training, and **expanded TTA** (flip + multi-crop; inference time is free).
6. **KD from an hlb-CIFAR10-class teacher** (+1-3 pp, zero bytes).

**Budget arithmetic** *[derived]*. 10,000 bytes minus ~300 B of decoder+metadata+stem
leaves ~9,700 B = 77,600 bits. At an entropy-coded ternary rate of ~1.2 b/w that is
**~64,000 weights** — versus µNAS's 11,400. **The bet is that 5.6x the parameters at
1/6.7 the bits per parameter is a net win.** Sanity check against the scaling law:
5.6x params implies an error ratio of 5.6^(-0.156) = 0.76, so 13.5% error → ~10.3%
(89.7%); subtract 1-2 pp for ternary at small scale and add 1-3 pp for KD + free
tricks. **Expected: 87-90% at 10 KB**, i.e. a real but not overwhelming margin over
µNAS. If it lands at 88%+ it is a defensible SOTA claim.

**Fallback if ternary QAT is unstable at this width:** int4 with k=16 clustering +
entropy coding, ~2.9 effective b/w, ~26,000 weights. Lower risk, smaller win.

### 3.2 The ~1 KB point — greenfield, and a different problem

**No 10-class CIFAR-10 result under ~10 KB is published.** Any number we produce is
the first, so the goal is a credible, honestly-measured point, not a margin.

**Budget arithmetic** *[derived]*. 1,024 B minus a minimal decoder (~150-250 B, so
keep the codec simple — this argues for bit-packing over rANS at this size), minus
~50 B of metadata and scales, minus a ~144 B whitening stem, leaves **~600 B ≈ 4,800
bits**. That is:
- ~4,000 weights at ~1.2 b/w (entropy-coded ternary), or
- ~1,200 weights at 4 b/w, or
- ~600 weights at 8 b/w.

Note the measured warning from Part 2c: **below ~1000 params general-purpose
compression is counterproductive**, and a k=256 codebook alone would exceed the
budget. **k<=4, one global or per-layer scale, one flat bitstream.**

**Recommended stack:** frozen whitening or **zero-byte DCT** stem → a ~3-4K-parameter
ternary micro-convnet (structured, no masks) → global pool → 10-way head, trained
with KD from a strong teacher and every free trick, with expanded TTA at eval.
Consider dropping the stem's 144 B in favor of an analytic DCT basis if it holds up —
that is 14% of the budget recovered.

**Honest accuracy expectation.** The scaling-law fit does not extend below 22K
params, so extrapolation is unjustified; I will not forecast a number I cannot
support. What can be said: the floor is ~41% (linear on raw pixels, which itself
needs 30 KB and is therefore not even reachable at 1 KB), and µNAS's search space
reaches 685 params at 77.5% on the *2-class* problem. **A defensible guess is 55-70%
at 1 KB, with wide error bars, and that guess should be stated as a guess.**

### 3.3 The ~50 KB and ~100 KB points

Straightforward extensions of 3.1 with more width: at ~47 KB LEMONADE gives 91.1% and
YYNet 89.46% at ~53 KB; at ~190 KB LEMONADE gives 94.5%. Our ternary+rANS stack
should reach these budgets at ~5-6x the parameter count of the int8 references, and
targeting **92-93% at 100 KB and ~90-91% at 50 KB** is reasonable. These points are
useful for the Pareto curve but they are not where the contribution is.

### 3.4 What NOT to do — the negative results, stated plainly

Four families were checked and are **verified losers on bytes-per-accuracy against a
plain, well-designed, structurally-pruned, low-bit convnet, at every budget checked**:

- **Unstructured magnitude / lottery-ticket pruning** — the H(p)/p arithmetic makes
  it unwinnable below ~100 KB, and the optimal mask encoding is already known
  (arithmetic-coded bitmask hits the combinatorial floor to 0.2%), so there is no
  headroom to find.
- **Supermasks over random weights** — 1 bit per parameter of the *full* host net,
  incompressible at the optimal k≈0.5. Conv-6: ~213 KB for 76.5%.
- **BatchNorm-only training** — dominated by 13-17 pp at equal bytes.
- **Intrinsic-dimension random subspaces** — CIFAR-10 has the worst D/d ratio (21x)
  in Li et al.'s table; ~2.9 KB buys ~55%.

Also: **compressing a big net is not a route to 1-100 KB.** Even 482x on VGG-16 lands
at ~1.1 MB. **Design small; do not compress big.**

### 3.5 Top three experiments, in order

**1. Build the byte-accounting harness and measure the baselines.** Until export is
real, every number in this document is a projection. Deliver: a single-flat-bitstream
exporter (BN folded into convs, normalization folded into the stem, no npz, raw
DEFLATE not gzip, shapes in the decoder), a decoder-size accounting policy per 3.0,
and measured raw+gzip bytes for three reference points — logistic regression, a small
int8 convnet, and a µNAS-sized net. **This de-risks everything else and will
immediately expose whether the µNAS/LEMONADE `[C-est]` int8 byte figures in Part 1.5
are trustworthy.** It is also the piece of the project that cannot be borrowed from
the literature, because nobody in this literature reports artifact bytes honestly.

**2. The width x bitwidth sweep — this produces the entire Pareto frontier in one
experiment.** Fix one architecture family (whitening stem + narrow convnet + global
pool + linear head), sweep width across ~4 points spanning 3K-100K params, and
bitwidth across {int8, int4, ternary}, all with the free training tricks and KD.
Measure accuracy against *post-entropy-coding effective bits/weight*, not nominal
bitwidth. This directly answers the central open question — **does 5x the parameters
at 1/6 the bits beat µNAS's 11.4K int8?** — and yields the deliverable curve as a
byproduct. Run the 10 KB row first.

**3. The zero-byte-basis test at 1 KB.** Compare, at a fixed ~600-byte weight budget:
(a) frozen 2x2 patch-whitening stem (~144 B), (b) analytic DCT stem (0 B), (c) no
stem. This tests the single most promising untested idea in the survey, occupies the
empty 1 KB cell, and the answer is cheap to obtain. If (b) matches (a), it is a free
14% of the 1 KB budget and a genuinely novel result.

*Deferred but worth a later look:* seeded/LFSR random sparsity over a small dense net
(the only unexhausted procedural idea), and an EPR-style entropy penalty applied
during training rather than coding whatever training happens to produce.

---

## Open items and known gaps

- **Unverified:** LeNet-on-CIFAR-10 baseline accuracy (~58-63%), used in the (f) byte
  verdict. The conclusion does not hinge on it but the number should not be quoted.
- **Unverified:** the tiny-net quantization-fragility claim (19.4pp at 8-bit, 67.1pp
  at 4-bit, mdpi.com/2079-9292/14/1/14) — snippet only, and it contradicts µNAS's
  clean int8. Fetch the table before relying on it either way. **Experiment 2 settles
  this empirically anyway.**
- **Unverified:** rANS practical redundancy vs entropy. Expect near-entropy dominated
  by frequency-table precision and the state flush; measure rather than quote.
- **Discrepancy:** airbench's whitening stem reported as both 2x2 (144 params, from
  hlb source) and 3x3 (648 params). Check airbench's source directly.
- **Discrepancy:** µNAS cites "LEMONADE ~91.77% @ 10K params"; LEMONADE's own paper
  has no row below 38K params. Do not use that cell.
- **Unread:** CERWU (arXiv:2505.18758) — rate-constrained quantization with published
  bits/weight Pareto fronts on CIFAR-10 ResNets. The closest published analogue to
  our deliverable; its appendix tables should be read in full before we publish ours.
