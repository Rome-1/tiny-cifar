# The small-model landscape

> **Verification warning.** Part of this survey was drafted in a pass that
> asserted sources it had not actually read, and the affected claims were removed
> when that was caught. Items marked unverified below are exactly that, and the
> verified marks elsewhere in this file have not all been independently
> re-checked. Treat this document as leads to follow rather than as settled
> citations, and confirm anything before relying on it.

Who else optimizes for smallness, what they actually measure, and whether anyone
measures *bytes of a self-contained artifact* — which is the axis this project is on.

This is a **landscape of projects and communities**, complementary to the three
method surveys already in `docs/`.
`method-survey.md` and `exotic-methods.md` cover the compression *techniques*;
`cmix-and-prequential.md` covers the coding theory.
This document covers the *ecosystem*: who is competing, on what metric, and where the
empty seat is.

**Verification convention.** Every claim is marked `[V]` (a page was fetched and the
text read), `[V-1]` (verified from a single source that could not be corroborated), or
`[UNVERIFIED]` (recalled but not confirmed — do not cite without checking).
Numbers that could not be verified are absent, not guessed.

## The one-sentence finding

Two mature traditions measure bytes of a self-contained artifact — **data compression
competitions** (Hutter Prize, LTCB, Calgary) and the **demoscene / code golf** — and
neither has anything to do with image classification;
the machine-learning world measures parameters, FLOPs, latency, energy, and dollars, and
treats size as a *fairness bucket or hard constraint, never the ranked quantity*.
Exactly one live ML effort scores bytes of a shipped artifact — **OpenAI's
parameter-golf (2026)** — and it is language modeling, with bytes as a cap rather than
the objective.
The cross-product is essentially unoccupied, and the CIFAR-10 cell of it is empty.

---

## 1. Speedruns — time-to-target, not size

### modded-nanogpt (Keller Jordan et al.) `[V]`

<https://github.com/KellerJordan/modded-nanogpt>

- **Optimized:** wall-clock minutes to reach **≤3.28 validation loss on FineWeb** on
  **8×H100**. Nothing else.
- **Rules**, four of them, verbatim: (1) *"Not modify the train or validation data
  pipelines. (You can change the batch size, sequence length, attention structure etc.;
  just don't change the underlying streams of tokens.)"* (2) attain ≤3.28 mean val loss.
  (3) *"Not use any extra `torch._inductor.config` or `torch.compile` flags."*
  (4) *"Run faster than the prior record when baselined on the same hardware."*
  Architecture, optimizer, initialization, and parameter count are **all free** — only
  the token stream and the loss target are pinned.
- **A statistical gate**, which is the part worth copying: *"Due to inter-run variance,
  submissions must provide enough run logs to attain a statistical significance level of
  p<0.01 that their mean val loss is ≤3.28."*
- Official runs are timed on 8×H100 via PrimeIntellect, who sponsor validation runs.
  A second track targets 2.92 val loss (Karpathy's 350M llm.c baseline).
- **Record:** #86, **1.266 minutes**, @aryavohra, **27 May 2026** — "algebraic rewrite
  of XSA, same math faster execution". The table runs 86 dated entries from Karpathy's
  llm.c baseline of 45 minutes (28 May 2024) — a ~35× improvement over 24 months. The
  shape of the ladder is instructive: #3 (24.9 min, Oct 2024) introduced Muon, #5
  (15.2 min) bundled ReLU², zero-init projections and QK-norm, and the last thirty
  records are single-digit-percent systems work (FP8 on one projection, fused kernels).
- **Bytes measured? No.** Model size is not constrained anywhere in the rules. The
  baseline is a 124M-parameter GPT-2, but that is a starting point, not a bound.
- **Muon's role:** Newton–Schulz orthogonalization in place of full SVD — momentum
  then spectral normalization; lower memory than Adam, better sample efficiency. It is
  the single most-cited algorithmic contribution to come out of the speedrun, and it
  escaped into the wider field.

**What we'd steal:** the *governance*, not the metric. A pinned data pipeline, a pinned
quality target, everything else free, a p<0.01 significance gate, and a dated
append-only table where each entry names one change — that is the minimum structure that
makes 86 records defensible instead of 86 anecdotes, and it is why the speedrun produced
a real optimizer rather than a tuned config.

### cifar10-airbench (Keller Jordan) `[V]`

<https://github.com/KellerJordan/cifar10-airbench>

- **Optimized:** seconds on a single A100 (400W) to reach a fixed accuracy, with
  *separate records per accuracy tier*.
  `airbench94_muon.py`: **94.01% in 2.59s, 0.29 PFLOPs**.
  `airbench96_faster.py`: **96.00% in 27.3s, 3.1 PFLOPs**.
  Legacy tiers at 94% (3.09s, 3.83s), 95% (10.4s), 96% (34.7s).
- Lineage, verbatim: it builds on hlb-CIFAR10 (**6.3 A100-seconds to 94%**), which
  builds on myrtle.ai's series (**26 V100-seconds**). Superseded by
  <https://github.com/hiverge/cifar10-speedrun> at **1.98s to 94.02%** `[V-1]`.
- **Bytes measured? No** — parameter count is not even reported. Columns are accuracy,
  seconds, PFLOPs. (~2.0M params by arithmetic over `airbench94_muon.py`; the repo never
  states it. `[V-1]`, derived.)
- **There is no leaderboard, no rules section, and no submission process** — the records
  are Keller's own, announced on X. `[V]`

**Two things to take from this.** First, positioning: this is our closest cultural
sibling — same dataset, public record-chasing, one honest number — on the orthogonal
axis, and *the CIFAR-10 side has the speedrun culture but never built the speedrun
institution*. A maintained public frontier would be filling a gap the accuracy side left
open, not competing with it. These optimize time, with size
entering only because bigger models are slower.

Second, and more concretely: `CifarNet`'s first layer is a **fixed, PCA-initialized,
sign-doubled whitening convolution** — not learned. That is the same component
`method-survey.md` names as the best accuracy-per-byte item it found, and `findings.md`
ranks #2 in what to try next. The fastest CIFAR-10 trainer in the world already ships it
as a frozen layer. That is a strong independent signal, and it is free under our
accounting.

---

## 2. Parameter golf and byte golf

### OpenAI `parameter-golf` — the closest methodological sibling that exists `[V]`

<https://github.com/openai/parameter-golf> — independently verified by two fetches.

- **Optimized:** validation **bits-per-byte on FineWeb** (lower is better),
  tokenizer-agnostic, subject to a hard size cap.
- **The size rule, which is nearly ours:** the budget is **16,000,000 bytes** (decimal,
  not MiB), and the rules state that *"the submission artifact is computed as code bytes
  plus compressed model bytes. All counted code should live in the `train_gpt.py`
  script."* Training is capped at 10 minutes on 8×H100; no network access at eval.
- **Leaderboard:** **1.0565** (codemath3000, 1 May 2026), 1.0576 (simonbissonnette),
  1.0586 (andrewbaggio1); baseline ~1.2244 BPB. Submission window closed 30 Apr 2026.
  A new record must *"beat the existing SOTA by at least 0.005 nats"* with run logs
  showing that improvement at **p<0.01**.
- **Bytes measured? Yes** — code + compressed weights, exactly the accounting in
  `harness.md`.
- **Why code bytes are counted at all**, and this is the crisp justification for our own
  most-contested rule: otherwise you smuggle weights into the training script and evade
  the weight cap. Merging code and weights into one budget is what makes an
  artifact-bytes metric well-posed. `harness.md` reaches the same conclusion in its own
  words ("without it, any amount of model could be smuggled into source as literals").
- Techniques visible in leaderboard entry descriptions: test-time training, GPTQ/AWQ-Lite
  quantization, LoRA-TTT, n-gram+neural hybrids, custom tokenizers, gated attention.
- Community volume is thinner than the press coverage implies — the main HN thread ran
  to about two comments. The activity is in the PR stream, not the discourse.

**This is the single most important find in the survey.** The measurement rule we
adopted independently is the one OpenAI adopted for language modeling. The differences
are that they fix bytes and optimize quality where we trace a Pareto frontier, and that
nobody has ported the framing to vision. That both validates the rule and leaves the
CIFAR-10 slot open.

**What we'd steal:** the statistical-significance gate on a new record (≥0.005 nats at
p<0.01). Our own `findings.md` already concedes a ~1 pp selection-noise fog over the
leaderboard; a significance gate is the standing fix, not just a caveat.

### "Parameter golf" / "neural golf" as a genre `[V]`

The term is live but it means **parameter count**, not bytes:

- **hexhowells.com, "Neural Golfing: 98% on MNIST with 575 parameters"** (26 Jan 2025)
  — 1,199,882-param baseline → hand-built CNN 1,055 params (98.06%) → Optuna 807
  (98.16%) → pruning 672 → PReLU swap → **575 params at 98%**. Bytes are never
  mentioned. <https://hexhowells.com/posts/neural-golfing.html>
- **stats.stackexchange.com Q656420** (Oct 2024) — 702 params at 98.2% MNIST, asking
  how to get under 1000. Zero answers.
- **neuralgolfing.com** — a site listing challenge categories including CIFAR-10, with
  parameter count as the stated metric. **`/challenges/cifar-10` returns HTTP 404 and
  the leaderboard is empty.** Evidence of latent demand, not a competitor.
- **Convolutional Differentiable Logic Gate Networks** (arXiv:2411.04732) — *"On
  CIFAR-10, we achieve an accuracy of 86.29% using only 61 million logic gates"*. Size
  in gates, not bytes; and 61M gates is enormous by our standard.

**The differentiator to lead with: bytes ≠ parameters.** Quantization, entropy coding,
and the code-vs-weights tradeoff are exactly the axis a parameter count cannot see. A
575-parameter MNIST net at float32 is 2.3 KB before you write a line of decoder; our
961-byte artifact ships a decoder *and* a conv bank *and* a head.

### Code golf for ML `[V]`

- **codegolf.stackexchange.com Q28207, "Recognize handwritten digits"** (May 2014) —
  the only genuinely byte-scored ML classifier challenge found anywhere. Score =
  `s·(1200−n)/1000`, with `s` = source bytes and `n` = correct out of 1000. Winner:
  Peter Taylor, **GolfScript, 101 bytes, 567/1000 = 56.7%**, score 63.933. Runner-up
  207 bytes / 41.4%, same author. **Two answers in twelve years.**
  <https://codegolf.stackexchange.com/questions/28207/recognize-handwritten-digits>
  - Its design flaws are instructive: the test set is 1,000 images drawn from MNIST's
    *training* split, and the scalar score conflates bytes with accuracy instead of
    tracing a frontier. Our harness avoids both.
- **codegolf.SE "Machine Learning Golf: Multiplication"** (Q187562) — minimize the
  *number of weights* in a net that multiplies two integers; best found 43. `[V]`
- The `machine-learning` tag on codegolf.SE is **empty — 0 questions**, and no
  CIFAR-flavored challenge was found. `[V, negative]`

> ⚠️ **A gap this survey could not close.** `codegolf.stackexchange.com` is unreachable
> from this environment (WebFetch is blocked at the domain, and the search budget was
> spent), so the codegolf tradition is documented here only through what the niche-hunt
> sweep recovered second-hand. **Specifically unverified and deliberately omitted:** the
> internals of Q28207's winning entry, the contents of the `neural-networks` tag, and
> any adjacent OCR-golf challenges. If the code-golf angle matters to a writeup, it
> needs a session that can reach the site. It is the one genuine source-code-bytes
> tradition and the survey is lopsided without it.

### js1k and the demoscene

- **js1k** (2010–2019): a 1024-byte cap, server-enforced. The 2019 entry **"Retro Neural
  Network" by Richard van der Dys, 994 bytes** `[V]`, trains a multi-layer network to
  draw the letter X, live, in under a kilobyte — framework, training loop, and rendering
  included. It is generation, not classification, but it is a real neural network inside
  1 KB and it is the only one found.
- **Could not verify** any demoscene intro shipping a neural network: pouet has no
  full-text search over descriptions and demozoo is Cloudflare-blocked. Treat "nobody
  has done it" as unproven in either direction. `[UNVERIFIED]`

<https://www.sizecoding.org> states the metric plainly — *"the size of these tiny
programs is measured by their total size in opcode bytes, and are usually presented as
an executable binary"* — with entries down to 16 and even 8 bytes, and **zero mention of
machine learning anywhere on the wiki**. `[V]`

This is the purest byte-minimization culture in existence and it is almost entirely
disjoint from ML. It is also where the procedural-generation instinct comes from, which
is precisely our "a PRNG seed is four bytes no matter how much it draws."

**What we'd steal:** two things. The discipline that *everything* in the artifact is
generated rather than stored — `findings.md` found this independently; the scene has
forty years of practice at it. And **packer discipline**: score the compressed artifact
and treat "which bytes cost the most" as a first-class debugging view. Our leaderboard
already reports raw/gzip/xz side by side for exactly this reason.

### tinygrad's line limit — real, enforced, and 25,000 `[V]`

The README states only philosophy — low line count is *"a guiding light"*, but
*"anything that remotely looks like code golf will be closed"*. **The actual limit is in
CI.** `.github/workflows/test.yml` runs:

```
- name: Repo line count < 25000 lines
  run: MAX_LINE_COUNT=25000 python sz.py
```

`sz.py` walks `tinygrad/` (excluding `runtime/autogen` and `viz/assets`) counting `.py`
and `.js`. **Python lines are counted by token span** —
`len(set(x for t in tokens for x in range(t.start[0], t.end[0]+1)))` over a whitelist of
`OP, NAME, NUMBER, STRING` — so **comments, blank lines, and docstrings are free**, and
semicolon-joining does not help.

Two lessons. The token-based counter is the right anti-gaming design: measure the thing
you care about, not its textual proxy. And the trajectory is the cautionary tale — Issue
#405, *"Get back to 1000 lines"* (geohot, Oct 2022), against a limit now at 25,000. **A
size limit with no adversarial pressure behind it ratchets.** `teenygrad` preserves the
original discipline with its own `sz.py` asserting <1000 lines.

The same "lines, never bytes" pattern holds for **llama2.c** (*"one simple 700-line C
file"*) and **micrograd**. Ethos-adjacent, metric-hostile — do not overclaim kinship.

---

## 3. TinyML and MCU-constrained ML — bytes are budgeted, never scored

This is the cluster people assume measures bytes. It budgets them against hardware; it
does not rank them.

### MLPerf Tiny (MLCommons) `[V]`

<https://mlcommons.org/benchmarks/inference-tiny/> · paper arXiv:2106.07597

- The image-classification task **is CIFAR-10**. Reference model "ResNet-8" —
  ResNetv1 with three residual stacks instead of four — at **96 KB int8 TFLite,
  86.5% top-1**.
- **Scored: latency (single-stream), accuracy, and optional energy (µJ/inference).
  Model size is not scored and not constrained.** Accuracy is an absolute floor of
  **≥85% top-1** in the closed division, set ~1.5 points below reference to absorb
  quantization.
- Latest round **v1.3, results 17 Sept 2025** — 70 results, 4 submitters. Prior rounds
  v0.5 (Jun 2021) through v1.2 (Apr 2024).

**What we'd steal:** the accuracy-floor-then-optimize contract, and ResNet-8/96 KB as
the canonical "what tinyML currently calls small" marker we are two orders of magnitude
below.

### µNAS (Liberis, Dudziak, Lane, EuroMLSys 2021) — the number to beat `[V]`

arXiv:2010.14246 · <https://github.com/eliberis/uNAS>

**Table 2, 10-class CIFAR-10: 86.49% at 11.4 KB model size, 15.4 KB peak RAM,
384K MACs** — confirmed by two independent fetches.

What the 11.4 KB *includes*, precisely:

- **int8 dense weights only, one byte per parameter.** Verbatim: *"NAS simply counts
  the number of parameters of a neural network at 8 bits = 1 byte per parameter."*
- **Excludes** activations (reported separately as peak RAM), interpreter/runtime code,
  biases, and quantization scale/zero-point tables.
- Pruning is **structured** (DPF, L2-norm channel removal) → dense storage, so there is
  no sparse-index overhead and none is charged. This is what makes the number
  defensible.
- **No entropy coding, no weight sharing, no sub-byte quantization anywhere.** The
  11.4 KB is uncompressed dense int8. *That is the headroom.*

Their objective is 4-way — `argmin {1−ValAcc, ModelSize, PeakMem, Latency}` — scalarized
per round Chebyshev-style with random weights.

Caveats to carry: there is **exactly one** 10-class CIFAR-10 point, not a frontier; the
repo ships search code but **no weights and no architecture spec** for the 86.49% model
(reproduction means re-running ~23 GPU-days); Appendix B concedes sparse-model sizes do
not account for mask storage; and the paper's LEMONADE comparison row is a *float32
parameter count*, not bytes, so it is not like-for-like.

### SpArSe (Fedorov et al., NeurIPS 2019) — the CIFAR numbers are 2-class `[V]`

arXiv:1905.12107. **SpArSe never reports a 10-class CIFAR-10 result.** §3 verbatim:
*"we also report on binary versions of these datasets, meaning that the classes are
split into two groups and re-labeled."* Every table column reads `CIFAR10-binary`.
Chance is 50%, not 10%. Best binary points: 73.84% at 0.78 KB model size / 1.28 KB
working memory.

Model size is `‖ω‖₀`, a count of non-zero weights at one byte each. The authors concede
*"(4)-(6) are reductive to varying degrees"* — **sparse-format overhead (indices, CSR
pointers, bitmasks) is not counted**, and Table 2 uses *unstructured* pruning, so those
counts are not byte counts at all.

### MCUNet (Lin et al., NeurIPS 2020 / V2 / V3) `[V]`

**V1 explicitly declines CIFAR:** *"We did not use datasets like CIFAR since it is a
small dataset with a limited image resolution (32×32), which cannot accurately represent
the benchmark model size or accuracy in real-life cases."* V2 has no CIFAR-10 at all.
V3 (on-device training, arXiv:2206.15472) reports **86.9% CIFAR-10**, but as a
*transfer-learning target* — fine-tuning the last two blocks of an ImageNet-pretrained
MCUNet-5FPS. The pretrained backbone is the artifact, so it says nothing about the
description length of a CIFAR-only classifier.

Optimizes against a *hardware* budget (256 KB/320 KB SRAM, 1 MB Flash), not a
description length. Profiled with their own TinyEngine codegen runtime, not TFLM — not
cross-runtime comparable. V1 concedes the numbers exclude im2col and other runtime
buffers.

### TensorFlow Lite Micro and the ARM ML-Zoo `[V]`

- **ML-Zoo is dead** — archived 18 July 2025, last content push March 2023.
  **Zero CIFAR-10 models.** Top-level README has no size column, and the `.tflite`
  blobs are Git-LFS pointers, so a naive `du` over a clone gives garbage.
- **tflite-micro is alive** but ships **no CIFAR-10 example**.
- **The interpreter tax — three published numbers measuring three different things:**
  1. **9,732 bytes** — TFLM framework flash on Cortex-M3, measured by subtracting a
     no-op baseline binary, **kernels excluded**. The reproducible one.
  2. **16 KB** — *"core runtime just fits in 16 KB on an Arm Cortex M3"*, vendor
     headline.
  3. **"less than 2 KB"** — the interpreter *object* only (TFLM paper, arXiv:2010.08678).
     Anyone quoting "TFLM costs 2 KB" against a flash budget is misusing it.

**The runtime tax is larger than µNAS's entire model.** That is the argument for our
self-containment rule: an artifact that leans on an interpreter has not accounted for
the bits the interpreter supplies.

### Edge Impulse — a product, not a leaderboard `[V]`

No public size-vs-accuracy leaderboard, no submissions, no held-out scoring, no CIFAR-10
content; the `eon-compiler-benchmarks` URL 404s. EON's own claims disagree across their
own sources (docs say *"25–65% less RAM / 10–35% less flash"*; the launch post says
*"25–55% less RAM, up to 35% less flash"*, then reports 37% ROM in its worked example,
exceeding its own ceiling). Vendor ranges, not measurements. The mechanism is
interesting though: EON is an AOT code generator, so "less flash" largely means "we
deleted the interpreter" — their claim and our byte objective measure the same thing
from opposite directions.

### The sub-KB cell — the prior survey's claim holds `[V]`

**No published 10-class CIFAR-10 result at or below 1 KB exists.** Every sub-KB CIFAR
number in the literature is a 2-class relabeling:

| Work | Accuracy | Size | Task |
|---|---:|---:|---|
| µNAS | 77.49% | 685 B | CIFAR-10 **binary** |
| SpArSe | 73.84% | 780 B | CIFAR-10 **binary** |
| BonsaiOpt | 73.05% | 0.50 KB | CIFAR-10-**2** |
| Bonsai | 73.02% | 1.98 KB | CIFAR-10-**2** |

Bonsai (ICML 2017) labels the dataset `CIFAR10-2` under a table headed "Binary
Datasets"; its multiclass table contains no CIFAR at all. Third-party confirmation in
Saha, Sandha & Srivastava (arXiv:2205.14550) Table V, whose legend reads *"B = binary
dataset, M = multiclass dataset"* and marks the Bonsai and SpArSe CIFAR rows `B`.

The actual 10-class KB-scale frontier, mostly from Müksch et al. (arXiv:2005.04968):

| Method | Accuracy (10-class) | Size |
|---|---:|---:|
| Bonsai, 10-class | 14.9% | 7.88 KB |
| ProtoNN, 10-class | 14.7% | 24.77 KB |
| FastGRNN | 48.2% / 53.3% | 7.57 / 15.80 KB |
| Direct-convolution CNN | **60.4%** | **5.39 KB** |
| Direct-convolution CNN | 62.9% / 64.3% | 8.65 / 19.91 KB |
| DWN (weightless, arXiv:2410.11112) | 57.51% | 23.4 KiB |
| µNAS | **86.49%** | **11.4 KB** |
| MLPerf ResNet-8 | 85% floor | 96 KB |

⚠️ **Accounting mismatch:** Müksch et al. state *"we keep the prediction parameters
full-rank and do not carry out quantization"* — likely fp32. Do not plot their rows on
one axis with µNAS's int8 rows without normalizing.

**Where this puts us.** Our 9,503 B / 64.29% point sits essentially on the Müksch
direct-convolution line (64.3% at 19.91 KB, unquantized) at half the size, and well
below µNAS. Our **961 B / 42.72%** point is in a cell with no published 10-class
competitor at all.

### Sub-KB artifacts that do exist — on MNIST, and mostly unreplicated `[V]`

The CIFAR-10 sub-KB cell is empty, but the *techniques* that get there exist and were
found on other tasks. These are a baseline menu, not results to cite as bars:

- **Bonsai** (Kumar, Goyal & Varma, ICML 2017) — a single shallow sparse tree over a
  sparse low-dimensional projection, fitting in **2 KB**. `[V]` The strongest
  peer-reviewed sub-KB precedent, and it measures real model bytes. Note the shape:
  *project down to a low dimension first, then a tiny decision structure.* Its
  10-class CIFAR-10 numbers are poor (14.9% at 7.88 KB, per Müksch et al. above) — the
  technique is the takeaway, not the result.
- **TBNN (Tiled Binary Normalized NNs)** —
  <https://github.com/joaocarvalhoopen/TBNN__Tiled_Binary_Normalized_Neural_Networks_in_Odin>.
  A shared circulant **bit tile** regenerates the weight matrix; LayerNorm with 3 floats
  per layer replaces biases. The `milli` preset claims 784-2048-2048-10 with 5,820,416
  virtual weights from 5,408 stored bits — a **720-byte model at ~0.00093 bits/weight,
  ~91–92% MNIST**. **This is the only route below 1 bit/weight found in the whole
  survey.** Author's own claim, unreplicated — but the mechanism is exactly our
  "generate, don't store" asymmetry pushed one step further than a PRNG seed, because a
  *learned* tile is cheap and still regenerates a large matrix.
- **ATtiny85-MNIST-RNN-EEPROM** — RNN weights in a **512-byte** internal EEPROM, author
  claims ~95% MNIST `[V]`. A hardware ceiling rather than an optimization target, but it
  is an existence proof at half a kilobyte.
- **702 parameters at 98.2% MNIST** (stats.stackexchange.com Q656420, Oct 2024) `[V]` —
  parameter count only, never converted to bytes. At fp32 that is ~2.8 KB, which is the
  whole point about bytes ≠ parameters.

**Coverage limit, stated plainly:** the search that established the empty cell exhausted
its web-search budget mid-sweep. **OpenReview full-text, Google Scholar, and GitHub code
search were not swept.** Before the empty-cell claim goes in print, those three are
where a stray result could still hide.

---

## 4. Model-compression leaderboards

### MicroNet Challenge (NeurIPS 2019) — the closest thing that ever existed, and it died `[V]`

<https://micronet-challenge.github.io/>

- **Score** = `params/baseline_params + math_ops/baseline_math_ops`, ranked ascending.
- **Bit-width rules, all verbatim-verified, and the interesting part:**
  - Params are **linear in bits/32** — *"an 8-bit parameter counts as 1/4th a
    parameter"*, so a 4-bit param is 1/8 of a param.
  - Math ops charge the **maximum input width**: a 4-bit weight × 8-bit activation
    costs 8/32, not 4/32.
  - **Additions never get the discount** — *"we do not allow 'freebie' quantization for
    addition operations"*. This is the clause that stops low-bit entries claiming a free
    32× on the accumulate side of a MAC.
  - The 16-bit freebie is **all-or-nothing**: *"If an entry quantizes any part of their
    model to less than 16-bits, then no 'free' quantization to 16-bits is allowed."*
  - Sparsity pays an honest index tax: nonzeros + a bitmask over the full tensor shape.
- **Baselines:** ImageNet — MobileNetV2 1.4×, ≥75% top-1. CIFAR-100 — WideResNet-28-10,
  ≥80% top-1. WikiText-103 — 1-layer LSTM, ppl <35.
- **Winners:** CIFAR-100 **RIAIRS (CAS), 0.0044** (30 entries); ImageNet **RIAIRSC
  (same team), 0.1295** (20 entries) — pruning → activation quantization → weight
  quantization, each with finetuning, plus distillation from EfficientNet
  (<https://github.com/wps712/MicroNetChallenge>). WikiText-103 **MIT-HAN-Lab, 0.0475**
  (arXiv:2005.07877). KAIST took CIFAR-100 2nd/3rd with lottery-ticket pruning.
- **Dates:** opened 1 June 2019, deadline 11 Oct 2019, results at NeurIPS 2019.
- **A 2020 edition was announced on stage and never happened** — Gale & Elsen gave a
  talk titled *"Updates and improvements for the 2020 MicroNet Challenge"*, but no 2020
  site, rules, or leaderboard exists. No successor found.
- **Bytes measured? No.** Params × bits/32 is *proportional* to storage but never a
  serialized file. No format, no compression, no checkpoint weighed.

**What we'd steal — the most directly transferable rule set in the survey.** The
bit-width-linear cost model with its two anti-gaming clauses. Its two fixable flaws are
exactly our design: it counted parameter storage rather than a self-contained artifact
(no decoder accounting at all), and it was one-shot rather than a standing ratchet.

### Blier & Ollivier, *The Description Length of Deep Learning Models* `[V]`

arXiv:1802.07044 · NeurIPS 2018. Directly load-bearing for our MDL track.

They measure **labels given inputs** — *"the number of bits necessary to send the labels
to someone who already has the inputs. This codelength includes the description length
of the model"* — i.e. `L = −Σ log₂ p(yᵢ|xᵢ)`. Explicitly *not* practical network
compression.

Table 1 (uniform baselines: MNIST 199 kbits, CIFAR-10 50,000·log₂10 = 166 kbits):

| Code | MNIST kbits | ratio | acc | CIFAR-10 kbits | ratio | acc |
|---|---:|---:|---:|---:|---:|---:|
| Uniform | 199 | 1.00 | 10% | 166 | 1.00 | 10% |
| float32 two-part | >8.6 Mb | >45 | 98.4% | >428 Mb | >2500 | 92.9% |
| Network compression | >400 | >2 | 98.4% | >14 Mb | >83 | 93.3% |
| Variational | 22.2 | 0.11 | 98.2% | 89.0 | 0.54 | 66.5% |
| **Prequential** | **4.10** | **0.02** | **99.5%** | **45.3** | **0.27** | **93.3%** |

Table 2 adds a *switch* code: CIFAR **34.6 kbits (ratio 0.21)**, fixing the catch-up
phenomenon — *"the VGGb model needs 5,000 samples on CIFAR to reach a cumulative
compression ratio < 1"*.

**A common misreading to avoid:** two-part codes are catastrophically bad (*"200 times
the uniform encoding on CIFAR10"*), but **variational is comfortably better than
uniform** — it just loses badly to prequential. The surprise is that a method built to
minimize this objective loses to a trivially simple online scheme.

**Citation hazard:** Table 1 gives variational MNIST as 22.2 kbits / 98.2%, while §3.3
and Table 2 both say 24.1 kbits / 95.5%. The inconsistency is in both the NeurIPS and
arXiv versions.

**Directly relevant to our numbers.** Their prequential CIFAR-10 figure of 45.3 kbits =
**5,663 B for 50,000 labels**, and the switch variant **4,325 B**. Our prequential track
in `cmix-and-prequential.md` reports 5,124 B warm for **10,000** labels. The comparison
is not like-for-like on stream length — but their catch-up finding is the direct
explanation of why our cold learner loses, and the switch code is the named fix.

### Bits-back and model-compression-as-coding `[V]`

Two literatures descending from one 1993 idea, and **nobody reports a single number
covering model + data**. That joint accounting is the gap.

**Data-side (model is free, not in the bitstream) — these measure real bytes.**
BB-ANS (arXiv:1901.04866): binarized MNIST 0.19 bits/dim (ELBO 0.19), full MNIST 1.41
(ELBO 1.39). Bit-Swap (arXiv:1905.06845): MNIST 1.29, **CIFAR-10 3.82**, ImageNet32
4.50. HiLLoC (arXiv:1912.09953): **CIFAR-10 3.56**, ImageNet32 4.20, full ImageNet 3.15
— trained only on ImageNet32 yet better on CIFAR-10 with no retraining. All write actual
ANS bitstreams and land within ~1–2% of their bounds.

**Model-side.** MIRACLE (Havasi et al., ICLR 2019, arXiv:1810.00440) transmits enough
bits for the receiver to *draw a sample* from `q_φ`, costing ≈ KL(q‖p):
LeNet-5/MNIST **1.52 kB at 0.96% error** (1110×) and **3.03 kB at 0.69%** (555×, which
*beats* the 0.7% uncompressed baseline — compression as regularization);
VGG-16/CIFAR-10 **384 kB at 6.57%** and **135 kB at 10.0%**.
Deep Compression (arXiv:1510.00149): prune → trained k-means weight sharing (32→5 bits)
→ Huffman; LeNet-5 1720 KB → **44 KB**, AlexNet 240 MB → 6.9 MB, VGG-16 552 MB →
11.3 MB. It charges CSR indices and Huffman-codes the index *differences*, and its
figure caption states *"the compression rate already included the meta-data for sparse
representation"* — better accounting than most, but still a bit budget, not a file.
**Deep Compression reports no CIFAR-10 at all.**

**Two hazards worth internalizing:**

1. **Measurement rigor is inversely correlated with headline ratio.** BB-ANS/Bit-Swap/
   HiLLoC write real streams and report modest, honest numbers. **Louizos et al.
   Bayesian Compression (arXiv:1705.08665) reports no bytes anywhere** — only sparsity
   percentages and compression *rates* — and posts 771×.
2. **Cross-paper numbers are not independent.** MIRACLE's "Bayesian Compression: 2.3 kB
   / 771×" row is Louizos's *ratio* divided into MIRACLE's own baseline. One arithmetic
   chain, three papers. Any frontier mixing these rows compares incompatible quantities.

A cautionary example of the failure mode: arXiv:1912.02254 abstracts *"reduce the size
of VGGNet by 9× from 20.04MB to 2.2MB"* on CIFAR-10 — but Table 1 shows those are
**parameter counts** (20.04M → 2.20M) relabeled as MB.

**What we'd steal:** MIRACLE's KL(q‖p) as a lower-bound reference line on the frontier;
and the standing habit of printing measured bytes beside any theoretical rate, which is
what `harness.md`'s round-trip-from-bytes rule already enforces.

---

## 5. Compression competitions — the only mature tradition that counts the decoder

This section is the one that matters most for our own rules.

### Hutter Prize `[V]`

<http://prize.hutter1.net/> (note: the site's TLS certificate is expired; fetched with
`curl -k`)

**The scoring rule, verbatim.** Primary form: publish `comp9.exe` producing a
self-extracting `archive9.exe` from `enwik9`; **`S := length(comp9.exe/zip) +
length(archive9.exe)`**. Relaxed form with a separate decompressor:
**`S := length(comp9a.exe/zip) + 2×length(decomp9.exe/zip) + length(archive9.bhm)`** —
the 2× drops to 1× if compressor and decompressor are the same binary.

This is **stronger than "count the decoder"**: it counts the *compressor* too, and
double-counts a standalone decompressor. The stated reason (FAQ `#addcomp`):

> If we were to limit only the compute time of the compressor C and not its size L(C),
> one could simply submit the archive A prefixed with a tiny program to output A
> verbatim, ignoring input enwik9. This formally satisfies the criteria of being a
> compressor, but with all the compute hidden in producing A in advance.

And on why the decompressor counts at all (`#adddecomp`): *"consider an extreme
'decompressor' of size 1GB that simply outputs enwik9 byte by byte (from a zero byte
archive), thus achieving a compressed size of 0."*

**Where the line is drawn.** *"Programs must run without input from other sources
(files, network, dictionaries, etc.) … Use of standard libraries as for file I/O are
allowed."* OS and standard libraries are free; anything data-heavy — dictionaries,
config, model weights — must be inside the counted bytes. Source-in-a-zip substitutes
for a binary ("C++, Python, and Assembler are accepted"). **Even command-line options
are charged**: *"If command-line options for execution or compilation are necessary,
their length is added to S."*

**Limits:** ≲50 hours single-core (formally 70,000/T hours by Geekbench5 score), ≤10 GB
RAM, ≤100 GB disk, **no GPU**. Two entries were rejected for exceeding memory, not size.

**Award:** `Z×(L−S)/L` with Z = €500,000, minimum 1% relative improvement (≈€5,000,
≈1 MB). 30-day public comment; OSI-licensed source required.

**Records (enwik9 era, since Feb 2020):**

| Author | Date | Decompressor | Total S |
|---|---|---|---:|
| **Kaido Orav & Byron Knoll** | **3 Sep 2024** | **fx2-cmix** | **110,793,128** |
| Kaido Orav | 2 Feb 2024 | fx-cmix | 112,578,322 |
| Saurabh Kumar | 16 Jul 2023 | fast cmix | 114,156,155 |
| Artemiy Margaritov | 31 May 2021 | starlit | 115,352,938 |
| Alexander Rhatushnyak | 4 Jul 2019 | phda9 v1.8 | 116,673,681 (pre-prize baseline) |

Next-record threshold: **<109,685,197**. A pending 9th record — **cmix-lex**, Ibrahim
Marcouch, announced 26 June 2026, a Linux SFX archive of **109,190,109 bytes**, verified
by James Bowery at 40.3h/9,993 MB within limits — had **not** been awarded as of the
fetch. Treat as pending. `[V]`

The enwik8-era table is worth noting for how it *reports*: it breaks out archive and
decompressor as separate columns (e.g. phda9 2017: 15,242,496 + 42,448 = 15,284,944).

### Large Text Compression Benchmark (Matt Mahoney) `[V]`

<https://mattmahoney.net/dc/text.html> — last updated **8 July 2026**.

> Compression programs will be ranked by the compressed size of enwik9 plus the size of
> a **zip archive (readable by unzip) containing the decompressor and any other files
> needed by the decompressor at run time (dictionaries, configuration files, .dll files
> not normally part of Windows, etc)**. The archive may contain either an executable
> program or source code in any general purpose programming language, **whichever is
> smaller**.

And the cleanest statement of where the line is, anywhere in the field:

> Compressors and decompressors do not have to be general purpose. They may be tuned
> specifically to this benchmark … However, **the test hardware, operating system,
> compiler, and programming language implementing the decompressor must be general
> purpose, available to the public, and not specifically designed to improve the ranking
> on this benchmark.**

Their decompressor column is *typed*: `x` = executable, `s` = source (if smaller),
`d` = separate from the compressor, and **0 for a self-extracting archive**, since the
decoder and payload are one file. No time or memory limit — which is why the neural
entries live here rather than at Hutter.

| Program | enwik9 | decompressor | Total | Alg | Added |
|---|---:|---:|---:|---|---|
| nncp v3.2 | 106,632,363 | 628,955 `xd` | **107,261,318** | Transformer | Oct 2023 |
| cmix v21 -t | 107,963,380 | 281,387 `sd` | **108,244,767** | CM | Sep 2024 |
| cmix-lex | 109,190,109 | 0 `xd` (SFX) | 109,190,109 | CM | Jun 2026 |
| fx2-cmix | 110,351,665 | 0 `xd` (SFX) | 110,351,665 | CM | Oct 2024 |
| jax-compress | 113,393,442 | 60,872 `sd` | — | LSTM/TPU | Mar 2026 |
| tensorflow-compress v4 | 113,542,413 | 55,283 `sd` | 113,597,696 | LSTM | Aug 2022 |
| zpaq 6.42 | 142,252,605 | 4,760 `sd` | 142,257,365 | CM | — |

**The ratio is the lesson.** At the top of this table the decompressor is **0.0–0.6% of
the score**. Counting it is a principled anti-cheat, not a live constraint — because the
payload is 1 GB. *Our situation is the inverse: in a 961-byte artifact the decoder is
most of the budget.* That inversion is what makes this project a different game from
Hutter, and it should be said plainly in any writeup.

Mahoney's rationale page has the one-sentence version: *"If the benchmark did not include
the decompressor, it would be possible to write a program to compress the data to an
empty file by writing a (very large) decompressor that stores a copy of the file."*

### Calgary Compression Challenge — the purest form, and the closest analogue in scale `[V]`

<https://mattmahoney.net/dc/calgary.html>. Broukhis 1996–2016, continued by Mahoney
without prize money.

> The goal of the contest is to produce the smallest possible archive containing
> **either** the 14 file Calgary corpus, **or a program that when run taking input from
> only other files in the archive (if any), outputs the 14 file Calgary corpus.**

Counting rule — and note it charges filenames: *"If submitting more than one file, then
the size of the archive is calculated as the sum of the file sizes, plus the lengths of
the file names, plus 4 bytes per file."* Six-hour limit on a Core i7 M620/4 GB.
Submissions must improve by **at least 1000 bytes**.

Record: **580,170 bytes, Alexander Rhatushnyak, 2 July 2010** — unbeaten for sixteen
years, on a 3,141,622-byte corpus. Here the decoder is a *material* fraction of the
score, which makes this the closest structural analogue to what we are doing.

### GDCC (Huawei) `[V]`

<https://globalcompetition.compression.ru/rules/>. `c_full_size = compressed-data size +
compressed-decompressor size`, and the normalization is unusually explicit:
**"We compress decompressors using bzip2 v.1.0.8 with the '-9' setting."** Four ~1 GB
sets, three time-limited categories each, €50,000 pool. The site documents the **2020**
edition (results 15 Dec 2020); whether later editions ran is **unverified**.

### Silesia and Squash — decoder *not* counted `[V]`

Mahoney's Silesia benchmark (updated 20 May 2026) ranks *"by total compressed size"*
with no decompressor column; leader paq8px_v215 at 27,825,511 bytes on a ~212 MB corpus.
Squash (<https://quixdb.github.io/squash-benchmark/>) measures ratio and speeds; codec
size is not a metric.

**The contrast is the point, and it is the cleanest justification for our rule:**
*every benchmark whose purpose is to measure modeling/inference counts the decoder;
every benchmark whose purpose is to select a shipping codec does not.* The dividing line
is whether the decoder's cost is amortizable over unbounded future data. It is not, for
a fixed corpus — and it is not for a CIFAR-10 classifier either.

### Neural compressors and the prequential trick — Hutter's own doctrine `[V]`

This is the citable authority for our prequential track, stated by the person who
designed the metric (FAQ `#largenn` / `#online`):

> Large Neural Networks can be trained to achieve excellent performance for text
> compression incl. enwik9, but are not competitive for the HKCP, since the networks
> often have (tens of) millions of (real-valued) parameters … On the other hand, the
> untrained NN usually has a simple description … **In this case it is possible to
> include only the smallish code and not the millions of trained weight values in the
> (de)compressor, provided the NN is trained sequentially or online (rather than in
> batch)** … Huge Transformer NN have achieved SOTA code-length on enwik8 and enwik9
> this way.

And on why not a train/test split (`#xvalid`): *"if the test set is taken from a public
source like Wikipedia, a gargantuan NN could be trained on all of Wikipedia or the whole
Internet. **Limiting the size of the decompression algorithm can prevent this. Indeed
this is the spirit of the used compression metric.**"*

**The worked example, from one author, is perfect.** Bellard's **nncp** is a 199M-
parameter Transformer whose *decompressor is 628,955 bytes* — the weights are not in
there, because it learns online. Bellard's **ts_zip** uses RWKV-169M quantized to 8
bits, reaching 1.084 bpb on enwik9 (better than anything on LTCB's enwik8 column) — but
that is **~169 MB of weights required at both ends**, larger than the 135 MB output it
produces. Under Hutter or LTCB accounting ts_zip would score worse than xz, and it is
**conspicuously absent from LTCB's ranked table**. Same lab, same author, two
compressors: one prequential and rankable, one pretrained and unrankable, and the
difference is entirely whether the parameters are inside the artifact.

cmix v21 (Byron Knoll, 10 Sep 2024) is the other instructive case: it ships a
**411,996-byte English dictionary**, and LTCB charges it — the ranked 281,387 `sd` is
the zipped *source*, chosen because it is smaller than executable+dictionary.

**What we'd steal:** all of it. This is the strongest external justification for
`harness.md`'s two most contested choices — counting `predict.py`, and forbidding the
artifact from reading the dataset.

---

## 6. Two more, briefly

**GGUF / llama.cpp quantization tiers** `[V]` — the one place mainstream ML already
headlines bytes, because params are fixed per base model so file size is the only
varying axis. Bits-per-weight: IQ1_S 1.56, IQ2_XXS 2.06, Q2_K 2.625, Q3_K 3.4375,
IQ4_XS 4.25, Q4_K 4.5, Q5_K 5.5, Q6_K 6.5625, Q8_0 8+scale. Real files
(TheBloke/Llama-2-7B-GGUF): Q2_K 2.83 GB … Q4_K_M 4.08 GB ("balanced — recommended") …
Q8_0 7.16 GB. **The sub-4-bit IQ tiers use importance-weighted non-uniform quantization
— the production-scale version of our finding that codebooks beat uniform grids below
5 bits.**

**"Smol" families headline params, universally** `[V]` — SmolLM3 (*"our 3B model"*),
TinyStories (*"below 10 million total parameters"*), Phi-3-mini (*"3.8B parameters"*).
None leads with bytes. The field measures *scale* in parameters and measures
*compression* in bytes, and the two registers never meet. That split is the gap a
bytes-first framing closes.

**ARC-AGI has no description-length component** `[V]` — the efficiency axis on
<https://arcprize.org/leaderboard> is **dollars**: cost-per-task, a `<$10,000` filter,
and a $50 compute budget for the Kaggle track. No size, program-length, or
Kolmogorov-complexity term appears on any fetched ARC page.

**One live competition may be an exception, and needs checking** `[V-1]` — a Kaggle
competition titled **"The 2026 NeuroGolf Championship"** exists (the title resolves; a
control fake slug 404s), reportedly asking for the smallest neural networks solving
ARC-AGI transformations as ONNX, scored on `max(1, 25 − ln(cost))` where
`cost = parameter count + memory footprint in bytes`, ≤1.44 MB per file, Apr–Jul 2026.
**Everything after the title comes from a single proxy render and Kaggle's SPA defeated
direct fetching.** Do not cite the formula without a browser check. If it holds, it is
the only live ML competition with bytes in the objective — and it is on ARC, not vision.

---

## (a) The table

Bytes = "is the ranked quantity the byte size of a self-contained artifact?"

| Project / benchmark | Optimizes | Bytes? | State |
|---|---|:--:|---|
| **Hutter Prize** | compressor + SFX archive bytes, ≤50h/10GB/no GPU | **✅ strongest** | fx2-cmix 110,793,128 B (Sep 2024); 9th pending |
| **LTCB** (Mahoney) | enwik9 + zip(decompressor + runtime files) | **✅** | nncp v3.2 107,261,318 B; updated Jul 2026 |
| **Calgary Challenge** | Σ files + Σ filenames + 4 B/file | **✅** | 580,170 B, Rhatushnyak, 2010; stale 16 yrs |
| **GDCC** (Huawei) | data + bzip2 -9 of decompressor | **✅** | 2020 edition; later editions unverified |
| **OpenAI parameter-golf** | BPB on FineWeb, subject to 16 MB of code+weights | **✅ (as constraint)** | 1.0565 BPB, May 2026 |
| **Demoscene / sizecoding** | opcode bytes of the binary | **✅** | live; 8 B–4 KB categories; zero ML |
| **codegolf.SE Q28207** (MNIST) | `bytes × (1200−correct)/1000` | **✅** | 101 B / 56.7%, 2014; 2 answers ever |
| **js1k** | ≤1024 B, server-enforced | **✅ (as cap)** | ran 2010–2019; one NN entry, 994 B |
| **Bonsai** (ICML 2017) | accuracy within a 2 KB model | **✅** | peer-reviewed sub-KB precedent |
| **TBNN** | bits/weight via a shared circulant tile | **✅** | 720 B / ~91% MNIST, unreplicated |
| **ATtiny85 MNIST RNN** | fits a 512 B EEPROM | **✅ (as cap)** | ~95% MNIST claimed; hardware ceiling |
| **NeuroGolf 2026** (Kaggle) | params + memory bytes, ARC-AGI | ⚠️ `[V-1]` | title verified, formula single-sourced |
| **Silesia** (Mahoney) | total compressed size | ❌ | paq8px_v215; updated May 2026 |
| **Squash** | ratio, comp/decomp speed | ❌ | dormant |
| **MicroNet Challenge** | params/baseline + ops/baseline, bit-width linear | ❌ (proportional) | CIFAR-100 0.0044; **dead after 2019** |
| **µNAS** | 4-way: acc, size, peak RAM, MACs | ❌ (int8 param count) | 86.49% @ 11.4 KB — **the bar** |
| **SpArSe** | ‖ω‖₀ and working memory | ❌ (nonzero count) | CIFAR results all **2-class** |
| **MCUNet** | accuracy under SRAM/Flash ceiling | ❌ (budget) | declines CIFAR; V3 86.9% is transfer |
| **MLPerf Tiny** | latency / energy at ≥85% accuracy floor | ❌ | v1.3, Sept 2025; CIFAR-10 is the IC task |
| **TFLite Micro / ARM ML-Zoo** | — (model zoo) | ❌ | ML-Zoo archived Jul 2025, no CIFAR |
| **Edge Impulse** | — (product) | ❌ | no leaderboard; claims self-contradict |
| **modded-nanogpt** | minutes to val loss 3.28 on 8×H100 | ❌ | 1.266 min, May 2026, record #86 |
| **cifar10-airbench** | seconds to 94% / 96% on one A100 | ❌ | 94.01% in 2.59s; successor 1.98s |
| **tinygrad** | token-aware line count | ❌ | CI-enforced at 25,000; began as 1,000 |
| **neuralgolfing.com** | parameter count | ❌ | CIFAR-10 page 404s; leaderboard empty |
| **ARC Prize** | accuracy under a dollar budget | ❌ | live; no size term |
| **BabyLM** | accuracy under a training-*data* budget | ❌ | live, EMNLP 2026; no model-size limit |
| **NeurIPS Edge-Device LLM** | accuracy under 12 GB peak memory | ❌ | params *"for information only, not ranking"* |
| **HF / Papers-With-Code** | accuracy | ❌ | **PWC is dead** — redirects to HF papers |
| **GGUF quant tiers** | file size vs quality, informally | ✅ in practice | live; no ranked leaderboard |

## (b) Is there a leaderboard to submit to?

**No. There is no live leaderboard anywhere that ranks CIFAR-10 by artifact size, and no
venue to submit to.** Concretely:

- **Papers With Code is dead** — the domain 302-redirects to `huggingface.co/papers/trending`;
  ~9,327 benchmark tables went with it. Its CIFAR-10 table tracked accuracy and
  parameter count, never bytes.
- **MLPerf Tiny is architecturally incompatible** — 85% is a qualifying gate and ranking
  is on latency/energy. A variable-size/variable-accuracy frontier has nowhere to go.
- **MicroNet was the closest fit and died after one edition** in 2019, with no
  successor.
- **neuralgolfing.com wanted to be this and never shipped** — its CIFAR-10 challenge
  page returns 404 with an empty leaderboard. That is evidence of demand, not of
  competition.
- **The HuggingFace leaderboards docs state the field's convention outright**: models
  should be compared *"in the same weight class (number of parameters)"*. Size is a
  control variable, never the score.

**Is the niche genuinely unoccupied? Yes — with two honest holes.**

What is occupied, and must be distinguished from in any writeup:

1. **Parameter-count golf is a real, live genre** (575-param MNIST at 98%; 61M-logic-gate
   CIFAR at 86.29%; codegolf.SE's ML corner). Our differentiator is that **bytes are not
   parameters**: quantization, entropy coding, and the code-versus-weights tradeoff are
   exactly what a parameter count cannot see.
2. **Byte-scored artifacts exist, on other tasks** — Hutter (text), OpenAI
   parameter-golf (language modeling), possibly NeuroGolf (ARC). The *rule* is not
   novel; its application to image classification is.
3. **The speedrun tradition** is our cultural template on the orthogonal axis.

The strongest evidence for the empty cell is the **MNIST negative**: a deeper, older,
easier-to-golf task with an active parameter-golf culture *still* has no
bytes-of-artifact tradition. If it existed anywhere, it would exist there.

The two holes, stated plainly: (i) the literature sweep exhausted its search budget
before covering **OpenReview full-text, Google Scholar, and GitHub code search**; (ii)
**Twitter/X and Reddit were not searchable at all** with available tooling. Neither
changes the conclusion, but both should be closed before "first ever" appears in print.

## (c) Three things this project should adopt

**1. A significance gate on new records — and note that all four serious contests have
one.**
OpenAI parameter-golf requires ≥0.005 nats at p<0.01; modded-nanogpt requires enough run
logs for p<0.01 on the loss target; Hutter requires 1% relative; Calgary requires 1,000
absolute bytes. `findings.md` already concedes a ~1 pp selection-noise fog and flags
sub-1.5 pp effects as unresolved. **Turn that caveat into a rule**: a Pareto point enters
the frontier only if it beats the incumbent by more than the binomial standard error at
n=10,000 (0.482 pp, so a ~1 pp threshold), *and* it was selected on a fixed validation
split with test touched once per method family. This is the already-filed fix; the
unanimity across four independent contests says it is standard practice, not pedantry.

**2. Report the decoder as a typed, separate line item — borrowed from LTCB and the
enwik8-era Hutter table.**
Both break out `archive` and `decompressor` as separate columns, with LTCB typing the
latter (`x`/`s`/`d`/`0`). At the top of LTCB the decoder is 0.0–0.6% of the score; in
our 961-byte artifact it is most of the budget. **That inversion is the single most
interesting structural fact about this project and the leaderboard currently hides it.**
Adding `code B` and `weights B` columns beside `size` would make visible which frontier
points are decoder-bound and which are weight-bound — and the ranked next step ("go
down, not up") is precisely a bet about the decoder-bound regime. Consider also GDCC's
normalization (charge the decoder at its bzip2 -9 size) as a *diagnostic* column: it
prices source by information content rather than character count.

**3. Make the quantizer aware of the code length it induces, instead of quantizing then
measuring.**
Two independent signals point here. `findings.md` already records the tension —
"codebooks destroy the redundancy the entropy coder was eating", better reconstruction
but worse residual compressibility — and currently treats it as a puzzle to note rather
than an objective to optimize. Meanwhile the two places in this survey that *did* couple
the two get real gains: Deep Compression Huffman-codes the index *differences* rather
than the indices, and llama.cpp's sub-4-bit IQ tiers use importance-weighted non-uniform
quantization, which is the production-scale version of our own "codebooks beat uniform
grids below 5 bits" finding. The concrete move is to select codebook centroids and
assignments against *bytes after entropy coding* rather than against reconstruction
error, then read accuracy off. This extends the existing negative result: reconstruction
error is not a proxy for accuracy, and it is not a proxy for size either.

A second, cheaper lever from the same observation: **every sub-KB method in the
literature projects to a low dimension first and then applies a tiny decision structure**
— Bonsai, ProtoNN, and SpArSe all do it, and it is why an ordinary linear model (30,730
params) cannot fit in 1 KB at all while their trees can. We already do this with random
conv features. The untried half is making the *projection* cheap-but-learned rather than
free-but-random, which is the same bet as the whitening stem.

Two further rules worth adopting as they come up, both cheap:

- **MicroNet's anti-gaming clauses** — additions always billed at 32-bit, the 16-bit
  freebie all-or-nothing — if a mixed-precision method ever lands on the board.
- **Hutter's charge-the-producer principle.** Hutter counts the *compressor* as well as
  the decompressor, closing the "precompute offline, ship a trivial replayer" hole. Our
  harness is partly defended already (the artifact cannot read the dataset, and is re-run
  from its own bytes in a clean process), but training is unmeasured. That is defensible
  for a deployment-oriented leaderboard and *not* for the MDL track, where the
  prequential program is the whole artifact. **The two tracks should state explicitly
  which of them charges the producer.**

One thing *not* to adopt: a single scalar score. Q28207 uses `bytes × error` and Moby
Dick uses `2·bytes + errors`; both produce one ranking and hide the tradeoff. Our
deliverable is a Pareto frontier, and airbench's tiered records (94% / 95% / 96%) are the
better shape — byte tiers with the best accuracy in each. The MDL track already supplies
the principled scalar where one is wanted.

---

## Sources

Speedruns and golf: [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) ·
[cifar10-airbench](https://github.com/KellerJordan/cifar10-airbench) ·
[hiverge/cifar10-speedrun](https://github.com/hiverge/cifar10-speedrun) ·
[openai/parameter-golf](https://github.com/openai/parameter-golf) ·
[tinygrad](https://github.com/tinygrad/tinygrad) ·
[codegolf.SE Q28207 (MNIST)](https://codegolf.stackexchange.com/questions/28207/recognize-handwritten-digits) ·
[Q187562 (ML golf: multiplication)](https://codegolf.stackexchange.com/questions/187562) ·
[Neural Golfing](https://hexhowells.com/posts/neural-golfing.html) ·
[Muon](https://kellerjordan.github.io/posts/muon/) ·
[airbench paper arXiv:2404.00498](https://arxiv.org/abs/2404.00498) ·
[hlb-CIFAR10](https://github.com/tysam-code/hlb-CIFAR10) ·
[teenygrad](https://github.com/tinygrad/teenygrad) ·
[sizecoding.org](https://www.sizecoding.org) ·
[js1k Retro Neural Network](https://js1k.com/2019-x/details/4099) ·
[TBNN](https://github.com/joaocarvalhoopen/TBNN__Tiled_Binary_Normalized_Neural_Networks_in_Odin) ·
[ATtiny85 MNIST](https://github.com/GiorgosXou/ATTiny85-MNIST-RNN-EEPROM)

TinyML: [MLPerf Tiny](https://mlcommons.org/benchmarks/inference-tiny/) ·
[arXiv:2106.07597](https://arxiv.org/abs/2106.07597) ·
[µNAS arXiv:2010.14246](https://arxiv.org/abs/2010.14246) ·
[uNAS repo](https://github.com/eliberis/uNAS) ·
[SpArSe arXiv:1905.12107](https://arxiv.org/abs/1905.12107) ·
[MCUNet arXiv:2007.10319](https://arxiv.org/abs/2007.10319) ·
[MCUNetV3 arXiv:2206.15472](https://arxiv.org/abs/2206.15472) ·
[TFLM arXiv:2010.08678](https://arxiv.org/abs/2010.08678) ·
[ARM ML-Zoo](https://github.com/ARM-software/ML-zoo) ·
[Müksch et al. arXiv:2005.04968](https://arxiv.org/abs/2005.04968) ·
[Saha et al. arXiv:2205.14550](https://arxiv.org/abs/2205.14550) ·
[Bonsai, ICML 2017](https://proceedings.mlr.press/v70/kumar17a/kumar17a.pdf)

Compression leaderboards and MDL:
[MicroNet Challenge](https://micronet-challenge.github.io/) ·
[scoring rules](https://micronet-challenge.github.io/scoring_and_submission.html) ·
[Blier & Ollivier arXiv:1802.07044](https://arxiv.org/abs/1802.07044) ·
[MIRACLE arXiv:1810.00440](https://arxiv.org/abs/1810.00440) ·
[Bayesian Compression arXiv:1705.08665](https://arxiv.org/abs/1705.08665) ·
[Deep Compression arXiv:1510.00149](https://arxiv.org/abs/1510.00149) ·
[BB-ANS arXiv:1901.04866](https://arxiv.org/abs/1901.04866) ·
[Bit-Swap arXiv:1905.06845](https://arxiv.org/abs/1905.06845) ·
[HiLLoC arXiv:1912.09953](https://arxiv.org/abs/1912.09953) ·
[Hinton & van Camp 1993](https://www.cs.toronto.edu/~fritz/absps/colt93.pdf)

Compression competitions: [Hutter Prize rules](http://prize.hutter1.net/hrules.htm) ·
[FAQ](http://prize.hutter1.net/hfaq.htm) ·
[LTCB](https://mattmahoney.net/dc/text.html) ·
[LTCB rules](https://mattmahoney.net/dc/textrules.html) ·
[Calgary Challenge](https://mattmahoney.net/dc/calgary.html) ·
[Silesia](https://mattmahoney.net/dc/silesia.html) ·
[Squash](https://quixdb.github.io/squash-benchmark/) ·
[GDCC rules](https://globalcompetition.compression.ru/rules/) ·
[nncp](https://bellard.org/nncp/) · [ts_zip](https://bellard.org/ts_zip/) ·
[cmix](https://www.byronknoll.com/cmix.html) ·
[gwern on the Hutter Prize](https://gwern.net/hutter-prize)

Other: [ARC Prize leaderboard](https://arcprize.org/leaderboard) ·
[GGUF docs](https://huggingface.co/docs/hub/en/gguf) ·
[HF leaderboards intro](https://huggingface.co/docs/leaderboards/en/leaderboards/intro) ·
[BabyLM](https://babylm.github.io/)
