# cmix, context mixing, and the prequential reframing

What the Hutter-Prize toolkit (PAQ/lpaq/cmix: context mixing, adaptive arithmetic
coding, online learning) actually is, which parts transfer to this project, and the
MDL reframing that makes "size" and "accuracy" one number.

**Provenance.** Facts marked *[verified]* were checked against the primary source
during the writing of this doc: Mahoney's tech report CS-2005-16 (full text),
Blier & Ollivier 2018 (full text, arXiv:1802.07044v5), Byron Knoll's cmix page
(byronknoll.com/cmix.html), and the Wikipedia PAQ article for the PAQ7+ mixer
formulas. Facts marked *[recall]* are from memory and should be re-checked before
being load-bearing in anything published. Arithmetic marked *[derived]* I did here,
by hand; the steps are shown.

---

## Part 1 — What cmix/PAQ actually does

### 1.1 The shape of the machine

A context-mixing compressor codes the input **one bit at a time**. For each bit,
hundreds to thousands of *context models* each output a probability that the next
bit is 1. Each model is a table lookup: hash the current context (last N bytes, a
word, a column position, a sparse byte pattern — any function of the history) to a
counter state, and read off a probability. A *mixer* combines all these
probabilities into one, an arithmetic coder codes the actual bit against that
probability, and then **everything updates**: the counters in every context model,
the mixer weights, and the SSE tables. The decoder runs the identical models in
lockstep — it can, because every update depends only on already-decoded bits.

Scale, for calibration *[verified]*: cmix v21 uses **2,077 independent models**
(most inherited from paq8l, paq8pxd, and fxcm), a three-layer gated-linear-network
mixer trained by SGD on cross-entropy with an L2 regularizer, an LSTM, and an SSE
stage. It compresses enwik9 (10^9 bytes of Wikipedia) to 107,963,380 bytes in
~623,000 s using 30.9 GB of RAM. The Hutter Prize measures compressed size **plus
the decompressor** — the same accounting rule as our harness — with time/memory
caps that cmix itself exceeds; the prize-winning entries are trimmed
context-mixing variants *[recall]*.

lpaq1 (Mahoney, 2007) is the minimal member of the family worth studying: on the
order of seven models (byte orders 1–6 plus a match model and a word context), one
logistic mixer, two APM stages *[recall — read the lpaq1.cpp header before citing
counts]*. It gets a large fraction of PAQ8's compression at ~30x the speed. The
lesson for us: **the returns curve on model count is very concave.** A handful of
well-chosen predictors plus a good mixer plus calibration is most of the machine.

### 1.2 The mixer, precisely

Two generations, both online gradient descent on coding loss.

**PAQ1–PAQ6: linear evidence mixing** *[verified, Mahoney CS-2005-16]*. Each model
i outputs evidence counts (n0i, n1i) — a prediction plus a confidence. The mix is

```
S0 = eps + sum_i w_i * n0i          S1 = eps + sum_i w_i * n1i
p1 = S1 / (S0 + S1)
```

After coding bit x, weights move along the gradient of the coding cost
log2(1/p_x):

```
w_i <- max[0, w_i + (x - p1) * (S*n1i - S1*n_i) / (S0*S1)]     n_i = n0i + n1i
```

The term (x − p1) is the prediction error. Mahoney reports the rule is robust with
essentially one tuning parameter (eps).

**PAQ7 onward: logistic mixing** *[verified, Wikipedia PAQ; consistent with
Mahoney's "Data Compression Explained"]*. Each model now outputs a probability
p_i. Probabilities are mapped to the logit domain, mixed **linearly in logits**,
and mapped back:

```
stretch(p) = ln(p / (1-p))          squash(x) = 1 / (1 + e^-x)   (inverses)

t_i  = stretch(p_i)
p    = squash( sum_i w_i * t_i )
w_i <- w_i + eta * t_i * (y - p)         eta ~ 0.002 .. 0.01
```

This is exactly **online logistic regression** with the stretched model
predictions as features and the actual bit y as the target; the update is the
gradient of the log loss −log2 p_y. Three properties make it the right mixer:

1. Mixing in the logit domain lets one confident model dominate many uncertain
   ones (logits add; probabilities would average toward mush). A model at p=0.99
   contributes t=+4.6; five models at p=0.55 contribute t=+0.2 each.
2. The weights are trained on **the same loss the coder pays**, so the mixer is
   directly minimizing output bits, online, with no separate training phase.
3. Weights can go negative — a reliably *anti*-correlated model is useful.

**Gating** ("context-selected weight sets"): instead of one weight vector, keep
many, indexed by a small context (e.g. the high bits of the previous byte — PAQ4
used 8 weight sets selected by 3 bits *[verified]*). Different weightings win in
different regimes, and the switch is a table lookup, not a learned gate. cmix's
"gated linear network" mixer is this idea in layers: only a small subset of
neurons is active per prediction, chosen by manually defined contexts
*[verified]*.

### 1.3 SSE / APM

Secondary Symbol Estimation (introduced in PAQ2, 2003 *[verified]*) inserts a
learned **recalibration table** between the mixer and the coder. The table is
indexed by (small context, quantized input probability); it outputs a corrected
probability, and after the bit is seen the entry is nudged toward the outcome.
PAQ3 refined it: quantize the input probability to 32 levels (finer near 0 and 1
— i.e. roughly uniform in the stretch domain) and **interpolate linearly between
adjacent entries**, updating the two nearest entries *[verified]*.

Why it helps: the mixer is a linear model in logit space, so it cannot express
"when this predictor family says 0.9 in this byte-position, the truth is 0.97."
SSE is a cheap nonparametric correction of exactly that miscalibration, learned
online, conditioned on a little context. Chaining 2–5 APM stages with different
contexts, and averaging each stage's output with its input, is standard
(PAQAR ran mixers into per-mixer SSE stages, then 5 more parallel SSE units
*[verified]*).

The transferable content: **calibration is a first-class, learnable component,
separate from ranking.** Our ridge margins order classes well but are not
probabilities at all; an SSE-like map is what turns them into codable ones.

### 1.4 The arithmetic coder, and why adaptivity is the point

The coder itself is trivial and boring *[verified — both the CS-2005-16 version
and the ZPAQ carry-free version]*: keep an interval [low, high); split it at
`mid = low + p0*(high−low)`; take the half matching the bit; emit leading bytes as
they match. Total overhead over the Shannon bound sum(−log2 p) is at most a few
bytes per stream. rANS is a modern equivalent with the same bound.

The strength of the family is **not** the coder — it is that the probability fed
to the coder is produced by a model that **updates after every symbol**. Two
consequences matter for us:

1. **No probability table is ever transmitted.** The decoder starts from the same
   fixed initial state and reproduces every update. An adaptive coder has zero
   header cost; a static-table coder pays for its table. At our sizes a table is
   a real line item, so adaptivity is not a nicety, it is the difference between
   viable and not.
2. **The learning curve is the price.** Early symbols are coded at bad odds while
   the model warms up. Over 10^9 bytes of enwik9 the warm-up is noise; over 10^4
   weight indices or labels it is a visible fraction of the total. This is the
   regret term of Part 2, and it is the correct mental model for "how much does
   online adaptation cost."

### 1.5 cmix's LSTM

cmix adds a byte-level LSTM (also released standalone as lstm-compress) trained
**during compression** by truncated backpropagation through time, with Adam,
layer normalization, and learning-rate decay *[verified, byronknoll.com/cmix.html;
the online-during-compression part is structural — a decoder can only mirror a
model trained on already-decoded data]*. Its hidden state summarizes an
unboundedly long history, which fixed-order context tables cannot; its stretched
prediction feeds the mixer alongside everything else, and the mixer learns online
how much to trust it. It is a large reason cmix beats paq8 on text.

### 1.6 What transfers, and what does not

**To coding classifier weights (our current artifacts):**

- *Transfers:* the adaptive binary/range coder (zero table cost, ~30–60 lines of
  numpy); one or two **context models over the index stream** (e.g. condition on
  the other indices in the same feature's row across the 10 class columns, or on
  the neighboring tap in the same conv filter); logistic mixing of 2–3 such
  models if more than one helps. All decoder-side, all cheap in source bytes.
- *Does not transfer:* the scale. 2,077 models and gigabytes of state amortize
  over 10^9 highly redundant bytes. Our weight stream is ~10^4 near-i.i.d.
  symbols, and — the harness's own finding — k-means index streams are already
  near uniform entropy. There is no long-range structure for match models or an
  LSTM to find, and every line of decoder source is charged. The sliver that
  transfers is the smallest one.

**To coding labels (Part 2's stream):**

- *Transfers, almost whole:* the prequential stance (model updates online, decoder
  mirrors, nothing shipped), logistic mixing of several cheap predictors — the
  mixer is literally the same online-logistic-regression update, with class-logit
  features instead of stretched bit predictions — and SSE/APM as online
  calibration of scores into probabilities. Weight-set gating by a confidence
  bucket is exactly an APM.
- *Does not transfer:* bit-level decomposition (a 10-way symbol should be coded
  directly; splitting it into bits just obscures the distribution); context
  models over the **label sequence** (given the images, test labels are
  exchangeable — the CIFAR-10 test batch is 1,000 of each class in randomized
  order *[recall, Krizhevsky's dataset notes]* — so sequential structure in the
  label stream is file-ordering trivia, not signal, and exploiting it would be
  gaming the container, which this project's harness philosophy explicitly
  rejects); the LSTM (no sequential structure to summarize).
- One legitimate crumb of sequence-free structure: the known 1,000-per-class
  composition. Coding with sampling-without-replacement counts instead of a fixed
  uniform prior saves log2(10^10000·(1000!)^10/10000!) ≈ 55 bits ≈ 7 bytes over
  the whole test set *[derived, Stirling]*. Worth seven bytes of amusement, no
  more.

---

## Part 2 — The prequential / MDL reframing

### 2.1 A model is a code for the labels

In the Hutter/Solomonoff framing a model of data *is* a lossless code for it, and
compression is the measure of understanding. For classification the analogous
object is not a code for the images — it is a code for the **labels given the
images**. Both encoder and decoder hold the images; in our harness the dataset is
already declared free on both sides ("it is the signal being modeled"). The
question "how small is a model that classifies CIFAR-10 at accuracy p?" becomes
"**how few bits does it take to transmit the 10,000 test labels to someone who
has the images?**"

Formally (Shannon–Huffman; Blier & Ollivier 2018, Prop. 1 *[verified]*): if
encoder and decoder share a conditional model q, an arithmetic coder transmits
labels y_1..y_n at cost

```
L_q(y_1:n | x_1:n) = sum_i  -log2 q(y_i | x_i, everything decoded before i)
```

within one bit of optimal for the whole sequence. If the model must itself be
shipped, the total is a **two-part code**:

```
L_total = |decoder program + model artifact|  +  L_q(labels | images)
```

This is the point of the reframing: **size and accuracy land in the same unit.**
A bigger model is worth shipping iff it saves more label-bits than it costs in
artifact-bits. The Pareto frontier we currently maintain collapses to a scalar,
and the scalar is not arbitrary — it is the codelength Rissanen's MDL principle
says to minimize (Rissanen 1978), and the gain over the trivial code is bounded
by the mutual information I(y; x) (Blier & Ollivier eq. 2.4 *[verified]*), i.e.
by how much label-information the images actually carry.

### 2.2 Prequential coding: ship the learner, not the weights

The two-part code is not the only option, and per Blier & Ollivier it is by far
the worst one for learned models. The **prequential (predictive-sequential) code**
— the term and the philosophy are Dawid's (Dawid 1984 *[verified in B&O's
bibliography: JRSS A 147(2):278]*), the codelength theory Rissanen's (1984) —
works as follows:

- Encoder and decoder agree, in advance, on a deterministic **learning
  algorithm** and an initial model (say, uniform). Fixed PRNG seed if the
  algorithm is stochastic *[B&O make exactly this provision, verified]*.
- Step i: the current model q_{i-1} (trained on pairs 1..i−1) prices label y_i
  given x_i; the arithmetic coder codes y_i at −log2 q_{i-1}(y_i|x_i) bits; then
  **both sides** update the model on (x_i, y_i).
- Total: L_preq = sum_i −log2 q_{i-1}(y_i | x_i). **No weights are ever
  transmitted.** The only artifact is the learning-algorithm source.

Why this is a legitimate code and not an accounting trick: the decoder can run
the update at step i because it has, by induction, already decoded y_1..y_{i-1}
exactly. Encoder and decoder states never diverge, so the arithmetic decode is
exact. It is a valid one-part code for the labels — every bit is accounted for,
nothing is smuggled — and our harness philosophy (re-run the artifact from its
bytes in a fresh process) verifies it the same way it verifies a PRNG seed:
determinism is checked, not assumed.

What it costs instead of weights: **the learning curve.** Early labels are coded
by an ignorant model at ~log2 10 bits each; the price of not shipping parameters
is the area between the learning curve and its asymptote — the *regret*. The
difference L_preq − sum −log2 q_final(y_i|x_i) is precisely "the information the
trained parameters contain about the data" (B&O, §3.4 *[verified]*).

### 2.3 The arithmetic

Throughout: n = 10,000 test labels, K = 10, log2 10 = 3.3219.

**(a) No model (uniform).**

```
L_unif = 10,000 × 3.3219 = 33,219 bits = 4,152 bytes ≈ 4.06 KiB
```

This number should be pinned to the wall: **the entire CIFAR-10 test-label
stream, from nothing, is 4.15 KB.** Our current 9.5 KB artifact is more than
twice the size of the raw uncertainty it is meant to reduce. (For the 50,000
training labels: 166,096 bits = 20,762 bytes; B&O's "166 kbits" *[verified]*.)

**(b) A shipped model of accuracy p, ideally calibrated.** Suppose the model puts
probability c on its argmax and spreads 1−c uniformly over the other 9 classes.
Expected bits/label is minimized at c = p (calibration), giving

```
L(p)/n = H_b(p) + (1-p) log2 9        H_b = binary entropy
```

*[derived]*, with the flat-confusion assumption stated. Numbers:

| model | bits/label | 10,000 labels | vs uniform |
|---|---|---|---|
| p = 0.10 (chance) | 3.322 | 4,152 B | 1.00x |
| p = 0.4272 (our 961 B point) | 2.800 | 3,501 B | 0.84x |
| p = 0.6341 (our 9.5 KB point) | 2.107 | 2,634 B | 0.63x |
| p = 0.93 (B&O's VGGb) | 0.588 | 735 B | 0.18x |

*[derived: e.g. p=0.6341: H_b=0.9476, 0.3659×log2 9=1.1599, sum 2.1074;
×10,000 = 21,074 bits = 2,634 B]*

Two refinements. Real confusions are structured (cat↔dog), which lets a real
softmax beat the flat-confusion figure; real miscalibration costs bits, and can
cost unboundedly many (a confident wrong label at q→0 codes at −log2 q → ∞ —
this is why SSE/calibration is load-bearing, §3.4). A typical well-calibrated
63%-accuracy model has test cross-entropy around 1.5–1.8 bits/label *[recall —
measurable in one line from our saved margins; do that before quoting]*.

Note what the closed form says at the break-even boundary: L(p) < log2 10 for
every p above chance. **Labels-given-model always beat uniform; it is only the
shipped artifact that can fail to pay for itself.** The artifact budget a model
earns is n·(3.3219 − L(p)/n)/8 bytes:

```
p = 0.6341:  budget = 10,000 × 1.2145 / 8 = 1,518 B   — our artifact: 9,500 B  (6.3x over)
p = 0.4272:  budget = 10,000 × 0.5215 / 8 =   652 B   — our artifact:   961 B  (1.5x over)
```

*[derived]* In the strict MDL currency over the 10,000-label test stream,
**neither current frontier point pays for its own weights.** Two-part totals:
9,500+2,634 = 12,134 B and 961+3,501 = 4,461 B, both above the 4,152 B trivial
code. This is not an indictment of the artifacts — it is the statement that the
two framings answer different questions (§2.5) — but it is the number the Hutter
framing forces us to look at.

**(c) Prequential from scratch.** The classical yardstick is Rissanen's: for a
k-parameter model family (satisfying regularity conditions), predictive/two-part/
Bayes codes all achieve

```
L_preq  ≈  n·H(Y|X; best-in-class)  +  (k/2) log2 n  +  O(1)
```

— Rissanen 1984 showed predictive coding attains the (k/2) log n redundancy;
Rissanen 1986 proved no code can do better than (k/2−ε) log n for essentially all
parameter values (the lower bound); Rissanen 1996 sharpened the O(1) via the NML
/ Fisher-information integral. B&O restate the equivalence and its conditions
*[verified: their §4, "L = nH(Y|X) + (d/2) log2 n + O(1)"]*. Since this crew is
named for the man: the (k/2) log n is *his* result, in both directions, and it is
the whole content of "how much does not shipping the weights cost."

Instantiation for a plausible online learner here — multinomial logistic
regression on d = 129 random conv features (128 + bias), k = (K−1)·d = 1,161
free parameters:

```
regret ≈ (1161/2) × log2 10,000 = 580.5 × 13.288 = 7,713 bits ≈ 964 B
```

*[derived]* On top of the model's asymptotic ~2.1 bits/label this predicts
roughly 21,100 + 7,700 ≈ 28,800 bits ≈ 3.6 KB for the whole stream — under the
4.15 KB uniform bar, with a **total shipped artifact of a few hundred bytes of
learner source**. The asymptotic formula flatters small n (the early-phase
"catch-up" excess is real; Van Erven–Grünwald–De Rooij 2012 treat it and fix it
by switching between model copies *[verified in B&O's discussion and refs]*),
so treat 3.6 KB as an estimate to be measured, not a promise.

The empirical anchor that says the whole idea works at CIFAR scale *[verified,
B&O Table 1]*: on the 50,000 CIFAR training labels, uniform = 166 kbits;
best variational code = 89.0 kbits (model reaching 66.5% test acc); **prequential
with a VGG-like net = 45.3 kbits (compression ratio 0.27) at 93% test accuracy**
— and on MNIST prequential reached ratio 0.021. Prequential beat variational —
the method explicitly built to minimize a description-length bound — by 2x on
CIFAR and 6x on MNIST. Simple incremental coding wins; that is the paper's
headline, and it is the strongest published license for this crew's direction.

### 2.4 The catch, stated plainly

The two framings answer different questions. Do not let the elegance of one erase
the other.

- **The model-artifact framing (current harness)** answers: *how small can a
  frozen, deployable classifier be?* The artifact is the deliverable; inference
  is cheap and stateless; the 63.41% @ 9.5 KB point is a real object you can put
  on a microcontroller. The literature we compare against (µNAS et al.) lives
  here.
- **The prequential framing** answers: *how much regularity does this learning
  algorithm extract from this data stream?* Its number charges the learning
  curve and refunds the weights. The deliverable is a learner, not a classifier;
  there is no frozen model to hand anyone, and the decoder must re-run the whole
  learning process (fine for us — the harness already re-runs artifacts, and
  inference/compute time is declared free).

Neither dominates. Prequential is not "the artifact framing done right": by
Rissanen's equivalence its codelength is asymptotically the *same*
nH + (k/2)log n that an optimally quantized two-part code achieves — what it
removes is the **engineering loss** of explicit quantization (our codebooks,
grids, and gzip-vs-raw games), and what it adds is dependence on the learning
algorithm's sample efficiency and a strict determinism requirement. Conversely,
the artifact framing answers a deployment question prequential cannot, and its
numbers are comparable to a published literature. The right move is to run
**both columns**: the existing leaderboard, and a prequential-bits leaderboard
(|learner source| + coded-label bits, in bytes) on the same test stream. They
share almost all code and will disagree in instructive ways — e.g. width, the
wrong place to spend artifact bits, may be the *right* place to spend regret.

One more catch: with the training set declared free to both sides, a prequential
(or plain conditional) code may **pretrain on all 50,000 labeled training pairs
for zero cost** before coding a single test label. Then the learning-curve tax is
already paid and L ≈ n·(cross-entropy of the trained model): ~2.6 KB for a
63%-class model, plus only the learner source. Note what this implies for the
current harness too: a closed-form ridge solve is deterministic numpy — if an
artifact were allowed to read the training set at decode time, the entire weight
matrix becomes "generatable" and the artifact collapses to code. The current
contract (predict.py may read only sibling files) is what blocks this, and that
is a **design decision still open**, not a law of nature: the prequential
track makes it explicit — anything both sides hold is free; the only honest
charges are the learner's source and the labels' codelength.

---

## Part 3 — Ranked experiments for this repo

All CPU-friendly, numpy-only, no training runs needed except where marked; every
estimate falsifiable in an afternoon. Ranked by expected insight per effort.

### 3.1 (Rank 1) The prequential track: online learner sweeping the test set

**Mechanism.** New evaluator (~50 lines beside `tinycifar/evaluate.py`): feed the
artifact (x_i) sequentially, collect its distribution q_i over 10 classes, score
−log2 q_i(y_i), then reveal y_i to the artifact's `update(x_i, y_i)`. Report
`|serialized learner| + ceil(sum/8)` bytes. Two entries: (a) **from scratch** —
the pure Hutter number; (b) **pretrain-free** variant if we allow dataset
access at decode time. Learner: online multinomial logistic (or recursive
least-squares ridge, which is exact and hyperparameter-light) on the existing
seed-generated random conv features; the feature code is already written.

**Expected bits** *[derived, §2.3]*. Scratch: ~28,000–31,000 bits ≈ 3.5–3.9 KB
total vs 4,152 B uniform — a real but thin win, k/2·log2 n ≈ 7.7 kbit of it
regret. Pretrained: ~21,000 bits ≈ 2.6 KB + a few hundred bytes of source. For
comparison, the current 9.5 KB point *in this currency* costs 12.1 KB (§2.3b).

**Implementation sketch.** RLS ridge: maintain P = (X^T X + λI)^{-1} via
Sherman–Morrison per sample (d=129 → 129x129 matrix, trivial); one-vs-all
targets; softmax over margins with an online temperature (see 3.4). Deterministic,
seed-fixed, ~60 lines.

**Falsifies it.** Measured total ≥ 4,152 B from scratch (learning too slow at
n=10^4 — then try feature dim sweeps or label-efficient learners); or the
pretrained variant's measured cross-entropy far above the 2.1 b/label calibration
bound (would mean our margins carry less probability information than their
accuracy suggests).

### 3.2 (Rank 2) SSE/APM: online calibration of ridge margins

**Mechanism.** Ridge margins are not probabilities; the minimal APM is a **single
online temperature** τ: q = softmax(margins/τ), τ updated by one gradient step on
each revealed label. The full APM is PAQ3's: bucket the top-margin gap into ~16
interpolated bins, keep a per-bin correction updated online, optionally gated by
predicted class. Costs ~10–30 lines and ~0 shipped bytes (state lives decoder-side).

**Expected bits.** This is not an add-on but the hinge of 3.1: uncalibrated
margins can code *worse than uniform* (one confident miss at q=10^-4 costs 13.3
bits). Literature and PAQ practice both say calibration recovers on the order of
0.05–0.2 bits/label over a plain temperature *[recall-grade estimate]* — 60–250 B
over the stream, i.e. potentially the whole margin of victory in 3.1a.

**Falsifies it.** If the measured per-label loss with plain temperature already
sits within ~0.03 bits of the flat-confusion bound L(p)/n, the APM has nothing to
correct and should be dropped (its own regret, ~bins·(K/2)·log2 n bits, would
exceed its savings).

### 3.3 (Rank 3) Context mixing over label predictors

**Mechanism.** Run M cheap predictors in parallel over the test stream: class
prior (with counts — includes the 7-byte hypergeometric crumb), tiny linear model
on 8x8 pixels, the random-conv ridge, and (in the pretrain variant) a k-NN over
training features. Mix in the log domain — the K-class generalization of PAQ's
logistic mixing: q ∝ exp(sum_m w_m · log q_m), weights updated by online gradient
descent on the coding loss (Mattern analyzed exactly this "geometric mixing" and
its regret *[recall — Mattern, DCC 2013-era papers; verify before citing]*).
Gate the weight vector by a confidence bucket of the strongest model — that is
PAQ's weight-set gating, verbatim.

**Expected bits.** Mixing is nearly free in the only currency that matters: a
Bayes mixture pays ≤ log2 M ≈ 2 bits *total* versus the best single expert, and
learned geometric mixing pays O((M/2) log2 n) ≈ 27 bits at M=4 — so the mixture
can only lose trivially and wins wherever experts err differently (k-NN vs
linear confusions are known to be complementary). Realistic gain over the best
single model: 0.05–0.3 bits/label — 60–370 B — plus robustness: the mixer
auto-mutes a predictor that turns out miscalibrated, which de-risks 3.1.

**Falsifies it.** Mixer weights collapsing onto one expert (measured gain < 20
bits over best-single): the experts were not complementary; drop to M=1 and bank
the simplicity.

### 3.4 (Rank 4) Adaptive range coder + context model for weight indices

**Mechanism.** Replace gzip/xz on the existing quantized-index artifacts with an
adaptive rANS/range coder (no table shipped, §1.4) plus one context model:
predict index i from (same feature row, other class columns) — ridge rows are
correlated across the 10 classes — and/or spatial neighbors within a conv
filter; optionally a 2-model logistic mix.

**Expected bits.** Bounded and probably thin. The harness already found codebook
indices near-uniform (gzip ≈ raw) — order-0 adaptive coding therefore saves
≈ nothing, and all hope rests on *conditional* structure. If row-context removes
0.1–0.5 bits per 6-bit index on ~12,000 indices, that is 150–750 B — against a
600–800 B decoder (survey's own estimate) plus the survey's warning that at 1 KB
scale the decoder is 15–80% of budget. Net at 9.5 KB: −650 B to +150 B. **Run the
measurement, not the coder**: compute H(index | context) from the artifact in a
20-line script; only if it comes in ≤ 5.5 bits (vs 6) is the coder worth writing.
The affine-quantized (non-codebook) artifacts are the better target — they retain
the skew gzip was eating, and beating gzip's 1.3–1.6x-over-entropy overhead there
is exactly what rANS is for; same falsifier applies.

**Falsifies it.** Measured conditional entropy within 0.05 bits of marginal
entropy — then the indices are conditionally uniform too, the idea is dead at any
decoder price, and the finding ("ridge class-columns are informationally
independent after k-means") is itself worth a line in findings.md.

### Sequencing

3.1 and 3.2 are one experiment (calibration is 3.1's error bar), and they
motivate the new evaluator; 3.3 rides on the same loop for one extra afternoon.
3.4 is independent and starts with a measurement, not code. Before any of it:
one line of numpy to compute the actual test cross-entropy of the existing 9.5 KB
model's margins under a fitted temperature — it turns §2.3(b)'s estimate into a
measured number and prices every design above.

---

## Sources

- M. Mahoney, *Adaptive Weighing of Context Models for Lossless Data
  Compression*, Florida Tech TR CS-2005-16, 2005. *[verified, full text]*
- Wikipedia, *PAQ* — PAQ7+ logistic mixing formulas. *[verified]*
- B. Knoll, cmix (byronknoll.com/cmix.html; github.com/byronknoll/cmix,
  /lstm-compress). *[verified: model count, mixer, LSTM, SSE, enwik results]*
- M. Mahoney, *Data Compression Explained* (mattmahoney.net/dc/dce.html) — coder
  detail *[verified]*; mixer/SSE sections *[recall, fetch truncated]*.
- L. Blier & Y. Ollivier, *The Description Length of Deep Learning Models*,
  NeurIPS 2018, arXiv:1802.07044. *[verified, full text incl. Table 1]*
- A. P. Dawid, *Statistical Theory: The Prequential Approach*, JRSS A 147(2):278,
  1984. *[citation verified via B&O bibliography; not fetched]*
- J. Rissanen: *Modeling by Shortest Data Description*, Automatica 14:465–471,
  1978; *Universal Coding, Information, Prediction, and Estimation*, IEEE T-IT
  30(4):629–636, 1984; *Stochastic Complexity and Modeling*, Ann. Statist. 14(3),
  1986; *Fisher Information and Stochastic Complexity*, IEEE T-IT 42(1), 1996.
  *[recall — standard citations, venues/years believed exact, not fetched]*
- T. van Erven, P. Grünwald, S. de Rooij, *Catching Up Faster by Switching
  Sooner*, JRSS B 74(3):361–417, 2012. *[citation verified via B&O]*
- C. Mattern, geometric/logistic mixture analyses, DCC ~2012–2013. *[recall —
  verify before citing in anything public]*
- Knoll & de Freitas, *A Machine Learning Perspective on Predictive Coding with
  PAQ8*, DCC 2012, arXiv:1108.3298. *[recall, not fetched]*
