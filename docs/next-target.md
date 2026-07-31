# The next target

**Recommendation: point the machinery at ARC-AGI, but not at the target the
previous memo proposed.** Do not enter ARC as a size-constrained solver
competition; instead use the harness and the arithmetic coder to *decompose ARC
itself* — measure how many of ARC's bits are generic 2D image structure and how
many are task semantics. A 783-byte zero-knowledge context coder already takes
the ARC-AGI-1 public evaluation set from 41,421 B to 8,129 B without solving a
single task, which means roughly 80% of what looks like "reasoning" on that
benchmark is compressible without any reasoning at all — and nobody has
published that number.

A note on how this memo was produced, because it changes how you should read it.
My first analysis concluded ARC was dead. I spawned a critic to attack that
conclusion; it reproduced my numbers, then falsified my central argument by
construction, and I verified its refutation by re-running its code myself. The
recommendation below is the one that survived being wrong. Where I was wrong I
say so, because the error is informative.

Every external number is marked VERIFIED with the URL fetched, or UNVERIFIED.
Numbers I measured on this box are marked MEASURED and the script is named.

---

## The arithmetic that organizes everything

A two-part MDL board is alive only if

    artifact_bytes  <  (H_baseline − H_model) × N / 8

So the first question for any domain is not "is it interesting" but "how many
bytes of prediction stream are there to amortize against." That single column
decides most candidates before any judgment is required.

| stream | raw bytes | note |
|---|---:|---|
| CIFAR-10 test labels (10,000 × log₂10) | 4,152 | the current board; MEASURED in findings.md |
| Speech Commands test (~11,000 × log₂12) | ~4,929 | MEASURED (arithmetic), counts UNVERIFIED |
| **ARC-AGI-1 public eval, naive** | **41,421** | MEASURED, `arc_stats.py` |
| ARC-AGI-1 public eval, after a free coder | 8,129 | MEASURED, `adaptive.py` (see below) |
| ImageNet-1k val (50,000 × log₂1000) | 62,286 | MEASURED (arithmetic) |
| OpenML CC-18, all 72 datasets | ~253,878 | MEASURED, from the OpenML API |
| ARC-AGI-1 entire corpus, weak coder | ~115,826 | critic's measurement, NOT independently reproduced |
| enwik8 | 100,000,000 | VERIFIED, https://mattmahoney.net/dc/text.html |

CIFAR-10's board was pinned because 4,152 B is a tiny budget *and* because
CIFAR-10 labels have essentially no structure to exploit without actually
classifying — they are near-uniform given no model. That second property is the
one that does not transfer, and missing it is what made my first analysis wrong.

---

## The survivors, ranked

### 1. ARC-AGI as an MDL decomposition — how much of ARC is reasoning?

**Headline if it works.** "Eighty percent of ARC-AGI-1's output entropy is
generic image structure. Reasoning is the last N kilobytes, and here is exactly
where it lives, per task." That is a claim about what the benchmark *measures*,
produced without solving it, and it is the kind of result that changes how a
field reads its own scoreboard. A second deliverable falls out for free: a
per-task bit-cost table that says which ARC tasks are informative and which are
compressible filler.

**Why this and not "ARC in minimal bytes."** The size-constrained-solver framing
is occupied and unpromising. CompressARC already frames ARC as "the code-golfing
problem: to find the shortest possible self-contained program that outputs the
entire ARC-AGI dataset" (VERIFIED,
https://iliao2345.github.io/blog_posts/arc_agi_without_pretraining/arc_agi_without_pretraining.html).
Ferré already uses L(M) + L(D|M) as an ARC search objective, scoring 24% on
training and 5.75% on evaluation (VERIFIED, https://arxiv.org/pdf/2311.00545).
And Chollet, who built the benchmark, argues against the framing directly: "we
do not expect that merely selecting the simplest possible program that works on
training pairs will generalize well" (VERIFIED,
https://arxiv.org/pdf/1911.01547v2). Entering as a competitor means fighting the
benchmark's author on his own turf with a metric he has already rejected.
Measuring the benchmark does not.

**The numbers that make it live.** MEASURED by me on a fresh clone of
https://github.com/fchollet/ARC-AGI (400 tasks, 419 test outputs, 98,515 cells):

| coder | eval test outputs |
|---|---:|
| naive uniform (log₂10/cell + dims) | 41,421 B |
| order-0 global colour model | 29,912 B |
| per-task free baseline (demo pairs only) | 13,998 B |
| adaptive context coder, 783 B of source | **8,129 B** |

The last row is the critic's `adaptive.py`; I re-ran it myself and reproduced
8,129 B at α=6.0. Its stripped implementation is 783 bytes raw, 442 gzipped —
both MEASURED by me. So an 800-byte artifact is MDL-positive against the naive
stream by ~32 KB while containing zero ARC knowledge. It is PPM with an
input-cell context.

That is simultaneously the reason the board is alive and the reason the
*interesting* question is decomposition rather than competition.

**What makes it fail.** If a stronger generic sequence model drives the eval
stream below ~4 KB, the semantic residue is too thin to build anything on and
the result collapses to "ARC outputs are compressible images" — still true,
still publishable, but much smaller. This is precisely what the de-risking
measurement below tests.

**Machinery port.** The arithmetic coder ports unchanged; it is already
integerized and round-trip verified, which is exactly what an adaptive per-cell
model needs. `artifact.py`'s serialization and size metric port unchanged. The
sandbox in `evaluate.py` needs its contract rewritten from `predict(x) -> labels`
to a coder interface, and the batch-consistency check becomes irrelevant (the
coder is inherently sequential). Estimate: the stated ~1 day is about right for
the harness, and the measurement scripts already exist in scratchpad.

**Compute.** Trivial. Every number in this memo was produced on the shared box in
minutes, CPU-only. This is the only candidate whose full pilot has *already run*.

---

### 2. A suite-scale prequential board (OpenML CC-18)

**Headline if it works.** "No published tabular model pays for its own
description length. Here is the smallest program that does, and it is a few
kilobytes." Or the negative: nothing does, and here is the crossover formula.

**Why it is live.** MEASURED by me from the OpenML API (study_99): 72 datasets,
874,726 instances, total label stream ~253,878 B under a crude order-0 model
(260,936 B at a uniform class prior). That is 60× CIFAR-10's budget. TabPFN v2's
checkpoint is 29 MB — reported by my tabular thread, NOT independently verified
by me — which would be ~114× over budget, making it a near-perfect foil.

**What makes it fail.** Three things, and they are why this is #2 not #1. First,
the idea is not new: Blier & Ollivier already established prequential coding as
the right tool and showed two-part codes are catastrophic. On CIFAR-10 (VERIFIED,
https://arxiv.org/abs/1802.07044): uniform baseline 166 kbits, variational
89.0 kbits, prequential 45.3 kbits, two-part with float32 parameters **>428 Mb** —
two-part coding is ~2,500× *worse* than sending the raw labels. On MNIST:
199 / 22.2 / 4.10 kbits, two-part >8.6 Mb. This is independent confirmation that
our prequential track is the right shape and that two-part framings fail, which
is useful — but it also means the headline finding is already published. The
contribution would be instantiation as byte-exact runnable artifacts rather than
asserted codelengths, which is real but narrower. Second, "tabular" is a misnomer for where the bits are: MEASURED, the
top four contributors are Devnagari-Script (63,521 B), mnist_784 (29,067 B),
Fashion-MNIST (29,067 B) and CIFAR_10 (24,914 B) — 146 KB of 248 KB is image data
in flat-vector form. Third, the deliverable is a benchmark, and a benchmark with
no striking first result is infrastructure, not research.

**Machinery port.** The coder and the prequential scorer port unchanged and are
currently underused. The artifact contract needs generalizing from images to
arbitrary feature matrices. `pack.py` ports unchanged.

**Compute.** CPU-native — this is the domain's home turf. Cheapest of the four.

---

### 3. Genomics: the two-part audit the field refuses to run

**Headline if it works.** "Ranking the DNA compression literature by total
description length instead of coded-stream size reverses the leaderboard."

**Why it is live.** VERIFIED by my exotic-domains thread: the field reports coded
stream only, no decompressor and no model — for the Sequence Compression
Benchmark, Jarvis, and XM. The one paper that does two-part accounting shows the
ranking flips: cmix wins on stream (988,958 B) and loses on total (1,282,852 B
including a 293,894 B decompressor) to PAQ8L at 1,238,330 B (VERIFIED,
https://arxiv.org/pdf/2012.12013). Baselines are solid: naive 2 bits/base, XM
1.6940 bpb (VERIFIED, https://allisons.org/ll/Publications/2007DCC/preprint.pdf),
and gzip is *worse* than naive at 2.150 bits/base (VERIFIED,
http://pizzachili.dcc.uchile.cl/texts.html). Headroom is unbounded.

**What makes it fail.** Infrastructure rot — the Sequence Compression Benchmark's
live host is dead (VERIFIED as unreachable), so you would be rebuilding a
benchmark as well as running it. And the audience is narrow: a correct result
that reorders a niche leaderboard is a smaller prize than a result about ARC.

**Machinery port.** Cleanest port of all four — it is natively a compression
task, so the coder and size metric apply directly. The image-specific parts of
the harness simply drop away.

**Compute.** Moderate; genome corpora are large but this is CPU work.

---

### 4. Joining the two axes nobody has joined: policy bytes

**Headline if it works.** "The first benchmark that scores a game-playing policy
on bytes and strength simultaneously."

**Why it is live.** This is the emptiest domain found, VERIFIED as an absence by
my exotic thread. Byte-golfed chess engines exist with real byte counts —
Toledo Atomchess 352 B (VERIFIED, https://github.com/nanochess/Toledo-Atomchess),
LeanChess 288 B (VERIFIED, https://leanchess.github.io/), BootChess 487 B
(VERIFIED, https://en.wikipedia.org/wiki/Olivier_Poudade) — but with *no playing
strength metric at all*, and several drop rules like castling to save bytes.
Meanwhile academic RL never reports bytes: the Koutník/Gomez/Schmidhuber
compressed-weight-space paper contains the strings "byte" and "KB" zero times
(VERIFIED by local text extraction of
https://people.idsia.ch/~juergen/gecco2010koutnik.pdf); WANN reports connections,
"Six Neurons" reports neurons. Nothing joins the axes.

**What makes it fail, and why it is #4.** The MDL scorer does not apply to
stochastic episodes — there is no fixed stream to code against — so the team's
distinctive second scoreboard is inert unless you reframe the task as *move
prediction over a corpus of games*, which restores a long codeable stream but is
a benchmark of your own invention. A metric nobody else accepts is a metric
nobody else competes on. Highest novelty, most new machinery, least
external validation.

---

## The graveyard

**ARC as a minimal-byte solver competition (the incumbent #1's actual framing).**
Dies on prior art and on solver economics, not on headroom. The framing is taken
(CompressARC, Ferré, above). And the solver economics are brutal: MEASURED from a
fresh clone of https://github.com/top-quarks/ARC-solution, icecuber's 2020 Kaggle
winner is 186,415 bytes of C++ source, 39,100 bytes xz'd, 7,180 lines — for 20%
on private eval (VERIFIED, https://arxiv.org/pdf/2412.04604). Against an 8,129 B
free-coder floor, the best-known pure program search is ~5× the entire remaining
budget. Per-task partial credit means a solver correct on fraction f saves ≈ f ×
8,129 B, so a 50%-accurate solver must fit in ~3.3 KB. Nothing close exists.

**A language model under 100 KB (incumbent #2).** Dies twice, and the second
reason is decisive.

*It imports a metric the field already has.* Three mature leaderboards use
exactly two-part accounting: the Hutter Prize scores S = S1 + S2 with the
compressor counted in full, record 110,793,128 B (Orav & Knoll, fx2-cmix, 3 Sep
2024) (VERIFIED, http://hutter1.net/prize/index.htm — note the TLS cert is
expired, reachable via `curl -k`); the Large Text Compression Benchmark adds "the
size of a zip archive containing the decompresser" (VERIFIED,
https://mattmahoney.net/dc/text.html); the Calgary Corpus Challenge, record
580,170 B (VERIFIED, https://mattmahoney.net/dc/calgary.html). Bringing a
description-length harness to text compression is bringing coal to Newcastle.

*The sub-100 KB cell is not empty — it is held, and held well.* On enwik8, with
decompressor size in bytes (all VERIFIED, https://mattmahoney.net/dc/text.html):

| compressor | bpc | decompressor |
|---|---:|---:|
| zpaq v6.42 | 1.429 | 4,760 B |
| phda9 v1.8 | 1.201 | 42,944 B |
| tensorflow-compress | 1.272 | 55,283 B |
| mcm 0.83 | 1.459 | 79,574 B |

A sub-100 KB neural artifact must beat ~1.20 bpc at 42.9 KB, or ~1.43 bpc at
4.8 KB at the small end. Classical context-mixing compressors already hold that
ground, and they are decades-optimized. For scale, the smallest published neural
model reporting enwik8 bpc is L3TC-200K at 1.404 bpc (VERIFIED,
https://arxiv.org/abs/2412.16642) — worse than a 4,760-byte zpaq. No published
neural LM with *stored* weights under ~100 KB reporting bpc was found, but that
is absence of evidence, explicitly not verified nonexistence.

*On the incumbent's premise.* OpenAI Parameter Golf does exist (VERIFIED,
https://github.com/openai/parameter-golf), but two details in the pitch need
correcting: the cap is 16,000,000 bytes **decimal**, and the metric is
bits-per-byte on a **FineWeb** validation set under a train-in-10-minutes-on-8×H100
constraint — not enwik8/9. Leaderboard top BPB 1.0565 against a 1.2244 baseline.
The "two orders below them" framing therefore compares against a different task
on different hardware, and the 8×H100 training constraint is itself outside this
team's budget. (I asserted Parameter Golf details in an earlier working note
before any source had been fetched — that was the exact failure this project's
rules exist to prevent, and it is recorded here rather than quietly corrected.)

**ImageNet in bytes (incumbent #3).** The incumbent dismissed this as "more
compute, not more ambition." Right conclusion, wrong reason: ImageNet's MDL
headroom is 62,286 B, 15× CIFAR-10's, so it is the *best*-conditioned vision
target on the arithmetic. It dies purely on compute — no standing GPU budget, and
this team just measured that its own compute is 6× off the field's at CIFAR
scale. Ambition is not the constraint; hardware is.

**Speech Commands / MLPerf Tiny.** Dies twice. The label stream is ~4,929 B —
CIFAR-10's disease in a new modality. And the domain already reports deployable
bytes throughout: Hello Edge gives DS-CNN-S 94.4% at 38.6 KB (VERIFIED,
https://ar5iv.labs.arxiv.org/html/1711.07128), MLPerf Tiny's KWS reference is
52.5 KB TFLite at 92.2% (VERIFIED, https://ar5iv.labs.arxiv.org/html/2106.07597),
and Arm's ML-zoo publishes exact byte counts. Ports in a day and teaches nothing.

**M4/M5 forecasting.** M4 is 100,000 series (VERIFIED by row count,
https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/M4-info.csv)
so headroom is fine, and no size axis exists in the scoring. It dies because
real-valued forecasts have no agreed quantization convention, so you would spend
the project defending the metric rather than producing a result. TSCom-Bench
(VERIFIED, https://arxiv.org/abs/2509.21002) scores coded data only and is a
framework paper, not a leaderboard.

**Symbolic regression.** SRBench already scores complexity — "the number of
mathematical operators, features and constants" — across 252 problems (VERIFIED,
https://arxiv.org/abs/2107.14351), and AI Feynman 2.0 already counts bits
(VERIFIED, https://arxiv.org/pdf/2006.10782). Converting node counts to bytes is
a re-parameterization, not a new measurement. Genuine gap, too small to be a
program.

**LLM contamination / memorization via description length.** The tool exists and
has never been aimed here — Delétang et al. compute that Chinchilla 70B on enwik9
goes from 8.3% to 14008.3% once parameters are counted (VERIFIED as a paper,
https://arxiv.org/abs/2309.10668; the exact digits are LOW-CONFIDENCE, single
fetch). Dies on access: no CPU-runnable frontier models and no byte-exact
artifacts for closed weights.

**Compressor-as-classifier (re-measuring the "gzip beats BERT" result under a
real harness).** Superficially the perfect target — a compression harness
auditing a compression-based classifier. It dies because the result is already
discredited and re-killing it adds nothing. The paper is "'Low-Resource' Text
Classification: A Parameter-Free Classification Method with Compressors" (ACL
Findings 2023, VERIFIED, https://aclanthology.org/2023.findings-acl.426). The
released code scores a k=2 prediction correct if *either* tied neighbour matches
— top-2 accuracy that peeks at the test label — and there is train/test
contamination (DengueFilipino train == test at 100%, KirundiNews ~90%). Corrected
k=1 numbers: Yahoo Answers 0.638 → 0.485, AG News 0.937 → 0.876 (VERIFIED). Worth
knowing defensively: if any future candidate leans on compressor-as-classifier
numbers, this is why not to trust them.

**OEIS / integer-sequence prediction by program length.** Dies on contamination
that cannot be measured — OEIS is in every pretraining corpus — and on a tiny
per-sequence stream. A benchmark whose leakage you cannot bound is not a
measurement.

**Turning the harness into a general MDL benchmark framework.** Not a research
target. A framework with no first result is infrastructure looking for a
customer.

---

## De-risking the top pick: one day, and most of it is done

House discipline is to measure the ceiling before building. For this target the
ceiling *is* the result, which is why it ranks first — the pilot has already run.
What remains is a single falsification test.

**The question:** how much of the 8,129 B residue is reachable without any ARC
knowledge? The project is only interesting if a meaningful semantic residue
survives a serious generic model.

**The measurement, in three steps, all CPU, all reusing scratchpad scripts:**

1. **Push the generic ceiling.** Run progressively stronger knowledge-free models
   on the eval outputs — higher-order context mixing, two-pass self-trained
   autoregressive coding as an upper bound on available generic structure. The
   critic's exploratory two-pass run took the 50 costliest tasks from 6,498 B to
   761 B, which I have *not* independently reproduced and which is the single
   most important number to check first.
2. **Price a real solver.** Take Hodel's 400 hand-written training-task solvers
   (VERIFIED to exist, https://github.com/michaelhodel/arc-dsl; note it publishes
   no eval solve rate at all, so any "arc-dsl scores X%" claim is UNVERIFIED) and
   measure what exact solutions actually save against the free-coder floor on the
   training stream.
3. **Report the split** as bits: generic structure vs. semantic residue.

**Kill criterion, stated before the measurement:** if a knowledge-free model
drives the ARC-AGI-1 public eval stream below **~4 KB**, abandon. At that point
the semantic residue is under 10% of the naive stream, solver credit of f × 4 KB
is unwinnable for any achievable f, and the honest write-up is a one-page
negative result rather than a program.

**Second kill criterion:** if step 2 shows that exact solutions on the training
stream save less than the free coder already saves, then solving ARC is not
where ARC's bits are, and the decomposition result is the *entire* project — take
the one-pager and move to survivor #2.

---

## Does the incumbent #1 survive?

**Yes — and my attempt to kill it failed on measurement, which is worth recording
in full because the failure is more useful than the verdict.**

I argued ARC's MDL board was structurally unwinnable, on three legs. Two broke.

*What broke, leg one.* I compared CIFAR-10's budget (raw uniform, 4,152 B) against
ARC's budget (best zero-knowledge coder, 13,998 B) — two different conventions.
Applied consistently, the CIFAR decoder also has the 50,000-image training set
free, and ARC is roomier than CIFAR by ~10× on raw terms rather than tighter.
Mixing conventions manufactured the asymmetry my conclusion rested on.

*What broke, leg two.* I claimed the bits sit in tasks nobody can solve, inferring
that "cheap under my coder" ≈ "easy for a solver" — a correlation I flagged as
unmeasured and then leaned on anyway. Measured against H-ARC human solve rates
and 73 scored ARC Prize model runs, Spearman is **−0.01** against human difficulty
and −0.27 against model difficulty (critic's measurement, sources VERIFIED to
exist: https://osf.io/bh8yq, https://huggingface.co/datasets/arcprize/arc_agi_v1_public_eval;
the correlation figures NOT independently reproduced by me). My 1,834 B figure was
near-tautological — "the cheapest half holds little of the cost" is true by
construction and says nothing about solvers. The defensible number is 3–4× larger.
**That figure should not be quoted anywhere.**

*What survived.* Concentration is real (top 10 tasks hold ~18% of bits under both
coders). The genuinely DSL-trivial tasks really are cheap, holding under 3% of
bits. And CIFAR-10's own diagnosis stands — near-uniform i.i.d. labels with no
free-lunch structure do pin that board. It simply does not transfer to a domain
whose outputs are structured images.

**What I would change about the framing.** Three things.

1. **Stop competing, start measuring.** "ARC in minimal bytes" fights an occupied
   framing and a hostile benchmark author. "How many of ARC's bits are reasoning?"
   is unoccupied, needs no solver, and yields a result either way.
2. **Fix the coding convention up front, in writing.** The free-lunch baseline is
   a moving target — it fell from 41,421 B to 8,129 B in one afternoon. Adopt
   prequential/online coding as the harness convention so the baseline is
   principled rather than negotiated, and so nobody can win by arguing about what
   the decoder gets for free. This is the single highest-risk governance decision
   in the project.
3. **Do not let the incumbent's stated risk — "all floor and no ceiling" — drive
   the decision.** It is backwards. ARC's floor is measurable today and its
   ceiling is irrelevant to the decomposition result, because the finding does not
   require solving anything.

One consequence worth stating plainly: on this reframing the first leaderboard
entries will be won by compression tricks with no reasoning content. That is not a
flaw to be engineered away. It *is* the measurement.
