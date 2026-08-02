# Does the frontier survive more classes? A half-day measurement

CIFAR-10 has one class count, so it cannot ask how description length scales
with the number of classes. A linear head costs `classes x features x bits`, so
the arithmetic predicts the sub-kilobyte region closes somewhere between ten
classes and a thousand and the whole frontier shifts right. Nobody has published
that curve. This memo measures it at K = 10, 100 and 1000.

Every number below is MEASURED on this box by a named script in this repo, or
VERIFIED against a URL that was actually fetched. Nothing is asserted from
memory, and no configuration was chosen on test.

**Recommendation: NO-GO as a project, and the measurement is the deliverable.**
Reasoning in the last section.

---

## Kill criterion 1 — license. DID NOT FIRE.

Downsampled ImageNet at 32x32 is obtainable without scraping, without
circumventing a gate, and without a mirror of unclear provenance.

The route used: **`benjamin-paine/imagenet-1k-32x32`** on HuggingFace, a Lanczos
repack of `ILSVRC/imagenet-1k` at exactly the Chrabaszcz et al. counts
(1,281,167 train / 50,000 validation). VERIFIED by me through the HF API:

    gated: False | license: ['other'] | license_details: imagenet-agreement

and the card reproduces the ImageNet Terms of Access in full. The operative
clause, quoted verbatim from
`https://huggingface.co/datasets/benjamin-paine/imagenet-1k-32x32/raw/main/README.md`:

> 1. Researcher shall use the Database only for non-commercial research and
> educational purposes.

That is exactly this use. Clause 6 binds a for-profit employer to the same
terms, which is worth knowing before any of this becomes commercial.

One honest wrinkle, stated rather than glossed: the card carries the terms under
`extra_gated_prompt` ("By clicking on 'Access repository' below, you also
agree…") but the maintainer did **not** enable the gate, so no click happens and
no consent is recorded. The substantive obligation is met either way. If any of
this is published, accept the terms once on `ILSVRC/imagenet-1k` (which is
`gated: "auto"` — self-serve, immediate) so that agreement is on record.

Two other routes exist and were not needed: the original Chrabaszcz files served
by image-net.org itself at `https://image-net.org/data/downsample/`, and the
self-serve gate on `ILSVRC/imagenet-1k`. The repo's `LICENSE` file is zero bytes
(MEASURED), so the card's `license_details` is the only license statement there
is.

Disk: 592 MB of ImageNet parquet in the HF cache, 136 MB for CIFAR-100, and a
1.69 GB working npz. Peak footprint about 2.4 GB, well inside the 10 GB budget.
The npz and the ImageNet cache were deleted afterwards; `df /home/rome` read
42 GB free before and 39 GB free after, the residue being other work on this
shared box rather than this measurement.

## Kill criterion 2 — chance-level accuracy at 1000 classes. DID NOT FIRE.

At 1000 classes chance is 0.10%. The family reaches **3.81% top-1 on 50,000
held-out ImageNet validation images** (MEASURED, `cs-K1000-k64-4b`, 270,200 B),
and even the 9,704 B floor scores 0.40%. With n = 50,000 the standard error at
3.81% is 0.086 pp, so this is 43 standard errors clear of chance. The frontier
at 1000 classes is not all floor.

It is, however, a frontier at 3.81%, which is the substance of the
recommendation rather than the kill criterion.

---

## The control this experiment was built around, verified

`experiments/label_stream.py`, MEASURED:

| stream | N | K | bits/label | bytes |
|---|---:|---:|---:|---:|
| CIFAR-10 test | 10,000 | 10 | 3.3219 | 4,152.41 |
| ImageNet-100 val, 50/class | 5,000 | 100 | 6.6439 | 4,152.41 |

The identity is structural, not a rounding coincidence:
`5000 x log2(100) = 5000 x 2 x log2(10) = 10000 x log2(10)`, equal to float
precision. The repo's quoted 4,152 B is the truncation of 4,152.41, not the
ceiling — worth pinning down, since the same figure appears in four documents.

So the 100-class leg can be run against a label stream *identical in size* to
the one the CIFAR-10 MDL board is scored on, with only the class count changed.
That leg is `cs-K100-*` on the ImageNet data, whose test set is exactly 5,000
images.

---

## Method

The family is the one this repo's sub-4 KB frontier is built on: random
convolutional features from a four-byte seed, average-pooled, into a ridge head
that is quantized and shipped. Only the head is real weight; the filters cost a
seed. `experiments/class_scaling.py` reuses `conv_features.py`'s feature source
verbatim, so trainer and artifact cannot drift.

Two things had to change for K > 10, and both are forced rather than chosen:

* **The head is K-wide**, so the decoder is emitted with K baked in.
* **The codebook is global, not per-column.** Per-column Lloyd-max costs
  `K x 2^b` float16 centroids: 160 B at ten classes, 32 KB at a thousand. Keeping
  it would have made the quantizer, not the head, the thing the curve measures.

Everything else is held fixed across K: same images, same 4x4 patches at stride
2, same 3x3 pooling, same seed, same lambda grid, same harness.

**Classes are nested.** Label `c` means the same class at every K, and the
K-class problem is exactly the images of classes `[0, K)`. The 10-class run sees
a strict subset of the 1000-class run's images.

**Fit set is 450 images per class and validation 50 per class at every K**, so
images-per-class is constant and only the class count moves. The consequence
worth stating: at K=10 the fit set is 4,500 images, a tenth of what the CIFAR-10
board trains on, which is why the K=10 accuracies here are well below the
board's. That confound depresses small-K accuracy and therefore makes the
"bytes buy less as K grows" reading *conservative*.

**Selection.** `(k, bits)` is the size axis and is *declared*: every point on it
is scored on test, so the curve is a measurement, not a best-of-N. Only lambda is
chosen, and it is chosen on the validation split, through the real quantized
artifact — float-head validation ranking disagrees badly with quantized ranking
at low bit widths, so selecting on the float head would pick the wrong lambda
for exactly the points that matter most. Test was scored once per declared
point.

**Harness.** Every artifact below passed `tinycifar/evaluate_nclass.py`, which
imports `evaluate._DRIVER` rather than copying it — same sandbox, same audit
hook, same batch-composition probe, same serialization and size metric. It was
cross-checked against the CIFAR path: re-scoring `artifacts/cf-k16-p4s2-4b-pc`
through it reproduces `results/cf-k16-p4s2-4b-pc.json` at 50.78% and 1,713 B
exactly. Four new tests in `tests/test_evaluate_nclass.py` pin the sandbox
behaviours; the suite goes 33 to 37 passing, with the CIFAR path untouched.

### Reproducing

```bash
python3 experiments/label_stream.py
python3 experiments/prep_dataset.py --source cifar100     --classes 100  --test-per-class 100 --out /tmp/c100.npz
python3 experiments/prep_dataset.py --source imagenet-hf  --classes 1000 --test-per-class 50  --out /tmp/in32.npz
python3 experiments/class_scaling.py --data /tmp/c100.npz --classes 10 100 \
    --k 8 16 32 64 --bits 1 2 3 4 --lam 10 100 1000 10000 --out /tmp/cs_c100
python3 experiments/class_scaling.py --data /tmp/in32.npz --classes 10 100 1000 \
    --k 8 16 32 64 --bits 1 2 3 4 --lam 10 100 1000 10000 --out /tmp/cs_in32
python3 experiments/class_scaling_report.py /tmp/cs_c100 /tmp/cs_in32
```

Nothing was written to `results/` or `LEADERBOARD.md`; these are not CIFAR-10
points and do not belong on that board.

---

## Result — the bytes-per-class curve

48 declared points on downsampled ImageNet, MEASURED,
`experiments/class_scaling.py`, tabulated by
`experiments/class_scaling_report.py`. `head B` is the arithmetic
`(dim+1) x K x bits / 8`; `bytes` is what the harness measured. `lift` is
accuracy divided by chance. `k` is the filter count, `b` the bit width.

| K | k | b | dim | bytes | head B | test | lift | B/class |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 8 | 1 | 72 | 734 | 91 | 25.80% | 2.6x | 73.4 |
| 10 | 8 | 3 | 72 | 944 | 274 | 43.20% | 4.3x | 94.4 |
| 10 | 32 | 4 | 288 | 2,119 | 1,445 | 46.00% | 4.6x | 211.9 |
| 10 | 64 | 4 | 576 | 3,543 | 2,885 | 47.80% | 4.8x | 354.3 |
| 100 | 8 | 1 | 72 | 1,524 | 912 | 2.88% | 2.9x | 15.2 |
| 100 | 16 | 2 | 144 | 3,888 | 3,625 | 8.12% | 8.1x | 38.9 |
| 100 | 32 | 4 | 288 | 14,377 | 14,450 | 12.34% | 12.3x | 143.8 |
| 100 | 64 | 4 | 576 | 28,000 | 28,850 | 13.60% | 13.6x | 280.0 |
| 1000 | 8 | 1 | 72 | 9,704 | 9,125 | 0.40% | 4.0x | 9.7 |
| 1000 | 8 | 3 | 72 | 26,179 | 27,375 | 1.56% | 15.6x | 26.2 |
| 1000 | 32 | 4 | 288 | 135,261 | 144,500 | 3.24% | 32.4x | 135.3 |
| 1000 | 64 | 4 | 576 | 270,200 | 288,500 | 3.81% | 38.1x | 270.2 |

Full 48-row table and the CIFAR-100 replication: run
`experiments/class_scaling_report.py` on the output directories.

### What the curve says

**1. Description length is `source + (dim+1) x K x bits / 8`, and the
measurement confirms it to within the compressor's margin.** Fitting the fixed
term at k=8, 1 bit gives 643 B at K=10, 612 B at K=100, 579 B at K=1000 —
constant to 10%, the drift being the entropy coder finding a little more
structure in a longer index array. At K=1000 and 4 bits the coder recovers about
6% against the arithmetic (270,200 measured against 288,500 predicted); at K=10
it recovers nothing. There is no regime where the head costs meaningfully less
than `classes x features x bits`.

**2. The sub-kilobyte band is closed at 1000 classes, and measurement agrees
with arithmetic.** The smallest artifact the family produced at K=1000 is
**9,704 B**, against 1,524 B at K=100 and 734 B at K=10. To get a linear
1000-class head under 1 KB at all you would need `(dim+1) x 1000 x 1 / 8 < 400`,
i.e. fewer than four features — so this is not a limitation of the sweep grid,
it is arithmetic. **The band this repo calls its most open frontier does not
exist at 1000 classes.**

**3. Accuracy per byte collapses as K grows, and it collapses faster than the
bytes grow.** At a fixed configuration (k=64, 4 bits) the artifact grows 3,543
to 28,000 to 270,200 B — a factor of 76 — while top-1 falls 47.80% to 13.60% to
3.81%. Lift over chance does rise (4.8x, 13.6x, 38.1x), so the model is doing
more work; it is just doing it at a byte cost that grows with the class count no
matter how good the features are.

**4. Low-bit quantization degrades non-monotonically at 1000 classes.** At 1 bit,
k=64 scores 0.54% against k=32's 1.03% — the wider head is *worse*, at twice the
bytes. Same inversion at 2 bits (1.10% against 2.04%). With n=50,000 those gaps
are 9 and 12 standard errors, so they are real, not noise. The global codebook is
being asked to serve 1000 columns with quite different scales; this is the same
failure the per-column codebook was invented to fix on CIFAR-10, and per-column
is unaffordable here (32 KB of centroids at 1000 classes and 4 bits). That is a
genuine open problem, and it is a problem about output-layer coding, not about
vision.

**5. The MDL board dies at 1000 classes.** Under a common conservative bound
(`price_labels` in the report script: code right/wrong, then which wrong class —
no fitted temperature, so nothing tunable), no artifact at any K pays for its
own transmission. At K=1000 the budget is 62,286 B and the cheapest total is
71,968 B. For a 9,704 B artifact to pay there it would need 23.5% top-1;
it gets 0.40%.

### The controlled comparison, on an identical label stream

This is the sharpest result here, and it is the one the 4,152 B identity was set
up to make possible. The ImageNet 100-class leg is scored on exactly 5,000 test
images, whose uniform label stream is **4,152 B — byte-for-byte the CIFAR-10
board's budget**. Same budget, same family, same harness, only the class count
changed. Priced under the same bound for both:

| board | artifact | accuracy | labels | total | vs 4,152 B budget |
|---|---:|---:|---:|---:|---|
| CIFAR-10, the repo's only point that pays under its own fitted coder | 961 B | 42.72% | 3,500 B | 4,461 B | over by 309 B |
| ImageNet-100, best total here | 1,524 B | 2.88% | 4,142 B | 5,666 B | over by 1,514 B |

(The CIFAR-10 row reads 4,461 B rather than `findings.md`'s 3,912 B because that
figure uses a temperature-fitted coder and this bound does not. Both rows use
this bound, so the comparison between them is sound; neither row should be
quoted against the fitted numbers in `findings.md`.)

Holding the transmission budget fixed and moving from 10 classes to 100 moves
the best two-part total from 7% over budget to 36% over. The class count, not
the image count and not the budget, is what kills it.

### Replication on CIFAR-100

The whole K=10-to-100 leg was also run on CIFAR-100, which is ungated, 136 MB (MEASURED),
and needs no ImageNet at all. It reproduces every structural claim: head bytes
scale exactly 10x with K (91 to 912, 721 to 7,212, 2,885 to 28,850), the floor
moves 744 B to 1,569 B, accuracy falls 64.20% to 26.51% at k=64/4 bits, and
nothing is MDL-positive at either K. CIFAR-100 is *easier* than 32x32 ImageNet
at matched K — 26.51% against 13.60% at 100 classes — which is expected and does
not affect any byte claim.

---

## Recommendation: NO-GO

Not because either kill criterion fired. Both passed. The reasons are what the
curve shows.

**The result is already in hand, and it is a memo, not a program.** The question
worth asking — how does description length scale with class count — took half a
day and has a clean answer: linearly in K with slope `(dim+1) x bits / 8`, a
fixed ~600 B of source, and no regime where a linear head escapes it. Standing up
ImageNet as a target would be spending three weeks to decorate a result that is
finished.

**At 1000 classes the size axis stops measuring what this repo is about.** Below
a kilobyte on CIFAR-10, two-thirds of the artifact is decoder source, which is
why golfing the emitted Python was worth 116 B and why the interesting work sits
there. At 1000 classes the source is 0.2% of a 270 KB artifact and the head is
everything. "Make the classifier smaller" degenerates into "make the output
layer smaller", which is low-rank factorization, hashing and hierarchical
softmax — a well-worked problem with a large existing literature, and not one
where this repo's harness is the differentiator.

**The band the repo is actually good at is arithmetically unreachable.** The
frontier's open ground is 500-700 B and the 1 KB region. Neither exists at 1000
classes for any linear head. Moving to ImageNet abandons the one part of the
board where this project has something nobody else has measured.

**The MDL board, the second scoreboard, dies outright.** On CIFAR-10 exactly one
artifact pays for its own transmission, which is thin but alive. At 1000 classes
nothing does and nothing plausibly will: closing the gap needs 23.5% top-1 in
under 10 KB, against 3.81% in 270 KB measured.

**And the class-count axis is available for free without ImageNet.** CIFAR-100
gives a full decade of K, ungated, at 136 MB and under fifteen minutes of CPU. It
reproduced every structural finding here. If the class-count question is ever
worth revisiting, that is where to ask it.

### What is worth taking from this

One genuinely open question fell out, and it is smaller and better posed than
"do ImageNet": **can a K-class head beat linear-in-K?** Finding 4 above says the
current quantizer breaks down exactly where it would matter — a global codebook
serving 1000 columns loses to a narrower head, while per-column codebooks cost
`K x 2^b` centroids and are unaffordable. Shared class codes, a factored head, or
a codebook indexed by class cluster would all attack this, and every one of them
can be measured at K=100 on CIFAR-100 in an afternoon. That is the interesting
descendant of this measurement, and it needs no license, no 1000-class corpus,
and no new compute.

---

## Provenance and hygiene

* Scripts, all in this repo, all re-runnable: `experiments/label_stream.py`,
  `experiments/prep_dataset.py`, `experiments/class_scaling.py`,
  `experiments/class_scaling_report.py`, `tinycifar/evaluate_nclass.py`,
  `tests/test_evaluate_nclass.py`.
* No number here was produced by anything not in that list. No external number
  is cited without the URL fetched, in the license section above.
* `results/` and `LEADERBOARD.md` were not touched; `docs/frontier.png` was not
  regenerated. These are not CIFAR-10 points.
* `python3 -m pytest tests -q`: 37 passed (33 before, plus 4 new for the
  K-class evaluator). `ruff check` clean on all new files.
* Disk: peak 2.4 GB, all of it dataset cache and one working npz. The npz and
  the ImageNet cache are deleted; the 64 selection artifacts are deleted and the
  48 scored ones (2.0 MB) are kept under `artifacts/class-scaling/` as the
  evidence behind the byte counts.
* Every heavy job ran under `nice -n 15` with `OMP_NUM_THREADS=3`, at most two
  at a time; box load stayed between 5 and 12 against 16 cores.

### Known weaknesses, stated rather than buried

* **K=10 is data-starved by construction.** Holding images-per-class fixed means
  the 10-class problem trains on 4,500 images, a tenth of the 100-class problem.
  That depresses small-K accuracy and makes the "bytes buy less as K grows"
  reading conservative — but it does mean the K=10 accuracies here are not
  comparable to the CIFAR-10 board's.
* **The K=10 test sets are small** (500 images on ImageNet, 1,000 on CIFAR-100),
  so SE is 1.5-2.2 pp and most differences within the K=10 block are unresolved
  against this repo's 1.06 pp noise floor. The K=100 and K=1000 blocks
  (5,000-50,000 images) are resolved to 0.1-0.5 pp.
* **k=64 is the top of the grid** and wins most K=100 and K=1000 rows on
  accuracy, so the accuracy ceiling at those class counts is not bracketed. It
  does not matter for any byte claim, which is the point of the memo, but no
  ceiling should be read off these tables.
* **The label pricing is a bound, not a fitted coder**, and therefore
  pessimistic for every row equally. A temperature-fitted coder would lower all
  MDL totals; at K=1000 it would have to lower them by 16% to change a verdict,
  which the CIFAR-10 gap between the two conventions (2.361 against 2.80
  bits/label) suggests is possible at high accuracy and not at 3.81%.

---

## Appendix — the full tables

Downsampled ImageNet 32x32, 48 declared points (`experiments/class_scaling_report.py`):

```
     K   k  b   dim     bytes   head B    test  chance   lift  B/class   MDL tot   budget  pays?
------------------------------------------------------------------------------------------------
    10   8  1    72       734       91  25.80%  10.00%   2.6x     73.4       932      208     no
    10  16  1   144       822      181  30.20%  10.00%   3.0x     82.2     1,016      208     no
    10  32  1   288     1,017      361  31.60%  10.00%   3.2x    101.7     1,209      208     no
    10  64  1   576     1,381      721  33.00%  10.00%   3.3x    138.1     1,571      208     no
    10   8  2    72       842      182  37.20%  10.00%   3.7x     84.2     1,026      208     no
    10  16  2   144     1,021      362  34.00%  10.00%   3.4x    102.1     1,210      208     no
    10  32  2   288     1,362      722  36.00%  10.00%   3.6x    136.2     1,548      208     no
    10  64  2   576     2,046    1,442  40.80%  10.00%   4.1x    204.6     2,224      208     no
    10   8  3    72       944      274  43.20%  10.00%   4.3x     94.4     1,118      208     no
    10  16  3   144     1,216      544  34.80%  10.00%   3.5x    121.6     1,403      208     no
    10  32  3   288     1,754    1,084  41.00%  10.00%   4.1x    175.4     1,932      208     no
    10  64  3   576     2,830    2,164  43.60%  10.00%   4.4x    283.0     3,003      208     no
    10   8  4    72     1,051      365  36.20%  10.00%   3.6x    105.1     1,236      208     no
    10  16  4   144     1,415      725  40.20%  10.00%   4.0x    141.5     1,594      208     no
    10  32  4   288     2,119    1,445  46.00%  10.00%   4.6x    211.9     2,288      208     no
    10  64  4   576     3,543    2,885  47.80%  10.00%   4.8x    354.3     3,709      208     no
   100   8  1    72     1,524      912   2.88%   1.00%   2.9x     15.2     5,666    4,152     no
   100  16  1   144     2,395    1,812   4.78%   1.00%   4.8x     23.9     6,513    4,152     no
   100  32  1   288     4,145    3,612   5.64%   1.00%   5.6x     41.5     8,250    4,152     no
   100  64  1   576     7,672    7,212   7.34%   1.00%   7.3x     76.7    11,748    4,152     no
   100   8  2    72     2,301    1,825   5.72%   1.00%   5.7x     23.0     6,405    4,152     no
   100  16  2   144     3,888    3,625   8.12%   1.00%   8.1x     38.9     7,949    4,152     no
   100  32  2   288     6,736    7,225   8.66%   1.00%   8.7x     67.4    10,786    4,152     no
   100  64  2   576    12,991   14,425   9.50%   1.00%   9.5x    129.9    17,024    4,152     no
   100   8  3    72     3,336    2,738   6.16%   1.00%   6.2x     33.4     7,433    4,152     no
   100  16  3   144     5,949    5,438   8.90%   1.00%   8.9x     59.5     9,994    4,152     no
   100  32  3   288    11,062   10,838  10.64%   1.00%  10.6x    110.6    15,070    4,152     no
   100  64  3   576    21,377   21,638  12.90%   1.00%  12.9x    213.8    25,333    4,152     no
   100   8  4    72     4,175    3,650   8.52%   1.00%   8.5x     41.8     8,228    4,152     no
   100  16  4   144     7,622    7,250  10.34%   1.00%  10.3x     76.2    11,637    4,152     no
   100  32  4   288    14,377   14,450  12.34%   1.00%  12.3x    143.8    18,346    4,152     no
   100  64  4   576    28,000   28,850  13.60%   1.00%  13.6x    280.0    31,938    4,152     no
  1000   8  1    72     9,704    9,125   0.40%   0.10%   4.0x      9.7    71,968   62,286     no
  1000  16  1   144    18,245   18,125   0.92%   0.10%   9.2x     18.2    80,420   62,286     no
  1000  32  1   288    35,944   36,125   1.03%   0.10%  10.3x     35.9    98,097   62,286     no
  1000  64  1   576    71,405   72,125   0.54%   0.10%   5.4x     71.4   133,648   62,286     no
  1000   8  2    72    16,458   18,250   1.00%   0.10%  10.0x     16.5    78,618   62,286     no
  1000  16  2   144    32,261   36,250   1.18%   0.10%  11.8x     32.3    94,381   62,286     no
  1000  32  2   288    63,146   72,250   2.04%   0.10%  20.4x     63.1   125,050   62,286     no
  1000  64  2   576   126,576  144,250   1.10%   0.10%  11.0x    126.6   188,715   62,286     no
  1000   8  3    72    26,179   27,375   1.56%   0.10%  15.6x     26.2    88,208   62,286     no
  1000  16  3   144    51,633   54,375   1.90%   0.10%  19.0x     51.6   113,576   62,286     no
  1000  32  3   288   101,295  108,375   2.87%   0.10%  28.7x    101.3   162,957   62,286     no
  1000  64  3   576   202,516  216,375   2.99%   0.10%  29.9x    202.5   264,143   62,286     no
  1000   8  4    72    34,916   36,500   1.93%   0.10%  19.3x     34.9    96,849   62,286     no
  1000  16  4   144    68,760   72,500   2.40%   0.10%  24.0x     68.8   130,563   62,286     no
  1000  32  4   288   135,261  144,500   3.24%   0.10%  32.4x    135.3   196,810   62,286     no
  1000  64  4   576   270,200  288,500   3.81%   0.10%  38.1x    270.2   331,564   62,286     no

smallest artifact per K, and what it scores:
  K=10    floor      734 B @ 25.80% (2.6x chance)   |  best 47.80% @ 3,543 B
  K=100   floor    1,524 B @  2.88% (2.9x chance)   |  best 13.60% @ 28,000 B
  K=1000  floor    9,704 B @  0.40% (4.0x chance)   |  best  3.81% @ 270,200 B
```

CIFAR-100 replication, 32 declared points:

```
     K   k  b   dim     bytes   head B    test  chance   lift  B/class   MDL tot   budget  pays?
------------------------------------------------------------------------------------------------
    10   8  1    72       744       91  37.90%  10.00%   3.8x     74.4     1,110      415     no
    10  16  1   144       843      181  34.80%  10.00%   3.5x     84.3     1,218      415     no
    10  32  1   288     1,022      361  43.20%  10.00%   4.3x    102.2     1,370      415     no
    10  64  1   576     1,380      721  43.20%  10.00%   4.3x    138.0     1,728      415     no
    10   8  2    72       839      182  37.80%  10.00%   3.8x     83.9     1,205      415     no
    10  16  2   144     1,022      362  42.90%  10.00%   4.3x    102.2     1,371      415     no
    10  32  2   288     1,365      722  46.10%  10.00%   4.6x    136.5     1,703      415     no
    10  64  2   576     2,064    1,442  51.40%  10.00%   5.1x    206.4     2,382      415     no
    10   8  3    72       943      274  45.80%  10.00%   4.6x     94.3     1,282      415     no
    10  16  3   144     1,219      544  47.30%  10.00%   4.7x    121.9     1,553      415     no
    10  32  3   288     1,758    1,084  58.90%  10.00%   5.9x    175.8     2,043      415     no
    10  64  3   576     2,836    2,164  61.70%  10.00%   6.2x    283.6     3,108      415     no
    10   8  4    72     1,051      365  47.30%  10.00%   4.7x    105.1     1,385      415     no
    10  16  4   144     1,408      725  54.30%  10.00%   5.4x    140.8     1,713      415     no
    10  32  4   288     2,117    1,445  59.40%  10.00%   5.9x    211.7     2,400      415     no
    10  64  4   576     3,523    2,885  64.20%  10.00%   6.4x    352.3     3,782      415     no
   100   8  1    72     1,569      912   7.06%   1.00%   7.1x     15.7     9,731    8,305     no
   100  16  1   144     2,455    1,812  10.04%   1.00%  10.0x     24.6    10,498    8,305     no
   100  32  1   288     4,197    3,612  13.11%   1.00%  13.1x     42.0    12,098    8,305     no
   100  64  1   576     7,792    7,212  13.97%   1.00%  14.0x     77.9    15,650    8,305     no
   100   8  2    72     2,370    1,825  12.88%   1.00%  12.9x     23.7    10,282    8,305     no
   100  16  2   144     4,029    3,625  14.42%   1.00%  14.4x     40.3    11,865    8,305     no
   100  32  2   288     7,182    7,225  13.35%   1.00%  13.4x     71.8    15,071    8,305     no
   100  64  2   576    13,368   14,425  17.33%   1.00%  17.3x    133.7    21,050    8,305     no
   100   8  3    72     3,380    2,738  14.30%   1.00%  14.3x     33.8    11,222    8,305     no
   100  16  3   144     5,999    5,438  16.25%   1.00%  16.2x     60.0    13,739    8,305     no
   100  32  3   288    11,114   10,838  18.16%   1.00%  18.2x    111.1    18,750    8,305     no
   100  64  3   576    21,572   21,638  22.98%   1.00%  23.0x    215.7    28,926    8,305     no
   100   8  4    72     4,254    3,650  15.03%   1.00%  15.0x     42.5    12,058    8,305     no
   100  16  4   144     7,731    7,250  18.98%   1.00%  19.0x     77.3    15,321    8,305     no
   100  32  4   288    14,551   14,450  24.00%   1.00%  24.0x    145.5    21,843    8,305     no
   100  64  4   576    28,356   28,850  26.51%   1.00%  26.5x    283.6    35,489    8,305     no

smallest artifact per K, and what it scores:
  K=10    floor      744 B @ 37.90% (3.8x chance)   |  best 64.20% @ 3,523 B
  K=100   floor    1,569 B @  7.06% (7.1x chance)   |  best 26.51% @ 28,356 B
```
