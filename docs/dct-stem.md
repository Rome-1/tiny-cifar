# Generated input transforms: what lever C is worth on 32x32 images

[what-to-try.md](what-to-try.md) item 8 named a **zero-byte analytic DCT stem** as
untested, with no published CIFAR-10 result.
The Speech Commands de-risk ([kws-derisk.md](kws-derisk.md)) made it worth testing now:
there, a complete log-mel front end cost **99 B** and stored nothing, because
`np.fft.rfft` lives inside the declared runtime and mel spacing is a formula,
while shipping the equivalent filterbank as weights cost **1,119 B**.
That is a ~1,020 B swing on lever C, and CIFAR-10 had never used it.

Every number below is produced by `experiments/dct_stem.py`, on this box.
Configurations are selected on the validation split from `tinycifar.data.load_dev()`.
Test is scored once per declared artifact through `tinycifar/evaluate.py`.

```bash
python3 experiments/dct_stem.py cost      # the transform-cost table
python3 experiments/dct_stem.py sweep     # matched-bytes comparison, on val
python3 experiments/dct_stem.py golf --names <stem>:<bits>:<lam> ...
python3 experiments/dct_stem.py final --names <stem>:<bits>:<lam> ...
python3 experiments/dct_stem.py mdl   --names <stem>:<bits>:<lam> ...
```

---

## The falsifier did not fire, and the reason is not the one item 8 predicted

*Declared falsifier: if a generated transform stem fails to beat the raw 8x8 RGB
block-mean input at matched bytes by more than the 1.06 pp selection noise, the
transform buys nothing on 32x32 images and item 8 should be struck.*

It beats it, in every byte band from 500 B up, by **+4.1 to +7.0 pp on validation**.
So item 8 is not struck.

But the mechanism is not what item 8 says it is, and the distinction decides what
to do next. **A change of basis buys nothing. The magnitude does.**

At 192 dimensions, with an identical decoder, an identical quantizer and 90
configurations swept per arm — so the selection pressure is the same on every row:

| stem, all 192-d | best val | vs `rgb8` | what it is |
|---|---:|---:|---|
| `rgb8` | 41.18% | — | 8x8 RGB block mean, the incumbent free input |
| `dct8k2` | 41.52% | +0.34 | 2x2 lowest DCT coefficients of 8x8 blocks |
| `had8k2` | 40.52% | −0.66 | the same, Walsh-Hadamard, trig-free |
| `dctabs8k2` | 45.30% | **+4.12** | `dct8k2` with `abs` on the coefficients |
| `hadabs8k2` | 44.64% | **+3.46** | `had8k2` with `abs` — no trig anywhere |
| `fftabs8k2` | **47.70%** | **+6.52** | modulus of the 2x2 lowest `rfft2` bins |
| `fftlog8k2` | 40.00% | −1.18 | the same, with `log1p` |
| `dctfft8k2` | 38.62% | −2.56 | truncated DCT magnitude routed through `np.fft` |
| `pw4r3` | 36.32% | −4.86 | patch-whitening basis, **shipped as bytes** |
| `rgb8r` | 29.74% | −11.44 | block means rectified about the image mean |

Read the first three rows against the last seven.

`dct8k2` is not merely close to `rgb8` — it is an exact orthogonal re-encoding of it.
`rgb8` is an 8x8 grid of 4x4-block means; `dct8k2` keeps the 2x2 lowest DCT
coefficients of each 8x8 block, which is a rotation of each group of four adjacent
4x4-block means.
A linear head can invert a rotation, so the two arms must agree, and they do: +0.34 pp,
a third of the noise floor.
The same holds on test, where the two artifacts were declared separately:
**39.63% for `rgb8` at 1,904 B and 39.28% for `dct8k2` at 1,952 B.**

Everything above the noise floor comes from `abs`.
`dctabs8k2` differs from `dct8k2` by an `abs(...)` wrapper — 3 source bytes, 2 artifact
bytes — and it is worth +3.78 pp.
That is not a property of the DCT: the trig-free Walsh version gets +4.12 pp for the
same three bytes, and the two land within a point of each other.

The best arm goes further than either, and for a reason worth naming.
`fftabs8k2` takes the **modulus of a complex Fourier bin**, not the absolute value of
a real cosine coefficient.
The modulus of bin (1,1) combines the cosine and sine parts of one 2-D frequency into
a phase-invariant energy; `|DCT coefficient|` rectifies the cosine part alone and
throws the sine part away.
That is worth a further +2.40 pp over `dctabs8k2` at 34 *fewer* artifact bytes.

The control that rules out "any nonlinearity helps" is the last row.
`rgb8r` applies a rectifier to the block means with no transform under it, and it
**loses 11.44 pp** — rectifying about a global mean destroys information rather than
exposing any.
The rectifier only pays when there is a local high-frequency coefficient to rectify.

### What that does to item 8

Item 8 should be **rewritten, not struck**.
Its generated-vs-stored claim is confirmed and is large.
Its specific proposal — an analytic DCT "in place of a learned basis" — is a no-op as
stated, because for a linear head any orthogonal basis is the same basis.
The thing that works is a *generated block transform followed by a magnitude*, and the
cheapest one is `abs(np.fft.rfft2(blocks))`, not a DCT.

---

## What a transform costs on its own

Measured first, as the KWS work did, before any classifier exists.
Each row is a complete runnable artifact whose classifier is deliberately degenerate
(`v.argmax(1)%10`), so the byte count is the floor the front end imposes.
`delta` subtracts the `null` row — the same artifact with the transform removed — and
so isolates the transform from the fixed cost of being a Python artifact.
Every row was run through the harness sandbox on real images, so a fragment that does
not execute cannot post a byte count.

| stem | bytes | delta | src B | coef B | dims | stored? |
|---|---:|---:|---:|---:|---:|---|
| `null` | 85 | +0 | 70 | 0 | 3 | no |
| `rgb8` / `rgb4` / `rgb16` | 120 | **+35** | 113–115 | 0 | 192 / 48 / 768 | no |
| `fftabs8k2` | **145** | **+60** | 144 | 0 | 192 | no |
| `fftabs16k2` | 146 | +61 | 146 | 0 | 48 | no |
| `fftlog8k2` | 151 | +66 | 154 | 0 | 192 | no |
| `dctfft8k2` | 158 | +73 | 162 | 0 | 192 | no |
| `rgb8r` | 161 | +76 | 176 | 0 | 192 | no |
| `had4k2` | 176 | +91 | 182 | 0 | 768 | no |
| `dct8k2` (and every other `dct*`) | 177 | +92 | 192 | 0 | 192 | no |
| `dctabs8k2` | 179 | +94 | 195 | 0 | 192 | no |
| `had8k2` | 179 | +94 | 193 | 0 | 192 | no |
| `hadabs8k2` | 181 | +96 | 196 | 0 | 192 | no |
| `pw2r3` | 297 | +212 | 260 | 72 | 768 | **YES** |
| `pw4r1` | 311 | +226 | 257 | 96 | 64 | **YES** |
| `pw4r3q` | 370 | **+285** | 267 | **144** | 192 | **YES** |
| `pw4r3` | 497 | **+412** | 258 | 288 | 192 | **YES** |

Four things this settles.

**A complete generated transform costs 60 to 98 bytes and stores nothing.**
That is the CIFAR-10 answer to the question KWS asked, and it is the same order of
magnitude: 99 B there, 60 B here.

**The stored basis is the expensive one, and it is the same object.**
`pw4r3q` is item 8's own "~144 byte patch-whitening stem" — 4x4 patches, PCA-whitened,
truncated to 3 components, 192 dims, the basis shipped as 48x3 int8, which is exactly
144 B of coefficient.
`dct8k2` produces the same 192 dims from comparable local support — 2x2 coefficients
of each 8x8 block — and ships nothing at all.
**370 B against 177 B: lever C is worth 193 B here**, and 320 B against the float16
version of the same basis.
Smaller than KWS's 1,020 B, and for the reason kws-derisk predicted — the value of
lever C scales with how much structure the transform has, and a 2-D block transform
has less of it than a 20x251 mel filterbank.

**`np.fft` is inside the declared runtime and this repo had never spent it.**
Confirmed against `tinycifar/artifact.py`'s `ALLOWED_IMPORTS`: `numpy` is declared, and
`np.fft.rfft2` is attribute access on it, not an import.
All eight `fft*` artifacts pass `check_imports` and run in the sandbox.
It is also the *cheapest* route to a block transform in the table — 60 B, against 92 B
for a generated cosine matrix — because numpy owns the twiddle factors and the
`axes=` argument removes the transpose the matrix form needs.

**The trig-free option is not the cheap one, again.**
kws-derisk found Haar *more* expensive than the FFT and warned against assuming
otherwise.
The same holds here: `had8k2` is 179 B against `fftabs8k2`'s 145 B, because
`np.kron` twice plus a sequency-order column pick costs more source than `np.fft.rfft2`
costs anything.
Natural Hadamard order is also not sequency order — column 1 of H8 is the *highest*
sequency — so "keep the low ones" needs an explicit reordering that the DCT and the
FFT get for free.

### Was this table capable of a negative?

`experiments/dct_stem.py cost` refuses to print a verdict unless some row costs 150 B
or more over `null`.
All four stored-basis rows do, at +212 to +412 B.
A criterion nothing can fail is not a criterion.

---

## The matched-bytes comparison

An identical quantized ridge head on every stem: global Lloyd-max codebook, 1–6 bits,
15 ridge penalties over 10^0 to 10^14, everything selected on the 5,000-image
validation split.
2,069 points, in `results/dct-stem-sweep.json`.

The lambda grid is wide and dense on purpose.
The stems differ in feature scale by up to 8x — an 8x8 DCT's DC term is 8x a block mean
— and the effective ridge penalty goes as the square of that.
A first pass on a coarse grid (powers of 100) produced an apparent +7.9 pp win for a
stem that is information-identical to `rgb4`; the entire effect was one arm being
better tuned than the other.
That is recorded here because it is the kind of confound this comparison invites.

| band | control | val | best transform | val | delta |
|---|---|---:|---|---:|---:|
| 300–500 B | `rgb4` | 27.50% | `hadabs8k2` | 28.34% | +0.84 (unresolved) |
| 500–700 B | `rgb4` | 32.46% | `fftabs16k2` | 36.54% | **+4.08** |
| 700–1000 B | `rgb4` | 34.98% | `hadabs8k2` | 38.66% | **+3.68** |
| 1000–1500 B | `rgb8` | 38.16% | `fftabs8k2` | 45.18% | **+7.02** |
| 1500–2200 B | `rgb8` | 41.18% | `fftabs8k2` | 47.70% | **+6.52** |

The control column is the *best* block mean in the band, not `rgb8` alone.
Against `rgb8` alone the margins are larger (+3.2 to +13.3 pp), but that is partly an
artifact: below 700 B a 192-d head does not fit and `rgb4`'s 48-d head does, so
comparing only against `rgb8` hands the small bands to any transform for a reason that
is not the transform.
Both readings are printed by the script; the stricter one is above.

The stored-basis control loses in every band, by +7.7 to +14.7 pp in the generated
stem's favour.

**On selection noise.** The repo's 1.06 pp floor is for n=10,000 over 45 configurations.
Validation is n=5,000, so the per-config standard error is 0.69 pp and the expected
inflation of a maximum over 90 configurations is about 2.1 pp.
That inflation applies equally to both arms of the 192-d table above, where every arm
has exactly 90 configurations, so it cancels in the difference.
It does not cancel in the band table, where the transform arm draws from more
configurations than the control; the margins there are 4 to 7 pp against a residual
bias of well under 1 pp.
The test numbers below are the check that matters, and every `fftabs` point tracks its
validation figure within 1.1 pp.

---

## Declared points, scored once on test

| artifact | bytes | src | coef | val | **test** |
|---|---:|---:|---:|---:|---:|
| `lt-fftabs16k2-2b` | 418 | 354 | 132 | 27.76% | **27.29%** |
| `lt-fftabs16k2-3b` | 531 | 415 | 201 | 33.02% | **32.72%** |
| `lt-fftabs16k2-4b` | 571 | 335 | 278 | 35.20% | **35.22%** |
| `lt-fftabs16k2-5b` | 698 | 416 | 372 | 36.54% | **37.62%** |
| `lt-fftabs8k2-4b` | 1,266 | 334 | 998 | 45.18% | **45.57%** |
| `lt-rgb8-6b` | 1,904 | 387 | 1,577 | 41.18% | 39.63% |
| `lt-dct8k2-6b` | 1,952 | 466 | 1,577 | 41.52% | 39.28% |
| `lt-fftabs8k2-6b` | 1,926 | 418 | 1,577 | 47.70% | **47.77%** |

Every val figure came back through the harness identical to the local mirror used for
the sweep, so there is no trainer/artifact drift to discount.
The last three rows are the mechanism controls, not frontier claims: all three are
dominated by `cf-k16-p4s2-4b-pc` at 1,713 B / 50.78%.

They do carry the mechanism result onto test, and it is larger there than on validation.
`fftabs8k2` beats `rgb8` by **+8.14 pp** on test at 22 more bytes, against +6.52 pp on
validation, while `dct8k2` — the exact orthogonal re-encoding of `rgb8` — comes in
0.35 pp *below* it.
The two block-mean-family controls are also the only declared points that drop more than
1.1 pp from validation to test (−1.55 and −2.24 pp), which widens the gap rather than
narrowing it.

### New frontier points

Against the existing board — `constant` 85 B / 10.00%, `obt-g8-b9` 470 B / 28.25%,
`g-k8-p4s2-4b` 931 B / 43.79%, `cf-k16-p4s2-4b-pc` 1,713 B / 50.78%:

| bytes | test | displaces |
|---:|---:|---|
| 418 | 27.29% | nothing between 85 B and 470 B |
| 531 | 32.72% | +4.47 pp over the 470 B oblivious table, for 61 B |
| 571 | 35.22% | +6.97 pp over the 470 B table, for 101 B |
| 698 | 37.62% | first occupant between 470 B and 931 B |
| 1,266 | 45.57% | +1.78 pp over the 931 B conv-ridge point |

**Five new Pareto points, and the 500–700 B band changes character.**
[what-to-try.md](what-to-try.md) calls that band "the one genuinely open band" and puts
the best candidates at ~512 B / 28.31% and ~680 B / 30.15% on validation, against a
best-below-700 B of 26.46% at 480 B.
`lt-fftabs16k2-4b` is 571 B at 35.22% **on test**, five points above any of them.

Two of these are close calls and should be read as such.
418 B / 27.29% against the 470 B table's 28.25% is 52 fewer bytes for 0.96 pp less,
which is inside the noise floor: smaller, not resolved on accuracy.
1,266 B / 45.57% against 931 B / 43.79% is +1.78 pp for 335 B, and the 931 B figure
predates the validation split and so is inflated by perhaps a point — treat that row as
the weakest of the five.

The three 500–700 B rows are not close calls.

---

## The golf, measured against the compressed size

Source is 55–65% of every artifact here, so this repo's largest sub-KB lever applies.
Three source-only changes: the input is left as uint8 for numpy to promote instead of
being cast on its own line; `np.fft.rfft2(..., axes=(2,4))` replaces a `.transpose` that
moved the block axes into place; and the codebook loses its intermediate name.

This repo has a recorded case where golfing cut 34 raw bytes and *added* 3 xz bytes, so
the comparison holds the weight file byte-identical and reports all three codecs
(`dct_stem.py golf`).

| point | form | raw | gzip | **xz** | src |
|---|---|---:|---:|---:|---:|
| `fftabs16k2` 2b | golfed | 506 | 436 | **418** | 354 |
| | ungolfed | 540 | 455 | 436 | 388 |
| `fftabs16k2` 4b | golfed | 633 | 586 | **571** | 335 |
| | ungolfed | 667 | 606 | 590 | 369 |
| `fftabs8k2` 4b | golfed | 1,352 | 1,287 | **1,266** | 334 |
| | ungolfed | 1,386 | 1,311 | 1,284 | 368 |
| `fftabs8k2` 6b | golfed | 2,015 | 1,975 | **1,926** | 418 |
| | ungolfed | 2,049 | 1,995 | 1,941 | 452 |

**34 raw source bytes become 15–18 xz bytes, on all eight points measured, with no
exceptions.**
The pathology did not recur here.
Predictions are verified unchanged: the two forms flatten the coefficients in different
orders, so the check compares the sorted feature rows rather than the raw ones — same
multiset per image, which is what "the golf changed no computation" means.

The `axes=` rewrite is the interesting half.
It is ten source bytes shorter than the transpose *and* it copies nothing, so it is
faster as well as smaller — the only reason the transpose form was written first is
that it reads more like the block structure it implements.

---

## The second scoreboard

`dct_stem.py mdl`, using `experiments/mdl_audit.py`'s own helpers rather than
reimplementing them: cross-entropy on the quantized weights actually shipped, with a
temperature fitted on the fit split, against the 4,152 B cost of sending the 10,000
test labels with no model at all.

| artifact | bytes | test | bits/label | labels | total | verdict |
|---|---:|---:|---:|---:|---:|---|
| `lt-fftabs16k2-2b` | 418 | 27.29% | 3.023 | 3,779 | 4,197 | costs 45 B |
| `lt-fftabs16k2-3b` | 531 | 32.72% | 2.686 | 3,358 | **3,889** | **pays, +264 B** |
| `lt-fftabs16k2-4b` | 571 | 35.22% | 2.646 | 3,307 | **3,878** | **pays, +274 B** |
| `lt-fftabs16k2-5b` | 698 | 37.62% | 2.559 | 3,199 | **3,897** | **pays, +255 B** |
| `lt-fftabs8k2-4b` | 1,266 | 45.57% | 2.267 | 2,834 | **4,100** | **pays, +52 B** |
| `lt-fftabs8k2-6b` | 1,926 | 47.77% | 2.220 | 2,775 | 4,701 | costs 548 B |

**Four new MDL-positive artifacts**, where the board previously had one.
[findings.md](findings.md) reports exactly one point that pays for its own
transmission — the 961 B ridge model at +241 B, or +356 B once golfed to 845 B.

It should be said plainly that **the best of these does not beat that point**: +274 B
against +356 B.
What changes is that the MDL-positive region stops being a single artifact and becomes
a band, with four points spread from 531 B to 1,266 B, and its cheapest occupant drops
from 845 B to 531 B.

The shape is the familiar one: the smallest artifact is not the best on this board
either.
418 B is the only declared point that fails, because at 27.29% it saves too few label
bytes to cover even 418.

---

## What this does not answer

**Whether the magnitude stem composes with the conv-ridge family.**
Everything above puts a *linear* head on the stem, which is the clean way to isolate the
transform but not the strongest classifier this repo has.
The conv family already applies random filters and a rectifier, which is a learned-free
generalisation of what `fftabs` does by hand, so the two may be redundant — or the stem
may be a cheaper first layer than the sliding-window conv.
At 931 B the conv-ridge point still beats every linear stem here; that comparison is the
obvious next experiment and it was not run.

**Whether TTA changes the ranking.**
Flip TTA is absent from every arm here and the omission is deliberate: with a linear head
over a linear stem, averaging the features of an image and its mirror is exactly
classifying the horizontally symmetrized image, which destroys every antisymmetric
coefficient — half of a block DCT.
The `fftabs` arms are *not* linear, so TTA would not be degenerate for them, and it
would be worth ~12 source bytes to find out.
That asymmetry means the comparison above is, if anything, conservative toward the
block mean.

**Bit widths above 6 and per-class codebooks.**
Both were left out to keep every arm identical. `findings.md` finding 7 says per-class
codebooks lose below ~2 KB anyway, which is the whole range here.

**Whether a better truncation pattern exists.**
Only square KxK coefficient blocks were tried, because that is what a matrix-form
transform expresses in one expression. A zigzag or an L-shaped pattern would need an
index list, which costs source.

---

## Hygiene

* All numbers from `experiments/dct_stem.py`; sweep record in
  `results/dct-stem-sweep.json`; the eight declared artifacts under `artifacts/lt-*`
  with their records in `results/lt-*.json`.
* Selection on `load_dev()`'s validation split throughout. Test scored once per
  declared artifact through `tinycifar/evaluate.py`.
* `python3 -m pytest tests -q`: **64 passed**, unchanged.
  `ruff check experiments tinycifar tests`: clean.
  (`ruff check .` reports 1,158 errors, all of them in pre-existing `artifacts/*/predict.py`
  — golfed artifact source is not lint-clean by construction.)
* `LEADERBOARD.md` was not regenerated and `docs/frontier.png` was not rebuilt.
  No existing file was modified.
* Every job under `nice -n 15` with `OMP_NUM_THREADS=3`, one at a time; box load stayed
  between 2.7 and 5.9 against 16 cores.
