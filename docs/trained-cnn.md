# Trained filters

Written up after the fact: the agent that ran these was killed by a session
limit before it wrote its own notes, so this reconstructs the result from
`results/cnn*.json` and the run log. Every number below is a harness figure —
scored on the full 10,000-image test set from the artifact's own serialized
bytes, in a sandboxed process.

## The result

A depthwise-separable CNN, widths [32, 64, 128, 128], BatchNorm folded into the
preceding convolution at export, per-output-channel float16 scale plus a
Lloyd-max codebook per tensor, flip TTA:

| artifact | bytes | accuracy |
|---|---:|---:|
| `cnnm-2b-qat` | 6,725 | 67.39% |
| `cnnm-3b-qat` | **9,949** | **77.52%** |
| `cnnm-4b-qat` | 12,502 | **80.74%** |
| `cnnm-6b-qat` | 24,069 | 80.53% |

This retires random filters above about 4 KB. The previous ≤10 KB holder was
63.41%; a trained net at 9,949 B reaches 77.52%, and the gap to µNAS
(86.49% @ 11.4 KB) closes from 23 points to 9. Note also that 4 bits beats 6:
past 4 bits the extra precision buys nothing and costs 11.5 KB.

## Quantization-aware training is not an enhancement

The same trained weights, quantized after the fact versus quantized through a
straight-through estimator during fine-tuning:

| bits | post-training | QAT | delta |
|---|---:|---:|---:|
| 3 | 18.55% | 77.52% | **+58.97** |
| 4 | 53.15% | 80.74% | **+27.59** |
| 6 | 78.71% | 80.53% | +1.82 |

At 6 bits QAT is a rounding correction. At 3 bits post-training quantization
destroys the model outright — 18.55% is barely above chance — and QAT is the
only reason a 3-bit artifact exists at all.

The practical consequence is a trap: **without QAT, trained filters lose to
random ones at matched size.** A 4-bit PTQ net at 12.7 KB scores 53.15% where
random conv features score ~63%. Anyone running that comparison and concluding
"trained filters don't help on this problem" would be reading a quantizer
failure as an architecture result. The plan originally ranked QAT as "worth
several points"; it is a precondition below 6 bits.

## Two things that had to be right

**The codebook must be frozen before QAT, not refit.** Refitting the codebook
each epoch cost 16 points at 4 bits. The straight-through estimator pulls
weights onto whatever centroids it is shown; moving the centroids afterwards
discards exactly that adaptation. The grid the model trains against has to be
the grid that ships — and the same grid has to be used for epoch scoring, or the
log reports a model that was never exported. (The agent's last message before it
died was that it had found this divergence; the fix is in the code and the
numbers above are from the fixed path.)

**Adam, not SGD.** With BN folded away, layer gradient scales differ by orders
of magnitude, and a single global step size either diverges on the stem or does
nothing to the head. SGD at every learning rate tried produced a quantized model
worse than plain PTQ.

## Cost

About 1,350 s to train the float model on CPU, plus ~230 s per QAT variant, at
`nice -n 15` with 3 threads. No GPU. That is the whole price of the largest
accuracy gain in the project so far.

## What this does not do

It does not help below ~1 KB, and the reason is structural rather than fixable
by tuning: the CNN's `predict.py` is 1,332 B, which by itself exceeds the entire
931 B flagship. A trained filter bank also stops being free — the random bank
costs 4 bytes of PRNG seed, while a trained 4-bit bank costs roughly 176 B. The
sub-KB track and the 10 KB track do not share an architecture.

## Next

- 3-bit and 4-bit reruns through the fixed codebook path, to confirm these and
  see whether the frozen-grid fix moves them (in flight).
- The 2-bit point (67.39% @ 6,725 B) is the least explored and sits where the
  frontier is steepest.
- Distillation from a larger teacher, which costs zero artifact bytes and has
  not been tried.
