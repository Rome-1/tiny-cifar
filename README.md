# tiny-cifar

How small can a CIFAR-10 classifier be? Not in parameters — in bytes of a
self-contained artifact that actually runs.

## The frontier

![size vs accuracy](docs/frontier.png)

Our artifacts against published models. ◆ marks the joint Pareto frontier: points
nothing smaller matches on accuracy.

<!-- FRONTIER:START -->
| | model | size | accuracy | source |
|---|---|---:|---:|---|
| ◆ | `constant` | 85 B | 10.00% | this repo |
| ◆ | `lin-gray4-5b-symmetric` | 480 B | 26.46% | this repo |
| ◆ | `g-k8-p4s2-4b` | 931 B | 43.79% | this repo |
| ◆ | `cf-k16-p4s2-4b-pc` | 1.7 KB | 50.78% | this repo |
| ◆ | `g-k32-p4s2-8b` | 3.8 KB | 57.33% | this repo |
| ◆ | `gc-cnntsmall-4b-qat` | 5.1 KB | 73.61% | this repo |
| ◆ | µNAS | 11.1 KB | 86.49% | Liberis et al. 2021 |
|  | `gc-cnnm-4b-qat` | 12.1 KB | 80.74% | this repo |
|  | `gc-cnnlqat-3b-qat` | 29.9 KB | 82.49% | this repo |
|  | linear SVM on raw pixels | 30.0 KB | 49.88% | classic baseline |
|  | `gc-cnnlqat-4b-qat` | 38.9 KB | 84.21% | this repo |
|  | `gc-cnnxlqat-4b-qat` | 69.3 KB | 85.37% | this repo |
| ◆ | Entropy Penalized Reparam. (VGG-16) | 101.0 KB | 90.00% | Oktay et al. 2020 |
|  | MIRACLE (VGG-16) | 135.0 KB | 90.00% | Havasi et al. 2019 |
|  | Coates K-means 1600 + SVM | 242.6 KB | 77.90% | Coates & Ng 2011 |
| ◆ | ResNet-20 | 1.0 MB | 91.25% | He et al. 2016 |
| ◆ | ResNet-56 | 3.2 MB | 93.03% | He et al. 2016 |
| ◆ | airbench94 (TTA) | 7.5 MB | 94.01% | Jordan 2024 |
|  | hlb-CIFAR10 | 16.5 MB | 94.00% | tysam-code 2023 |
<!-- FRONTIER:END -->

Regenerate with `python -m tinycifar.frontier`. Byte accounting for the published
rows is in [docs/sota-datapoints.md](docs/sota-datapoints.md).

Three caveats before drawing conclusions from any row:

*The comparison is biased against us.* Every published byte count here excludes
the architecture and inference code our metric charges us for. µNAS's 11.4 KB is
the authors' own `params × int8` arithmetic, not a measured file; the ResNet and
hlb figures are `params × fp32` derived by us. Only the two VGG-16 rows are real
bitstreams. Our numbers are bytes of a thing that runs.

*Below ~11 KB there is nothing to compare against.* The published sub-KB
CIFAR-10 results are 2-class relabelings where chance is 50%, confirmed from the
primary sources. The 10-class cell under 10 KB is empty, so our points there have
no published rival — a statement about what has been measured, not a trophy.

*Accuracy is top-1 on the full 10,000-image test set,* scored by re-running each
artifact from its serialized bytes in a clean subprocess. Most rows predate a
validation split, so their configurations were chosen on test — which inflates
the best of a large sweep by roughly a point. Treat differences under ~1.5 points
as unresolved. New work tunes on a held-out split of train.

## How size is measured

An artifact is a directory with a `predict.py` exposing `predict(x)` for uint8
`[N,32,32,3]` images. Size is the smallest of raw, gzip and xz over a minimal
container — see [docs/harness.md](docs/harness.md) for why it is hand-rolled and
what the rules are.

The decoder source counts. Without that, any amount of model can be smuggled
into code as literals, and the smallest classifier would be a program with the
answers baked in. It has a practical consequence: below a kilobyte the Python is
about two-thirds of the artifact, so shrinking the emitted decoder matters more
than shrinking the weights.

The artifact may not read the dataset, import anything outside numpy, or let one
image's prediction depend on the others in the batch. All three are enforced, not
requested.

## What is here

```bash
python -m tinycifar.leaderboard      # rebuild LEADERBOARD.md from results/
python experiments/trained_cnn.py    # the >4 KB family
python experiments/golf.py           # the <4 KB family
python tests/test_artifact.py        # tests for the size metric itself
```

The frontier has two halves. Below ~4 KB it is closed-form ridge regression with
no backprop, on features that cost nothing to ship: a PRNG seed is four bytes
however much it draws, so random convolutional filters are free and only the
classifier head is real weight. Above ~4 KB it is trained depthwise-separable
CNNs with quantization-aware training, which retired the random-filter family
entirely — 73.61% in 5.1 KB against 71.33% in 69 KB.

QAT turned out to be a precondition rather than a refinement: at 3 bits,
post-training quantization scores 18.55% and QAT scores 77.52%.

Scaling that family up to 85.37% in 69 KB costs six times the bytes µNAS spends
to beat it, so the gap at 10 KB is architecture and not training budget. Within
the family, precision beats width at fixed bytes as reliably as it did for random
filters: 4 bits on the smaller net wins 84.21% in 38.9 KB against 3 bits on the
larger net's 83.26% in 51.9 KB.

Results and reasoning: [findings.md](docs/findings.md),
[trained-cnn.md](docs/trained-cnn.md), [what-to-try.md](docs/what-to-try.md).

There is a second way to score this, where artifact bytes and accuracy become
one number: the bytes needed to transmit the test labels, plus the model that
predicts them. On that board most of this frontier does not pay for itself.
See [the MDL track](docs/findings.md#the-mdl-track--pricing-accuracy-in-bits).

## Contributions

Beating a row is the point. If you have a smaller artifact at the same accuracy,
or a better one at the same size, open a PR with the artifact, the script that
produced it, and the number `tinycifar.evaluate` gives it.

Anything is fair game as long as it survives the harness — architectures,
quantizers, entropy coders, procedural weights, or straightforward golf. The
1 KB region is the least explored and the easiest place to find something new.
