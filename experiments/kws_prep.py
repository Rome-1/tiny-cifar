"""Build the 12-class Speech Commands v2 arrays from the shipped release.

The standard setup, as defined by Google's own `input_data.py` in the
TensorFlow speech_commands example and described in Warden 2018
(arXiv:1804.03209): ten target words, plus `_silence_` and `_unknown_`.

    yes no up down left right on off stop go  +  _silence_  _unknown_

Split membership is **not chosen here**. Every real utterance is assigned to
train / validation / test by the `validation_list.txt` and `testing_list.txt`
files that ship inside the tarball, which is the dataset's own speaker-disjoint
hash split. Nothing in this repo selects on test.

Two things the released lists do not determine, and how they are settled:

  * **How many `_unknown_` and `_silence_` items each split gets.** Google's
    reference code takes 10% of the split's target-word count for each. That
    convention is what makes the canonical test set 4,890 utterances, so it is
    reproduced exactly (`ceil` at 10%, as in `input_data.py`).
  * **Which unknowns, and what the silence waveforms are.** The unknown pool is
    the twenty non-target words already assigned to that split by the shipped
    lists, sampled with a fixed seed. Silence is a one-second crop of one of the
    six `_background_noise_` recordings scaled by a uniform gain, again with a
    fixed seed.

    The background recordings are shared across splits in Google's code. Here
    each recording is cut into three contiguous regions and a split may only
    draw from its own region. That is a deviation, and it is the conservative
    direction: it removes the possibility of a model recognising a particular
    stretch of dishwasher noise it saw in training.

Output is a single npz of int16 waveforms, [N, 16000], zero-padded or truncated
to exactly one second at 16 kHz -- the dataset's own nominal format.

Run:
    python3 experiments/kws_prep.py --out data/sc12.npz
"""

from __future__ import annotations

import argparse
import math
import sys
import wave
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

SC = REPO / "data" / "sc"
SR = 16000

WORDS = ("yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go")
SILENCE, UNKNOWN = 10, 11
CLASSES = WORDS + ("_silence_", "_unknown_")

# Google's input_data.py defaults.
SILENCE_PCT = 10.0
UNKNOWN_PCT = 10.0


def read_wav(path: Path) -> np.ndarray:
    """A one-second int16 mono frame at 16 kHz, padded or truncated."""
    with wave.open(str(path), "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2 or w.getframerate() != SR:
            raise ValueError(f"{path}: unexpected format")
        a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    if len(a) >= SR:
        return a[:SR].copy()
    out = np.zeros(SR, dtype=np.int16)
    out[: len(a)] = a
    return out


def read_wav_full(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def split_map() -> dict[str, str]:
    """relative path -> 'val' | 'test'. Everything else is train."""
    m = {}
    for name, tag in (("validation_list.txt", "val"), ("testing_list.txt", "test")):
        for line in (SC / name).read_text().split():
            m[line.strip()] = tag
    return m


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(REPO / "data" / "sc12.npz"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    if not (SC / "testing_list.txt").exists():
        raise FileNotFoundError(
            f"{SC} missing. Fetch and extract "
            "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz "
            "(CC-BY-4.0; the LICENSE ships inside the tarball)."
        )

    smap = split_map()
    targets = {s: [] for s in ("train", "val", "test")}
    unknowns = {s: [] for s in ("train", "val", "test")}

    for d in sorted(SC.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        for p in sorted(d.glob("*.wav")):
            rel = f"{d.name}/{p.name}"
            s = smap.get(rel, "train")
            (targets if d.name in WORDS else unknowns)[s].append((p, d.name))

    n_files = sum(len(v) for v in targets.values()) + sum(
        len(v) for v in unknowns.values())
    print(f"{n_files:,} wav files  "
          f"(release states 105,829 for v0.02)")

    # Background noise, cut into one contiguous region per split.
    bg = [read_wav_full(p) for p in sorted((SC / "_background_noise_").glob("*.wav"))]
    print(f"{len(bg)} background recordings, "
          f"{sum(len(b) for b in bg) / SR / 60:.1f} min total")
    region = {"train": 0, "val": 1, "test": 2}

    out = {}
    rng = np.random.default_rng(a.seed)
    for split in ("train", "val", "test"):
        tgt = targets[split]
        n_t = len(tgt)
        n_sil = int(math.ceil(n_t * SILENCE_PCT / 100))
        n_unk = int(math.ceil(n_t * UNKNOWN_PCT / 100))

        xs = np.zeros((n_t + n_sil + n_unk, SR), dtype=np.int16)
        ys = np.zeros(len(xs), dtype=np.int64)

        for i, (p, word) in enumerate(tgt):
            xs[i] = read_wav(p)
            ys[i] = WORDS.index(word)

        pool = unknowns[split]
        pick = rng.choice(len(pool), size=n_unk, replace=n_unk > len(pool))
        for j, k in enumerate(pick):
            xs[n_t + j] = read_wav(pool[k][0])
            ys[n_t + j] = UNKNOWN

        r = region[split]
        for j in range(n_sil):
            b = bg[rng.integers(len(bg))]
            lo, hi = len(b) * r // 3, len(b) * (r + 1) // 3
            if hi - lo <= SR:
                lo, hi = 0, len(b)
            o = int(rng.integers(lo, hi - SR))
            gain = float(rng.random())
            xs[n_t + n_unk + j] = np.clip(
                b[o:o + SR].astype(np.float32) * gain, -32768, 32767
            ).astype(np.int16)
            ys[n_t + n_unk + j] = SILENCE

        perm = rng.permutation(len(xs))
        out[f"x{split}"] = xs[perm]
        out[f"y{split}"] = ys[perm]
        print(f"{split:>5}: {len(xs):>6,} = {n_t:,} words "
              f"+ {n_unk:,} unknown + {n_sil:,} silence   "
              f"counts {np.bincount(ys, minlength=12).tolist()}")

    Path(a.out).parent.mkdir(exist_ok=True)
    np.savez(a.out, **out)
    print(f"\nwrote {a.out}  ({Path(a.out).stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
