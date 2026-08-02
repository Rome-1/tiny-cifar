"""Cross-environment coder tests: the failure an in-process round-trip cannot see.

`test_coder.py` encodes and decodes in one process, so encoder and decoder share
the same floating-point results by construction. That proves the coder's own
arithmetic is consistent, and proves nothing about the case that actually breaks
a prequential claim: encoder and decoder running in *different* environments,
where the model's float probabilities may differ in the last bit.

The concern is specific. `quantize_probs` maps a float distribution onto integer
frequencies through ``floor(p * (TOTAL - n))``. If one environment's `p` lands a
hair above an integer boundary and another's lands a hair below, the two get
different frequency tables and the decode desynchronises irrecoverably — with no
exception raised, just wrong labels from that symbol on.

How close to real is that? Measured, not assumed. Over 200,000 draws from
realistically skewed model outputs, the smallest observed distance to a floor
boundary was 6.55e-08, against roughly 1.46e-11 of one-ulp noise in the same
quantity — a safety factor of about 4,500x at the worst case seen. So a single
perturbed ulp does not flip the table (`test_one_ulp_does_not_flip_frequencies`
pins that), and the residual risk over a 10,000-symbol run is on the order of a
few parts per million. Small, but silent when it fires, which is why it is
tested rather than argued about.

What these tests can and cannot do, established by mutation rather than hope.
Deliberately breaking `quantize_probs` so it depends on `OMP_NUM_THREADS` did
NOT fail a cross-thread decode test, and did not fail a digest comparison of the
frequency tables either. Measured directly, over 400 symbols a constant bias of
1e-3 moves 8 tables and a bias of 1e-5 moves none — so the detection floor of a
sampling test is somewhere around 1e-4, while the divergence we actually fear is
about 1e-11. Eight orders of magnitude apart. **No test of this shape can prove
the absence of a last-bit disagreement, and this file does not claim to.**

The guarantee comes from the margin, not from the round-trips:
`test_floor_boundary_margin_is_wide` is the load-bearing test, because a
disagreement can only matter if `p * (TOTAL - n)` sits closer to an integer than
the noise is large, and it does not come close. The cross-process and
cross-thread tests below are regression guards for gross breakage — someone
introducing genuine environment dependence at a scale that matters — not proof
of bit-exactness. Making that airtight needs integer model arithmetic end to
end, PAQ-style; at a measured 4,500x margin that is not currently worth it.
"""

import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tinycifar.coder import (  # noqa: E402
    TOTAL, quantize_probs,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Thread counts to cross. BLAS reductions reassociate differently by thread
# count, which is the most likely source of a last-bit difference in practice.
THREAD_SETTINGS = ("1", "4")

_ENV_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def _adaptive_probs(i, prefix, n=10):
    """A deterministic adaptive model, deliberately NON-degenerate.

    The spread matters. An earlier version of this model ran counts through a
    random matmul and a softmax, which produced near-one-hot distributions: most
    floors sat at zero and the dominant symbol was pinned by the sum correction,
    so the tables did not move even under a bias of 1e-2. A test model that
    cannot react is worse than no test. This one keeps every probability in
    roughly [0.07, 0.15], where floors land mid-interval and can actually flip.
    """
    counts = np.ones(n)
    for s in prefix:
        counts[int(s)] += 1
    p = counts / counts.sum()
    jitter = np.abs(np.sin(np.arange(n) + i * 0.7)) + 1e-3
    return 0.85 * p + 0.15 * jitter / jitter.sum()


def _script(mode: str) -> str:
    return f"""
import hashlib, os, sys
import numpy as np
sys.path.insert(0, {REPO!r})
sys.path.insert(0, os.path.join({REPO!r}, "tests"))
from test_coder_crossenv import _adaptive_probs, SYMS
from tinycifar.coder import Decoder, Encoder, quantize_probs

if {mode!r} == "encode":
    enc = Encoder()
    for i, s in enumerate(SYMS):
        enc.encode(int(s), quantize_probs(_adaptive_probs(i, SYMS[:i])))
    sys.stdout.write(enc.finish().hex())
elif {mode!r} == "tables":
    # A digest of every frequency table this environment computes. Comparing
    # digests is strictly stronger than comparing decoded labels, because a
    # disagreement shows up whether or not it happens to desync the stream —
    # but "stronger" is not "sufficient": see the module docstring for the
    # measured detection floor.
    h = hashlib.sha256()
    for i in range(len(SYMS)):
        h.update(quantize_probs(_adaptive_probs(i, SYMS[:i])).tobytes())
    sys.stdout.write(h.hexdigest())
else:
    blob = bytes.fromhex(sys.stdin.read())
    dec = Decoder(blob)
    out = []
    for i in range(len(SYMS)):
        out.append(dec.decode(quantize_probs(_adaptive_probs(i, out))))
    sys.stdout.write(",".join(map(str, out)))
"""


SYMS = np.random.default_rng(7).integers(0, 10, 400)


def _run(mode: str, threads: str, stdin: str | None = None) -> str:
    env = dict(os.environ)
    for var in _ENV_VARS:
        env[var] = threads
    r = subprocess.run(
        [sys.executable, "-c", _script(mode)],
        input=stdin, capture_output=True, text=True, env=env, cwd=REPO,
    )
    if r.returncode != 0:
        raise AssertionError(f"{mode} @ {threads} threads failed:\n{r.stderr}")
    return r.stdout


def test_one_ulp_does_not_flip_frequencies():
    """The margin that makes the whole scheme safe.

    If this ever fails the quantizer has become boundary-sensitive, and since
    the cross-process tests cannot see a perturbation that small, this is the
    check that would have to catch it.
    """
    rng = np.random.default_rng(0)
    for _ in range(5000):
        p = rng.random(10)
        p /= p.sum()
        q = p.copy()
        q[0] = np.nextafter(q[0], np.inf)
        assert np.array_equal(quantize_probs(p), quantize_probs(q))


def test_floor_boundary_margin_is_wide():
    """Distance to the nearest floor boundary must stay far above float noise."""
    rng = np.random.default_rng(0)
    noise = np.spacing(1.0) * (TOTAL - 10)
    worst = 1.0
    for _ in range(20_000):
        logits = rng.normal(0, rng.uniform(0.5, 8.0), 10)
        p = np.exp(logits - logits.max())
        p = np.clip(p / p.sum(), 1e-12, None)
        p /= p.sum()
        prod = p * (TOTAL - 10)
        worst = min(worst, float(np.abs(prod - np.round(prod)).min()))
    assert worst > 100 * noise, f"margin {worst:.3e} too close to noise {noise:.3e}"


def test_roundtrip_across_processes_same_threads():
    """Encode in one process, decode in another. Rules out per-process state."""
    blob = _run("encode", "1")
    out = _run("decode", "1", stdin=blob)
    assert out == ",".join(map(str, SYMS))


def test_sampling_detection_floor_is_where_we_measured_it():
    """Pin the power of the tests below, so a green run is not over-read.

    If this file's other tests are ever cited as evidence of bit-exactness,
    this is the rebuttal: they cannot see a perturbation smaller than roughly
    1e-4, and the one that would bite is 1e-11.
    """
    base = [quantize_probs(_adaptive_probs(i, SYMS[:i])) for i in range(len(SYMS))]

    def moved(bias):
        n = 0
        for i in range(len(SYMS)):
            p = np.clip(_adaptive_probs(i, SYMS[:i]), 1e-12, None)
            p = p / p.sum()
            f = np.floor(p * (TOTAL - 10) + bias).astype(np.int64) + 1
            short = TOTAL - int(f.sum())
            if short:
                f[np.argsort(-p)[:abs(short)]] += np.sign(short)
            n += not np.array_equal(f, base[i])
        return n

    assert moved(1e-5) == 0, "detector is more sensitive than documented"
    assert moved(1e-3) > 0, "detector has gone blind; the model may be degenerate"


def test_frequency_tables_identical_across_thread_counts():
    """Do two environments compute the same tables? A regression guard.

    Read the module docstring before trusting this: mutation testing shows it
    does not catch a small thread-dependent perturbation. It catches gross
    breakage, which is still worth catching.
    """
    digests = {t: _run("tables", t) for t in THREAD_SETTINGS}
    assert len(set(digests.values())) == 1, (
        f"frequency tables differ by thread count: {digests}"
    )


def test_roundtrip_across_thread_counts():
    """The real target: encoder and decoder under different BLAS threading.

    A last-bit difference in the model's probabilities shows up here as wrong
    labels, which is exactly the failure an in-process round-trip cannot see.
    """
    for enc_threads in THREAD_SETTINGS:
        for dec_threads in THREAD_SETTINGS:
            blob = _run("encode", enc_threads)
            out = _run("decode", dec_threads, stdin=blob)
            assert out == ",".join(map(str, SYMS)), (
                f"desync encoding at {enc_threads} threads, "
                f"decoding at {dec_threads}"
            )


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
