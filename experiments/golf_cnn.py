"""Golf the trained-CNN decoder, without retraining anything.

The weight blob is already optimal for its bit width; what is not optimal is the
1,329 bytes of Python that unpacks it. At the 4,133 B artifact that source is 32%
of the raw bytes, and at the 3,077 B one it is 36% — the same finding that made
source the largest sub-KB lever for the ridge family applies here, just less
extremely.

Nothing is retrained. This reads an existing artifact, re-emits `predict.py` in a
tighter form against the *same* `w` file, and refuses to keep the result unless
it reproduces the original's predictions on the full test set exactly. A smaller
artifact that predicts differently is not the same model.

The savings are the same kinds as `golf.py`: a width-specialized unpacker where
the width divides 8, no `pathlib` import for one path join, and dead generality
removed.

One thing that did *not* work, and is worth remembering: aliasing `np.frombuffer`
and `np.float32` cut 34 raw bytes and **added 3 to the xz size**. Golfing removes
repetition, and repetition is exactly what the compressor was eating. Source golf
has to be measured against the compressed metric, not the raw one.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from experiments.baselines import emit  # noqa: E402
from experiments.golf import unpack_expr  # noqa: E402
from tinycifar import artifact as A  # noqa: E402
from tinycifar.evaluate import evaluate, summarize  # noqa: E402

TEMPLATE = '''import numpy as np
B=open(__file__[:-10]+"w","rb").read()
b={BITS};k={K};c={WIDTHS};o={OFF}
M=np.maximum
C=np.frombuffer(B,np.float16,16*k,1).astype(np.float32).reshape(16,k)
S=np.frombuffer(B,np.float16,{NS},o).astype(np.float32)
q={UNPACK}
SP=[(c[0],(c[0],3,3,3)),(1,(c[0],))]
for i in range(3):SP+=[(1,(c[i],3,3)),(1,(c[i],)),(c[i+1],(c[i+1],c[i])),(1,(c[i+1],))]
SP+=[(10,(10,c[3])),(1,(10,))]
P=[];p=0;g=0
for j,(m,s) in enumerate(SP):
 n=int(np.prod(s));P.append((C[j,q[p:p+n]].reshape(m,-1)*S[g:g+m,None]).reshape(s));p+=n;g+=m
def sw(x):
 return np.lib.stride_tricks.sliding_window_view(np.pad(x,((0,0),(1,1),(1,1),(0,0))),(3,3),(1,2))
def mp(x):
 n,h,w,d=x.shape
 return x.reshape(n,h//2,2,w//2,2,d).max((2,4))
def fw(x):
 v=sw(x);h=mp(M(v.reshape(len(x),32,32,-1)@P[0].reshape(c[0],-1).T+P[1],0))
 for i in range(3):
  j=2+4*i
  h=M((sw(h)*P[j]).sum((4,5))+P[j+1],0)
  h=M(h@P[j+2].T+P[j+3],0)
  if i<2:h=mp(h)
 return h.mean((1,2))@P[14].T+P[15]
def predict(x):
 return np.concatenate([np.argmax(fw(z)+fw(z[:,:,::-1]),1)for z in np.array_split(x.astype(np.float32)/255,-(-len(x)//250))])
'''


def parse(src: str) -> dict:
    """Recover the generator's parameters from the artifact it emitted."""
    widths = re.search(r"c=(\[[^\]]+\])", src).group(1)
    ns = int(re.search(r"np\.float16,(\d+),o\)", src).group(1))
    nq = int(re.search(r"\[:(\d+)\*b\]", src).group(1))
    return {"widths": widths.replace(" ", ""), "ns": ns, "nq": nq}


def build(art: Path) -> dict[str, bytes]:
    old = (art / "predict.py").read_text()
    blob = (art / "w").read_bytes()
    p = parse(old)
    bits = blob[0]
    off = 1 + 2 * 16 * (1 << bits)
    # golf.py's byte-split unpacker returns a (nbytes, codes-per-byte) array and
    # relies on its caller's reshape to flatten it. This decoder indexes q as a
    # flat array, so it has to be flattened here — 2-bit slipped through only
    # because its code count is not divisible by four.
    u = unpack_expr(bits, p["nq"], off + 2 * p["ns"])
    if bits in (1, 2, 4) and p["nq"] % (8 // bits) == 0:
        u = f'{u}.reshape(-1)'

    src = TEMPLATE.format(
        BITS=bits, K=1 << bits, WIDTHS=p["widths"], OFF=off, NS=p["ns"],
        UNPACK=u,
    )
    return {"predict.py": src.encode(), "w": blob}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("artifacts", nargs="+",
                    help="artifact names, with or without the artifacts/ prefix")
    a = ap.parse_args(argv)

    for spec in a.artifacts:
        spec = Path(spec).name  # tolerate a tab-completed artifacts/<name> path
        art = REPO / "artifacts" / spec
        if not (art / "predict.py").exists():
            print(f"  ! {spec}: no artifact")
            continue

        before = A.measure(A.read_dir(art))
        files = build(art)
        after = A.measure(files)
        name = f"gc-{spec}"
        d = emit(name, files)

        # Same weights, smaller source: any prediction change is a bug.
        base = evaluate(art, name=f"_base-{spec}", save=False)
        r = evaluate(
            d, name=name,
            method=f"golfed decoder for {spec} (weights unchanged)",
            notes=f"source {len(files['predict.py'])} B "
                  f"(was {len((art / 'predict.py').read_bytes())} B)",
        )
        ok = abs(r["accuracy"] - base["accuracy"]) < 1e-9
        print(f"  {summarize(r)}")
        print(f"    {before.description_length:,} -> {after.description_length:,} B "
              f"({before.description_length - after.description_length:+,}), "
              f"accuracy {'identical' if ok else 'CHANGED — REJECT'}")
        if not ok:
            raise SystemExit(f"{name}: predictions changed; not a valid golf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
