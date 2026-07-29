"""Trained depthwise-separable CNNs — the first backprop in this rig.

Everything on the frontier so far ships a closed-form ridge head on filters that
were never learned. That buys a lot for four bytes, and it stops at ~66%. The
published bar (µNAS, 86.5% in 11.4 KB) is made of *trained* filters, so the gap
is not mysterious: we have never spent a gradient.

    x -> conv3x3 -> [depthwise 3x3, pointwise 1x1] x 3 -> GAP -> linear

Depthwise-separable because the metric is bytes: a 3x3 depthwise layer costs 9
parameters per channel instead of 9*C_in, so almost the entire parameter budget
lands in the pointwise mixing where it does the most work per byte.

What is free, and therefore used without restraint
--------------------------------------------------
Training-time anything. Random crop and flip augmentation, label smoothing, an
EMA of the weights, a cosine schedule, and quantization-aware fine-tuning all
cost exactly zero artifact bytes. Flip TTA at inference costs about thirty bytes
of source and is worth a couple of points, because inference time is free too.

What is not free
----------------
BatchNorm. Its four parameters per channel are pure overhead once training is
over, so it is folded into the preceding convolution at export and never
shipped. And the decoder source itself: the forward pass is ~900 bytes of
numpy, which is a fifth of the smallest artifact here. It is written tersely for
that reason.

The quantizer
-------------
Per-output-channel scale (float16) plus a Lloyd-max codebook shared across all
weight tensors of the normalized values. The per-channel scale is what makes a
single codebook viable: folding BN leaves channel norms varying by an order of
magnitude, and one global grid over the raw weights spends its levels on
whichever channel dominates. Depthwise tensors get one scale for the whole
tensor rather than one per channel — nine weights do not justify a two-byte
scale.

Trainer/artifact drift
----------------------
The verification step execs the artifact's own ``predict.py`` and requires exact
agreement with the torch model on 2,000 test images. A silent mismatch between
the two forward passes would show up as a plausible accuracy, not a crash, which
is exactly the failure mode that is hardest to notice.

Model selection is done on a 5,000-image split held out of train. The test set
is touched once per exported point.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("MKL_NUM_THREADS", "3")

from experiments.baselines import emit  # noqa: E402
from experiments.quant_sweep import lloyd_max  # noqa: E402
from tinycifar import pack as P  # noqa: E402
from tinycifar.data import load  # noqa: E402
from tinycifar.evaluate import evaluate, summarize  # noqa: E402

ARCHS = {
    "t": [16, 32, 48, 64],
    "s": [24, 48, 80, 96],
    "m": [32, 64, 128, 128],
    "l": [48, 96, 192, 256],
    "xl": [64, 128, 256, 384],
}

# ---------------------------------------------------------------------------
# the artifact — a numpy forward pass and the unpacker for its weights
# ---------------------------------------------------------------------------

ARTIFACT_SRC = '''import numpy as np,pathlib
B=(pathlib.Path(__file__).parent/"w").read_bytes()
b=B[0];k=1<<b;c={W};o=1+2*{NCB}*k
C=np.frombuffer(B,np.float16,{NCB}*k,1).astype(np.float32).reshape({NCB},k)
S=np.frombuffer(B,np.float16,{NG},o).astype(np.float32)
q=np.unpackbits(np.frombuffer(B[o+2*{NG}:],np.uint8),bitorder="little")[:{NW}*b].reshape(-1,b)
q=(q.astype(np.uint32)<<np.arange(b,dtype=np.uint32)).sum(1)
SP=[(c[0],(c[0],3,3,3)),(1,(c[0],))]
for i in range(3):SP+=[(1,(c[i],3,3)),(1,(c[i],)),(c[i+1],(c[i+1],c[i])),(1,(c[i+1],))]
SP+=[(10,(10,c[3])),(1,(10,))]
P=[];p=0;g=0
for j,(m,s) in enumerate(SP):
 n=int(np.prod(s));P.append((C[{U},q[p:p+n]].reshape(m,-1)*S[g:g+m,None]).reshape(s));p+=n;g+=m
def sw(x):
 x=np.pad(x,((0,0),(1,1),(1,1),(0,0)))
 return np.lib.stride_tricks.sliding_window_view(x,(3,3),(1,2))
def mp(x):
 n,h,w,d=x.shape
 return x.reshape(n,h//2,2,w//2,2,d).max((2,4))
def fw(x):
 v=sw(x);h=np.maximum(v.reshape(v.shape[0],32,32,-1)@P[0].reshape(c[0],-1).T+P[1],0);h=mp(h)
 for i in range(3):
  j=2+4*i
  h=np.maximum((sw(h)*P[j]).sum((4,5))+P[j+1],0)
  h=np.maximum(h@P[j+2].T+P[j+3],0)
  if i<2:h=mp(h)
 return h.mean((1,2))@P[14].T+P[15]
def predict(x):
 o=np.empty(len(x),np.int64)
 for i in range(0,len(x),250):
  z=x[i:i+250].astype(np.float32)/255
  o[i:i+250]=np.argmax(fw(z){TTA},1)
 return o
'''

TTA_SRC = "+fw(z[:,:,::-1])"


def spec(widths, cb="tensor"):
    """(codebook, n_groups, shape) per tensor — mirrors the artifact source.

    `cb="tensor"` gives each tensor its own codebook; `"shared"` gives one to
    all weights and one to all biases (weights sit at even indices).
    """
    c = widths
    sp = [(c[0], (c[0], 3, 3, 3)), (1, (c[0],))]
    for i in range(3):
        sp += [(1, (c[i], 3, 3)), (1, (c[i],)),
               (c[i + 1], (c[i + 1], c[i])), (1, (c[i + 1],))]
    sp += [(10, (10, c[3])), (1, (10,))]
    return [((j if cb == "tensor" else j % 2), ng, sh)
            for j, (ng, sh) in enumerate(sp)]


# ---------------------------------------------------------------------------
# torch side
# ---------------------------------------------------------------------------

def build_torch(widths):
    import torch.nn as nn

    c = widths
    L = [nn.Conv2d(3, c[0], 3, padding=1, bias=False),
         nn.BatchNorm2d(c[0]), nn.ReLU(inplace=True), nn.MaxPool2d(2)]
    for i in range(3):
        L += [nn.Conv2d(c[i], c[i], 3, padding=1, groups=c[i], bias=False),
              nn.BatchNorm2d(c[i]), nn.ReLU(inplace=True),
              nn.Conv2d(c[i], c[i + 1], 1, bias=False),
              nn.BatchNorm2d(c[i + 1]), nn.ReLU(inplace=True)]
        if i < 2:
            L += [nn.MaxPool2d(2)]
    return nn.Sequential(*L, nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                         nn.Linear(c[3], 10))


class Folded:
    """The exact forward pass the artifact runs, in torch, with BN folded in.

    Kept as a plain parameter list so it can be fake-quantized during QAT and
    compared tensor-for-tensor with what gets packed.
    """

    def __init__(self, tensors, widths):
        import torch

        self.w = [torch.nn.Parameter(torch.as_tensor(t, dtype=torch.float32))
                  for t in tensors]
        self.c = widths
        self.torch = torch

    def parameters(self):
        return self.w

    def forward(self, x, ws=None):
        F = self.torch.nn.functional
        p = ws if ws is not None else self.w
        h = F.relu(F.conv2d(x, p[0], p[1], padding=1))
        h = F.max_pool2d(h, 2)
        for i in range(3):
            j = 2 + 4 * i
            h = F.relu(F.conv2d(h, p[j].unsqueeze(1), p[j + 1], padding=1,
                                groups=self.c[i]))
            h = F.relu(F.conv2d(h, p[j + 2].unsqueeze(-1).unsqueeze(-1), p[j + 3]))
            if i < 2:
                h = F.max_pool2d(h, 2)
        return h.mean((2, 3)) @ p[14].T + p[15]


def fold_bn(model, widths):
    """conv -> BN collapses to conv + bias. BN parameters are never shipped."""
    import torch
    import torch.nn as nn

    convs, bns = [], []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            convs.append(m)
        elif isinstance(m, nn.BatchNorm2d):
            bns.append(m)
    out = []
    for cv, bn in zip(convs, bns):
        s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        w = cv.weight * s.reshape(-1, 1, 1, 1)
        b = bn.bias - bn.running_mean * s
        if cv.groups > 1:                      # depthwise: (C,1,3,3) -> (C,3,3)
            w = w.squeeze(1)
        elif cv.kernel_size == (1, 1):         # pointwise: (Co,Ci,1,1) -> (Co,Ci)
            w = w.reshape(w.shape[0], w.shape[1])
        out += [w.detach().cpu().numpy(), b.detach().cpu().numpy()]
    fc = [m for m in model.modules() if isinstance(m, nn.Linear)][0]
    out += [fc.weight.detach().cpu().numpy(), fc.bias.detach().cpu().numpy()]
    got = [t.shape for t in out]
    want = [s[2] for s in spec(widths)]
    if got != want:
        raise RuntimeError(f"folded shapes {got} != spec {want}")
    return out


# ---------------------------------------------------------------------------
# quantization: per-group scale + Lloyd-max codebook
#
# Two knobs mattered more than expected, both found by measurement rather than
# taste. (1) A codebook per tensor rather than one shared across the network:
# the layers' normalized weight distributions differ enough that one grid fits
# none of them. (2) The per-group scale must not be max|w|. A single outlier in
# a channel stretches the grid over a value that carries almost none of the
# decision, which is the same failure the codebook sweep found in the ridge
# head. The scale is picked per group by minimizing reconstruction error over a
# small grid of clipping ratios.
# ---------------------------------------------------------------------------

CLIP_GRID = (0.45, 0.55, 0.65, 0.75, 0.85, 1.0)


def _assign(v, c):
    return np.abs(v[:, None] - c[None, :]).argmin(1)


FP16_MIN = 6.2e-5      # below this a float16 scale underflows to zero


def _max_scale(w2, a=1.0):
    base = np.abs(w2).max(1) * a
    base = np.maximum(base, FP16_MIN)
    return base.astype(np.float16).astype(np.float32)


def _best_scale(w2, c, grid=CLIP_GRID):
    """Per-group scale minimizing squared reconstruction error."""
    best_s, best_e = None, None
    for a in grid:
        s = _max_scale(w2, a)
        idx = _assign((w2 / s[:, None]).reshape(-1), c).reshape(w2.shape)
        e = ((c[idx] * s[:, None] - w2) ** 2).sum(1)
        if best_e is None:
            best_s, best_e = s, e
        else:
            m = e < best_e
            best_s = np.where(m, s, best_s)
            best_e = np.where(m, e, best_e)
    return best_s.astype(np.float32)


def _fit(pool, bits, rng):
    if len(pool) > 100_000:
        pool = pool[rng.choice(len(pool), 100_000, False)]
    c, _ = lloyd_max(pool, bits)
    return np.asarray(c, np.float16)


def fit_codebooks(tensors, sp, bits, ncb, scales=None):
    """Lloyd-max over the normalized values of every tensor in each group."""
    pools = [[] for _ in range(ncb)]
    for j, (t, (u, ng, _)) in enumerate(zip(tensors, sp)):
        v = np.asarray(t, np.float32).reshape(ng, -1)
        s = _max_scale(v) if scales is None else scales[j]
        pools[u].append((v / s[:, None]).reshape(-1))
    rng = np.random.default_rng(0)
    return [_fit(np.concatenate(p), bits, rng) for p in pools]


def encode(tensors, sp, bits, cbs=None, refit=1, grid=CLIP_GRID, only=None):
    """Quantize. Returns (blob, dequantized tensors, codebooks).

    The scale and the codebook are fit against each other: a codebook fit on
    max-normalized weights is the wrong grid once the scales clip, so after the
    scales move the codebook is refit on the values it will actually see.
    `refit=0` skips that and is the naive version.

    `cbs` may be reused across calls: refitting Lloyd-max is seconds of work,
    which is fine once an epoch and far too slow once a step. `only`, if given,
    is the set of tensor indices to quantize — the rest pass through in float,
    which is how per-tensor sensitivity is measured.
    """
    ncb = max(s[0] for s in sp) + 1
    w2s = [np.asarray(t, np.float32).reshape(s[1], -1) for t, s in zip(tensors, sp)]
    fixed = cbs is not None
    if not fixed:
        cbs = fit_codebooks(tensors, sp, bits, ncb)
        for _ in range(refit):
            sc = [_best_scale(w, np.asarray(cbs[u], np.float32), grid)
                  for w, (u, _, _) in zip(w2s, sp)]
            cbs = fit_codebooks(tensors, sp, bits, ncb, sc)

    codes, deq, scales = [], [], []
    for j, (w2, (u, ng, sh)) in enumerate(zip(w2s, sp)):
        c = np.asarray(cbs[u], np.float32)
        s = _best_scale(w2, c, grid)
        idx = _assign((w2 / s[:, None]).reshape(-1), c).astype(np.uint16)
        codes.append(idx)
        scales.append(s)
        q = (c[idx].reshape(ng, -1) * s[:, None]).reshape(sh)
        deq.append(w2.reshape(sh) if (only is not None and j not in only) else q)
    blob = (struct.pack("<B", bits)
            + b"".join(np.asarray(c, np.float16).tobytes() for c in cbs)
            + np.concatenate(scales).astype(np.float16).tobytes()
            + P.bitpack(np.concatenate(codes), bits))
    return blob, deq, cbs


def build_artifact(tensors, widths, bits, name, cb="tensor", tta=True):
    sp = spec(widths, cb)
    blob, deq, _ = encode(tensors, sp, bits)
    src = ARTIFACT_SRC.format(
        W=repr(list(map(int, widths))),
        NCB=max(s[0] for s in sp) + 1,
        U="j" if cb == "tensor" else "j%2",
        NG=int(sum(s[1] for s in sp)),
        NW=int(sum(int(np.prod(s[2])) for s in sp)),
        TTA=TTA_SRC if tta else "",
    )
    d = emit(name, {"predict.py": src.encode(), "w": blob})
    return d, deq


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def make_batches(xpad, y, bs, rng, torch):
    n = len(y)
    order = torch.randperm(n)
    ar = torch.arange(32)
    for i in range(0, n - bs + 1, bs):
        idx = order[i:i + bs]
        xb = xpad[idx]
        oi = torch.randint(0, 9, (bs,))
        oj = torch.randint(0, 9, (bs,))
        rows = (oi[:, None] + ar)[:, :, None]
        cols = (oj[:, None] + ar)[:, None, :]
        xb = xb[torch.arange(bs)[:, None, None], rows, cols]      # B,32,32,3
        f = torch.rand(bs) < 0.5
        xb[f] = xb[f].flip(2)
        yield xb.permute(0, 3, 1, 2).float() / 255.0, y[idx]


def evaluate_torch(fwd, x, bs=500, tta=True):
    import torch

    x = torch.as_tensor(x)
    out = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            z = x[i:i + bs].permute(0, 3, 1, 2).float() / 255.0
            lo = fwd(z)
            if tta:
                lo = lo + fwd(torch.flip(z, [3]))
            out.append(lo.argmax(1))
    return torch.cat(out)


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def train(widths, xtr, ytr, xva, yva, epochs, bs, lr, wd, ls, ema_decay, log):
    import torch
    import torch.nn as nn

    model = build_torch(widths)
    nparam = sum(p.numel() for p in model.parameters() if p.dim() > 1)
    log(f"  {nparam:,} conv/linear params, {sum(p.numel() for p in model.parameters()):,} total")

    xpad = torch.nn.functional.pad(
        torch.as_tensor(xtr), (0, 0, 4, 4, 4, 4))          # NHWC -> pad H,W
    yt = torch.as_tensor(ytr)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                          weight_decay=wd, nesterov=True)
    crit = nn.CrossEntropyLoss(label_smoothing=ls)
    steps = epochs * (len(ytr) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=steps, pct_start=0.15)
    ema = copy.deepcopy(model)
    for p in ema.parameters():
        p.requires_grad_(False)

    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    for ep in range(epochs):
        model.train()
        tot = seen = 0
        for xb, yb in make_batches(xpad, yt, bs, rng, torch):
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            sched.step()
            with torch.no_grad():
                for pe, pm in zip(ema.state_dict().values(),
                                  model.state_dict().values()):
                    if pe.dtype.is_floating_point:
                        pe.mul_(ema_decay).add_(pm, alpha=1 - ema_decay)
                    else:
                        pe.copy_(pm)
            tot += float(loss) * len(yb)
            seen += len(yb)
        if ep % max(1, epochs // 10) == 0 or ep == epochs - 1:
            model.eval()
            ema.eval()
            a = (evaluate_torch(model, xva) == torch.as_tensor(yva)).float().mean()
            e = (evaluate_torch(ema, xva) == torch.as_tensor(yva)).float().mean()
            log(f"  ep {ep + 1:3d}/{epochs} loss {tot / seen:.3f} "
                f"val {a * 100:.2f}% ema {e * 100:.2f}% "
                f"({time.perf_counter() - t0:.0f}s)")
    return model, ema, time.perf_counter() - t0


def qat(folded, widths, sp, bits, xtr, ytr, xva, yva, epochs, bs, lr, ls, log):
    """Fine-tune through the quantizer with a straight-through estimator.

    Post-training quantization at 3 bits costs several points; putting the
    rounding inside the training loop lets the surviving weights move to
    compensate. The codebook is refit each epoch from the current weights.
    """
    import torch

    net = Folded(folded, widths)
    # Adam, not SGD: BN is folded away by this point, so the layers' gradient
    # scales differ by orders of magnitude and a single global step size either
    # diverges on the stem or does nothing to the head. SGD at any lr tried
    # made the quantized model worse than plain PTQ.
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    crit = torch.nn.CrossEntropyLoss(label_smoothing=ls)
    xpad = torch.nn.functional.pad(torch.as_tensor(xtr), (0, 0, 4, 4, 4, 4))
    yt = torch.as_tensor(ytr)
    steps = epochs * (len(ytr) // bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    rng = np.random.default_rng(1)

    cbs = [None]
    best = (-1.0, None)

    def fake():
        cur = [p.detach().cpu().numpy() for p in net.w]
        _, deq, _ = encode(cur, sp, bits, cbs[0])
        out = []
        for p, d in zip(net.w, deq):
            t = torch.as_tensor(d, dtype=torch.float32)
            out.append(p + (t - p).detach())      # STE
        return out

    t0 = time.perf_counter()
    for ep in range(epochs):
        # refit the codebook once an epoch; nearest-centroid assignment is what
        # runs every step
        cur = [p.detach().cpu().numpy() for p in net.w]
        cbs[0] = fit_codebooks(cur, sp, bits, max(s[0] for s in sp) + 1)
        for xb, yb in make_batches(xpad, yt, bs, rng, torch):
            opt.zero_grad(set_to_none=True)
            crit(net.forward(xb, fake()), yb).backward()
            opt.step()
            sched.step()
        # Score the epoch through the *export* path — codebooks refit from the
        # current weights, exactly as build_artifact will do. Scoring against
        # the epoch's stale codebooks reported a model that was never shipped.
        cur = [p.detach().cpu().numpy() for p in net.w]
        _, deq, _ = encode(cur, sp, bits)
        a = float((evaluate_torch(Folded(deq, widths).forward,
                                  torch.as_tensor(xva))
                   == torch.as_tensor(yva)).float().mean())
        star = ""
        if a > best[0]:
            best = (a, cur)
            star = " *"
        log(f"  qat ep {ep + 1}/{epochs} val(quant) {a * 100:.2f}%{star} "
            f"({time.perf_counter() - t0:.0f}s)")
    log(f"  qat best val {best[0] * 100:.2f}%")
    return best[1], time.perf_counter() - t0


# ---------------------------------------------------------------------------
# verification: the artifact must agree with the model it came from
# ---------------------------------------------------------------------------

def verify(art_dir, deq, widths, x, n=2000, tta=True):
    import torch

    src = (Path(art_dir) / "predict.py").read_text()
    ns = {"__file__": str(Path(art_dir) / "predict.py")}
    exec(compile(src, "<artifact>", "exec"), ns)  # noqa: S102
    got = ns["predict"](x[:n])
    want = evaluate_torch(Folded(deq, widths).forward,
                          torch.as_tensor(x[:n]), tta=tta).numpy()
    agree = float((got == want).mean())
    if agree < 1.0:
        raise RuntimeError(
            f"artifact disagrees with the torch model it was exported from: "
            f"{agree:.3%} agreement on {n} images")
    return agree


# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", default="m", choices=list(ARCHS))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.2)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--ls", type=float, default=0.1)
    ap.add_argument("--ema", type=float, default=0.995)
    ap.add_argument("--bits", nargs="*", type=int, default=[4])
    ap.add_argument("--qat-epochs", type=int, default=0)
    ap.add_argument("--qat-lr", type=float, default=3e-4)
    ap.add_argument("--nval", type=int, default=5000)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--tag", default="")
    ap.add_argument("--cb", default="tensor", choices=["tensor", "shared"])
    ap.add_argument("--study", action="store_true",
                    help="report quantization variants and per-tensor sensitivity")
    ap.add_argument("--resume", action="store_true",
                    help="reuse the folded checkpoint instead of training again")
    ap.add_argument("--no-eval", action="store_true",
                    help="skip the test-set evaluation (val-only pilot)")
    a = ap.parse_args(argv)

    import torch
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)

    widths = ARCHS[a.arch]
    sp = spec(widths, a.cb)
    ckpt = REPO / "artifacts" / f"_folded-{a.arch}{a.tag}.npz"
    xtr, ytr, xte, yte = load()
    xva, yva = xtr[-a.nval:], ytr[-a.nval:]
    xfit, yfit = xtr[:-a.nval], ytr[:-a.nval]

    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    log(f"arch {a.arch} {widths}  fit on {len(yfit):,}, val {len(yva):,}")
    pick = "checkpoint"
    if a.resume and ckpt.exists():
        z = np.load(ckpt)
        folded = [z[f"t{i}"] for i in range(len(sp))]
        tsec = float(z["train_seconds"])
        log(f"  resumed folded checkpoint {ckpt.name} ({tsec:.0f}s of training)")
    else:
        model, ema, tsec = train(widths, xfit, yfit, xva, yva, a.epochs, a.bs,
                                 a.lr, a.wd, a.ls, a.ema, log)
        model.eval()
        ema.eval()
        accs = {}
        for tag, m in (("raw", model), ("ema", ema)):
            accs[tag] = float((evaluate_torch(m, torch.as_tensor(xva))
                               == torch.as_tensor(yva)).float().mean())
        pick = max(accs, key=accs.get)
        log(f"  float val: raw {accs['raw'] * 100:.2f}% "
            f"ema {accs['ema'] * 100:.2f}% -> using {pick}")
        folded = fold_bn(model if pick == "raw" else ema, widths)
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        np.savez(ckpt, train_seconds=tsec,
                 **{f"t{i}": t for i, t in enumerate(folded)})

    if a.study:
        vx, vy = torch.as_tensor(xva), torch.as_tensor(yva)

        def vacc_of(tens):
            return float((evaluate_torch(Folded(tens, widths).forward, vx)
                          == vy).float().mean()) * 100

        log(f"  float val {vacc_of(folded):.2f}%")
        names = ["c1w", "c1b"] + sum(
            [[f"dw{i}w", f"dw{i}b", f"pw{i}w", f"pw{i}b"] for i in range(1, 4)],
            []) + ["fcw", "fcb"]
        for bits in a.bits:
            for label, kw in (("naive", dict(refit=0, grid=(1.0,))),
                              ("clip", dict(refit=0)),
                              ("clip+refit", dict(refit=1))):
                _, deq, _ = encode(folded, sp, bits, **kw)
                log(f"  {bits}b {label:<11} {vacc_of(deq):.2f}%")
            _, deq, _ = encode(folded, sp, bits)
            log(f"  {bits}b per-tensor sensitivity (only that tensor quantized):")
            for j, nm in enumerate(names):
                _, d1, _ = encode(folded, sp, bits, only={j})
                log(f"    {nm:<5} {vacc_of(d1):.2f}%")
        return 0

    out = []
    for bits in a.bits:
        tens, qsec = folded, 0.0
        if a.qat_epochs:
            tens, qsec = qat(folded, widths, sp, bits, xfit, yfit, xva, yva,
                             a.qat_epochs, a.bs, a.qat_lr, a.ls, log)
        name = f"cnn{a.arch}{a.tag}-{bits}b" + ("-qat" if a.qat_epochs else "")
        d, deq = build_artifact(tens, widths, bits, name, cb=a.cb)
        vacc = float((evaluate_torch(Folded(deq, widths).forward,
                                     torch.as_tensor(xva))
                      == torch.as_tensor(yva)).float().mean())
        log(f"  {name}: quantized val {vacc * 100:.2f}%")
        verify(d, deq, widths, xte)
        log("  artifact == torch on 2,000 test images")
        if a.no_eval:
            continue
        r = evaluate(
            d, name=name,
            method=f"trained depthwise-separable CNN {widths}, BN folded, "
                   f"{bits}-bit per-channel codebook, flip TTA"
                   + (", QAT" if a.qat_epochs else ""),
            notes=f"{sum(int(np.prod(s[2])) for s in sp):,} params; "
                  f"{a.epochs} epochs on {len(yfit):,} images "
                  f"(val {vacc * 100:.2f}%); crop+flip aug, label smoothing "
                  f"{a.ls}, EMA {pick}",
            train_seconds=tsec + qsec,
        )
        log("  " + summarize(r))
        out.append(r)

    lg = REPO / "results" / "_trained_cnn_log.txt"
    with open(lg, "a") as fh:
        fh.write("\n".join([f"=== {time.ctime()} {' '.join(sys.argv[1:])}"]
                           + lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
