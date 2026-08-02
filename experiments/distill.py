"""Distillation into the small end. Zero artifact bytes, so the only cost is time.

TTA buys accuracy and pays for it in source bytes. Distillation is the lever
that does not: it changes what the student is trained against and leaves the
artifact's shape, its weight count and its decoder untouched. Whatever it is
worth, it is worth for free.

The teacher is the `xl` network already on disk — `artifacts/_folded-xlqat.npz`,
85.50% float on val, the best model this rig has trained. It is never shipped.

Two students are asked, because they are different questions:

*The trained CNN.* A soft-label objective replaces the hard-label one directly.
The complication is that the student trains on random crops, so the teacher has
to be asked about the *same* crop or its logits describe a different image. Doing
that live costs an `xl` forward pass per step, which is most of the training
budget. Instead K augmented views per image are drawn once, the teacher scores
them once, and training samples from that cache. The hard-label twin is trained
through the identical loop over the identical cache, so the only difference
between the two runs is the loss — which is the comparison the falsifier needs.

*The closed-form ridge head.* Ridge fits a linear map to targets, and soft
targets are still targets, so the one-hot matrix is simply replaced by the
teacher's probabilities. It is one extra solve and costs nothing to ask.

Everything is selected on `tinycifar.data.load_dev()`. Test is scored once per
exported artifact through `tinycifar.evaluate`.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("MKL_NUM_THREADS", "3")

from experiments import tta_ridge as TR  # noqa: E402
from experiments.conv_features import geometry  # noqa: E402
from experiments.trained_cnn import (  # noqa: E402
    ARCHS, Folded, build_artifact, build_torch, evaluate_torch, fold_bn, qat,
    spec, verify,
)
from experiments.tta_cnn import mcnemar  # noqa: E402
from tinycifar import artifact as A  # noqa: E402
from tinycifar.data import load, load_dev  # noqa: E402
from tinycifar.evaluate import evaluate, summarize  # noqa: E402

TEACHER = REPO / "artifacts" / "_folded-xlqat.npz"


def load_teacher(path=TEACHER, arch="xl"):
    z = np.load(path)
    widths = ARCHS[arch]
    tens = [z[f"t{i}"] for i in range(len(spec(widths)))]
    return Folded(tens, widths), widths


# ---------------------------------------------------------------------------
# the augmentation cache: K views per image, scored by the teacher once
# ---------------------------------------------------------------------------

def draw_views(n, k, seed=7):
    """K crop offsets and flips per image, fixed for the whole run."""
    g = np.random.default_rng(seed)
    oi = g.integers(0, 9, (k, n)).astype(np.int8)
    oj = g.integers(0, 9, (k, n)).astype(np.int8)
    fl = (g.random((k, n)) < 0.5)
    return oi, oj, fl


def apply_view(xpad, idx, oi, oj, fl, torch):
    """Crop 32x32 out of the 40x40 padded image at the stored offset, flip."""
    ar = torch.arange(32)
    rows = (torch.as_tensor(oi.astype(np.int64))[:, None] + ar)[:, :, None]
    cols = (torch.as_tensor(oj.astype(np.int64))[:, None] + ar)[:, None, :]
    xb = xpad[idx][torch.arange(len(idx))[:, None, None], rows, cols]
    f = torch.as_tensor(fl)
    xb[f] = xb[f].flip(2)
    return xb.permute(0, 3, 1, 2).float() / 255.0


def teacher_cache(teacher, x, k, bs, log, seed=7):
    """[K, N, 10] teacher logits on the cached views."""
    import torch

    n = len(x)
    oi, oj, fl = draw_views(n, k, seed)
    xpad = torch.nn.functional.pad(torch.as_tensor(x), (0, 0, 4, 4, 4, 4))
    out = np.empty((k, n, 10), np.float16)
    t0 = time.perf_counter()
    with torch.no_grad():
        for v in range(k):
            for i in range(0, n, bs):
                idx = torch.arange(i, min(i + bs, n))
                xb = apply_view(xpad, idx, oi[v, i:i + bs], oj[v, i:i + bs],
                                fl[v, i:i + bs], torch)
                out[v, i:i + bs] = teacher.forward(xb).numpy().astype(np.float16)
            log(f"  teacher view {v + 1}/{k} ({time.perf_counter() - t0:.0f}s)")
    return out, (oi, oj, fl)


# ---------------------------------------------------------------------------
# student training over the cache
# ---------------------------------------------------------------------------

def train_kd(widths, x, y, xva, yva, tlog, views, epochs, bs, lr, wd, ls,
             ema_decay, alpha, temp, log):
    """Identical to trained_cnn.train except the batch source and the loss.

    `alpha=0` is the hard-label twin: same cache, same order, same schedule.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    oi, oj, fl = views
    k, n = oi.shape
    model = build_torch(widths)
    log(f"  {sum(p.numel() for p in model.parameters()):,} params, "
        f"alpha={alpha} T={temp}")

    xpad = torch.nn.functional.pad(torch.as_tensor(x), (0, 0, 4, 4, 4, 4))
    yt = torch.as_tensor(y)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                          weight_decay=wd, nesterov=True)
    crit = nn.CrossEntropyLoss(label_smoothing=ls)
    steps = epochs * (n // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=steps, pct_start=0.15)
    ema = copy.deepcopy(model)
    for p in ema.parameters():
        p.requires_grad_(False)

    g = np.random.default_rng(11)
    t0 = time.perf_counter()
    for ep in range(epochs):
        model.train()
        order = torch.randperm(n)
        vsel = g.integers(0, k, n)
        tot = seen = 0
        for i in range(0, n - bs + 1, bs):
            idx = order[i:i + bs]
            ii = idx.numpy()
            v = vsel[ii]
            xb = apply_view(xpad, idx, oi[v, ii], oj[v, ii], fl[v, ii], torch)
            out = model(xb)
            loss = crit(out, yt[idx])
            if alpha:
                tl = torch.as_tensor(tlog[v, ii].astype(np.float32))
                kd = F.kl_div(F.log_softmax(out / temp, 1),
                              F.log_softmax(tl / temp, 1),
                              reduction="batchmean", log_target=True)
                loss = (1 - alpha) * loss + alpha * temp * temp * kd
            opt.zero_grad(set_to_none=True)
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
            tot += float(loss) * bs
            seen += bs
        if ep % max(1, epochs // 10) == 0 or ep == epochs - 1:
            model.eval()
            ema.eval()
            a = (evaluate_torch(model, xva) == torch.as_tensor(yva)).float().mean()
            e = (evaluate_torch(ema, xva) == torch.as_tensor(yva)).float().mean()
            log(f"  ep {ep + 1:3d}/{epochs} loss {tot / seen:.3f} "
                f"val {a * 100:.2f}% ema {e * 100:.2f}% "
                f"({time.perf_counter() - t0:.0f}s)")
    return model, ema, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# the ridge question: soft targets into a closed-form solve
# ---------------------------------------------------------------------------

def teacher_logits(teacher, x, bs=500):
    """Flip-averaged teacher logits on unaugmented images."""
    import torch

    out = np.empty((len(x), 10), np.float32)
    with torch.no_grad():
        for i in range(0, len(x), bs):
            z = torch.as_tensor(x[i:i + bs]).permute(0, 3, 1, 2).float() / 255
            lo = teacher.forward(z) + teacher.forward(torch.flip(z, [3]))
            out[i:i + bs] = (lo / 2).numpy()
    return out


def softmax_t(lo, temp):
    e = np.exp((lo - lo.max(1, keepdims=True)) / temp)
    return e / e.sum(1, keepdims=True)


def ridge_run(k, bits, patch, stride, pool, seed, lams, feats, G, Hy, Fva,
              yva, tag):
    """Solve the golfed conv-ridge head against pre-accumulated targets.

    Identical to `golf.py`'s flip-TTA configuration in every respect except the
    right-hand side, so any difference is the targets and nothing else. Lambda
    is chosen on validation, as it is for the hard-label arm.
    """
    best = None
    t0 = time.perf_counter()
    for lam in lams:
        W = TR.solve(G, Hy, lam)
        files, Wq = TR.pack(W, bits, k, patch, stride, pool, seed,
                            [(0, 0)], True, "wrap")
        ok = np.argmax(Fva @ Wq, 1) == yva
        n = A.measure(files).description_length
        if best is None or ok.mean() > best["val"]:
            best = dict(tag=tag, k=k, bits=bits, val=float(ok.mean()),
                        bytes=n, lam=lam, ok=ok, files=files)
    print(f"  ridge k={k} {bits}b  {tag:<22} val {best['val'] * 100:6.2f}%  "
          f"{best['bytes']:,} B  (lam {best['lam']:g}, "
          f"{time.perf_counter() - t0:.0f}s)", flush=True)
    return best


# ---------------------------------------------------------------------------


def compare(names, xva, yva):
    """Paired val comparison between two shipped artifacts.

    The 1.06 pp floor in the docs prices the *maximum* of an unpaired sweep. A
    distilled student and its twin are scored on the same 5,000 images, so the
    paired disagreement counts say more than the difference of two means; both
    are printed and neither is allowed to stand in for the other.
    """
    from experiments.tta_cnn import load_fw, view_logits

    oks = []
    for nm in names:
        art = REPO / "artifacts" / nm
        lo = view_logits(load_fw(art), xva, (0, 0), False)
        lo = lo + view_logits(load_fw(art), xva, (0, 0), True)
        ok = lo.argmax(1) == yva
        oks.append(ok)
        print(f"  {nm:<28} val {ok.mean() * 100:6.2f}%  "
              f"{A.measure(A.read_dir(art)).description_length:,} B")
    if len(oks) == 2:
        from experiments.tta_cnn import mcnemar
        lo_, wi_, p = mcnemar(oks[0], oks[1])
        print(f"  {names[1]} vs {names[0]}: "
              f"{(oks[1].mean() - oks[0].mean()) * 100:+.2f} pp, "
              f"paired +{wi_}/-{lo_}, exact McNemar p={p:.3g}")
    return oks


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", default="cnn", choices=["cnn", "ridge", "compare"])
    ap.add_argument("--compare", nargs="*", default=[])
    ap.add_argument("--arch", default="t")
    ap.add_argument("--tag", default="kd")
    ap.add_argument("--views", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.2)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--ls", type=float, default=0.1)
    ap.add_argument("--ema", type=float, default=0.995)
    ap.add_argument("--alpha", type=float, default=0.9)
    ap.add_argument("--temp", type=float, default=4.0)
    ap.add_argument("--bits", nargs="*", type=int, default=[4])
    ap.add_argument("--qat-epochs", type=int, default=8)
    ap.add_argument("--qat-lr", type=float, default=3e-4)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--k", type=int, default=8, help="ridge mode: filter count")
    ap.add_argument("--lams", nargs="*", type=float,
                    default=[25.0, 1e2, 4e2, 1.6e3, 6.4e3])
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)

    import torch
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)

    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    xfit, yfit, xva, yva = load_dev()
    if a.mode == "compare":
        compare(a.compare, xva, yva)
        return 0

    teacher, twidths = load_teacher()
    log(f"teacher xl from {TEACHER.name}")

    if a.mode == "ridge":
        tacc = float((evaluate_torch(teacher.forward, xva)
                      == torch.as_tensor(yva)).float().mean())
        log(f"  teacher val {tacc * 100:.2f}%")
        rows = []
        Yhard = np.zeros((len(yfit), 10))
        Yhard[np.arange(len(yfit)), yfit] = 1.0
        tl = teacher_logits(teacher, xfit)
        log(f"  teacher logits on fit split: {tl.shape}")

        # The Gram matrix does not depend on the targets, so it is accumulated
        # once and every target set is a solve. Only Hy changes.
        feats = TR.make_feats(a.k, 4, 2, 5, 1)
        dim = a.k * geometry(4, 2, 5)[1] ** 2
        sh, fl, md = [(0, 0)], True, "wrap"
        Ffit = np.concatenate([TR.design(feats, xfit, sh, fl, md, i, i + 2000)
                               for i in range(0, len(xfit), 2000)])
        Fva = np.concatenate([TR.design(feats, xva, sh, fl, md, i, i + 500)
                              for i in range(0, len(xva), 500)])
        G = Ffit.T @ Ffit
        log(f"  gram {G.shape} accumulated once; dim {dim}")

        def run(Y, tag, bits):
            return ridge_run(a.k, bits, 4, 2, 5, 1, a.lams, feats, G,
                             Ffit.T @ Y, Fva, yva, tag)

        for bits in a.bits:
            rows.append(run(Yhard, "hard one-hot", bits))
            for T in (0.5, 1.0, 2.0, 4.0):
                Ys = softmax_t(tl, T)
                rows.append(run(Ys, f"teacher soft T={T:g}", bits))
                rows.append(run(0.5 * (Ys + Yhard), f"half soft T={T:g}", bits))
        ref = rows[0]
        print("\nagainst the hard-label arm at matched bytes:")
        for r in rows[1:]:
            lo_, wi_, p = mcnemar(ref["ok"], r["ok"])
            print(f"  {r['tag']:<22} {(r['val'] - ref['val']) * 100:+6.2f} pp  "
                  f"{r['bytes'] - ref['bytes']:+4d} B  paired +{wi_}/-{lo_} "
                  f"p={p:.3g}")
        if a.out:
            Path(a.out).write_text(json.dumps(
                [{k: v for k, v in r.items() if k not in ("files", "ok")}
                 for r in rows], indent=2))
        return 0

    widths = ARCHS[a.arch]
    sp = spec(widths)
    # The cache is the expensive part and is identical for every student, so
    # the hard-label twin and the distilled run share it byte for byte. That is
    # what makes the comparison paired: same crops, same order, different loss.
    cpath = REPO / "artifacts" / f"_teacher-xl-v{a.views}.npz"
    if cpath.exists():
        z = np.load(cpath)
        tlog = z["logits"]
        views = (z["oi"], z["oj"], z["fl"])
        log(f"  reused teacher cache {cpath.name} {tlog.shape}")
    else:
        tlog, views = teacher_cache(teacher, xfit, a.views, 500, log)
        np.savez(cpath, logits=tlog, oi=views[0], oj=views[1], fl=views[2])
    tv = float((evaluate_torch(teacher.forward, xva)
                == torch.as_tensor(yva)).float().mean())
    log(f"  teacher val {tv * 100:.2f}%; cache {tlog.shape}")

    ckpt = REPO / "artifacts" / f"_folded-{a.arch}{a.tag}.npz"
    model, ema, tsec = train_kd(widths, xfit, yfit, xva, yva, tlog, views,
                                a.epochs, a.bs, a.lr, a.wd, a.ls, a.ema,
                                a.alpha, a.temp, log)
    model.eval()
    ema.eval()
    accs = {t: float((evaluate_torch(m, torch.as_tensor(xva))
                      == torch.as_tensor(yva)).float().mean())
            for t, m in (("raw", model), ("ema", ema))}
    pick = max(accs, key=accs.get)
    log(f"  float val: raw {accs['raw'] * 100:.2f}% ema {accs['ema'] * 100:.2f}%"
        f" -> using {pick}")
    folded = fold_bn(model if pick == "raw" else ema, widths)
    np.savez(ckpt, train_seconds=tsec,
             **{f"t{i}": t for i, t in enumerate(folded)})

    _, _, xte, _ = load()
    for bits in a.bits:
        tens, cbs, qsec = folded, None, 0.0
        if a.qat_epochs:
            tens, cbs, qsec = qat(folded, widths, sp, bits, xfit, yfit, xva,
                                  yva, a.qat_epochs, a.bs, a.qat_lr, a.ls, log)
        name = f"cnn{a.arch}{a.tag}-{bits}b-qat" if a.qat_epochs \
            else f"cnn{a.arch}{a.tag}-{bits}b"
        d, deq = build_artifact(tens, widths, bits, name, cbs=cbs)
        vacc = float((evaluate_torch(Folded(deq, widths).forward,
                                     torch.as_tensor(xva))
                      == torch.as_tensor(yva)).float().mean())
        log(f"  {name}: quantized val {vacc * 100:.2f}%  "
            f"({A.measure(A.read_dir(d)).description_length:,} B)")
        verify(d, deq, widths, xte)
        if a.no_eval:
            continue
        r = evaluate(
            d, name=name,
            method=f"distilled depthwise-separable CNN {widths} (teacher xl), "
                   f"BN folded, {bits}-bit codebook, flip TTA, QAT",
            notes=f"alpha={a.alpha} T={a.temp}, {a.views} cached teacher views; "
                  f"{a.epochs} epochs on {len(yfit):,} images (val "
                  f"{vacc * 100:.2f}%); EMA {pick}",
            train_seconds=tsec + qsec)
        log("  " + summarize(r))

    lg = REPO / "results" / "_distill_log.txt"
    with open(lg, "a") as fh:
        fh.write("\n".join([f"=== {time.ctime()} {' '.join(sys.argv[1:])}"]
                           + lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
