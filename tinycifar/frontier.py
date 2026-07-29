"""Build the combined size-accuracy frontier: our artifacts against the world.

Cross-paper byte comparisons are treacherous — published work reports parameter
counts far more often than artifact bytes, and when it does report bytes it
rarely means the same thing we do. So every external point carries where its
byte figure came from, and derived figures are marked as derived rather than
quietly mixed in with reported ones.

Reads `docs/sota.json` for external points and `results/*.json` for ours.
Writes `docs/frontier.png` and prints the README table.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SOTA = REPO / "docs" / "sota.json"
PLOT = REPO / "docs" / "frontier.png"


def load_ours() -> list[dict]:
    from .leaderboard import load_results, pareto

    return [
        {
            "name": r["name"],
            "accuracy": r["accuracy"] * 100,
            "bytes": r["description_length"],
            "ours": True,
            "note": r.get("method", ""),
        }
        for r in pareto(load_results())
    ]


def load_sota() -> list[dict]:
    if not SOTA.exists():
        return []
    rows = json.loads(SOTA.read_text())
    return [dict(r, ours=False) for r in rows]


def pareto_front(points: list[dict]) -> list[dict]:
    """Points beaten by nothing smaller-or-equal with at least their accuracy."""
    out = []
    for p in points:
        if not any(
            o is not p
            and o["bytes"] <= p["bytes"]
            and o["accuracy"] >= p["accuracy"]
            and (o["bytes"], -o["accuracy"]) < (p["bytes"], -p["accuracy"])
            for o in points
        ):
            out.append(p)
    return sorted(out, key=lambda p: p["bytes"])


def fmt_bytes(n: float) -> str:
    if n < 1024:
        return f"{n:,.0f} B"
    if n < 1024 ** 2:
        return f"{n / 1024:.1f} KB"
    return f"{n / 1024 ** 2:.1f} MB"


def plot(ours: list[dict], sota: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=160)

    if sota:
        ax.scatter([p["bytes"] for p in sota], [p["accuracy"] for p in sota],
                   s=34, c="#888", marker="s", label="published", zorder=3)
        sf = pareto_front(sota)
        ax.plot([p["bytes"] for p in sf], [p["accuracy"] for p in sf],
                c="#bbb", lw=1.0, ls="--", zorder=2)
        for p in sf:
            ax.annotate(p["name"], (p["bytes"], p["accuracy"]),
                        textcoords="offset points", xytext=(5, -9),
                        fontsize=6.5, color="#666")

    of = pareto_front(ours)
    ax.plot([p["bytes"] for p in of], [p["accuracy"] for p in of],
            c="#c1272d", lw=1.6, marker="o", ms=4.5,
            label="this repo (Pareto frontier)", zorder=5)

    ax.axhline(10, c="#ddd", lw=0.8, zorder=1)
    ax.annotate("chance (10 classes)", (62, 11.4), fontsize=6.5, color="#999")

    ax.set_xscale("log")
    ax.set_xlabel("artifact size (bytes, log scale)")
    ax.set_ylabel("CIFAR-10 test accuracy (%)")
    ax.set_title("CIFAR-10: accuracy vs. artifact size", fontsize=11)
    ax.grid(alpha=0.18, lw=0.6)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.set_ylim(0, 100)
    allb = [p["bytes"] for p in ours + sota]
    ax.set_xlim(min(allb) * 0.55, max(allb) * 6)   # headroom for point labels
    fig.tight_layout()
    PLOT.parent.mkdir(exist_ok=True)
    fig.savefig(PLOT)
    plt.close(fig)


def curate(ours: list[dict]) -> list[dict]:
    """Best of ours per size octave. The full frontier is 28 points, which is a
    leaderboard, not a README table."""
    picked = {}
    import math
    for p in ours:
        oct_ = math.floor(math.log2(max(p["bytes"], 1)))
        if oct_ not in picked or p["accuracy"] > picked[oct_]["accuracy"]:
            picked[oct_] = p
    return sorted(picked.values(), key=lambda p: p["bytes"])


def table(ours: list[dict], sota: list[dict]) -> str:
    rows = sorted(curate(ours) + sota, key=lambda p: p["bytes"])
    front = {id(p) for p in pareto_front(rows)}

    L = ["| | model | size | accuracy | source |", "|---|---|---:|---:|---|"]
    for p in rows:
        mark = "**◆**" if id(p) in front else ""
        who = "**this repo**" if p["ours"] else p.get("source", "")
        name = f"`{p['name']}`" if p["ours"] else p["name"]
        L.append(f"| {mark} | {name} | {fmt_bytes(p['bytes'])} | "
                 f"{p['accuracy']:.2f}% | {who} |")
    return "\n".join(L)


README = REPO / "README.md"
START, END = "<!-- FRONTIER:START -->", "<!-- FRONTIER:END -->"


def write_readme(body: str) -> bool:
    """Replace the generated block in README.md, leaving the prose alone."""
    if not README.exists():
        return False
    t = README.read_text()
    if START not in t or END not in t:
        return False
    head, rest = t.split(START, 1)
    _, tail = rest.split(END, 1)
    README.write_text(f"{head}{START}\n{body}\n{END}{tail}")
    return True


def main() -> int:
    ours, sota = load_ours(), load_sota()
    if not sota:
        print(f"note: {SOTA} missing — plotting our points only")
    plot(ours, sota)
    print(f"wrote {PLOT}  ({len(ours)} ours, {len(sota)} published)\n")
    body = table(ours, sota)
    if write_readme(body):
        print("updated README.md frontier block")
    print(body)
    print(f"\n({len(ours)} frontier points ours, {len(curate(ours))} shown)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
