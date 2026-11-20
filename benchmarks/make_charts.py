"""Draw the README charts from benchmarks/results/*.json.

usage: python benchmarks/make_charts.py
writes assets/injection_chart.png and assets/pii_chart.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"
ASSETS = ROOT / "assets"

# Fixed categorical order (blue, orange, aqua); one colour per detector in every panel.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _bars(ax: plt.Axes, groups: list[str], series: dict[str, list[float]], *, label_values: bool) -> None:
    n = len(series)
    width = 0.8 / n
    for i, (name, values) in enumerate(series.items()):
        xs = [g + (i - (n - 1) / 2) * width for g in range(len(groups))]
        bars = ax.bar(xs, values, width=width - 0.03, color=SERIES[i], label=name, edgecolor="white", linewidth=1)
        if label_values:
            for bar, value in zip(bars, values, strict=True):
                ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.2f}", ha="center", va="bottom",
                        fontsize=7.5, color=INK)
    ax.set_xticks(range(len(groups)), groups)
    ax.set_ylim(0, 1.08)


def injection_chart() -> Path:
    data = json.loads((RESULTS / "injection.json").read_text(encoding="utf-8"))
    names = ["rules", "classifier (bge-small)", "rules + classifier (bge-small)"]
    names = [n for n in names if n in data["deepset_test"]] or list(data["deepset_test"])[:3]
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1, 2.3]},
                                      facecolor="white")
    metrics = ["precision", "recall", "f1"]
    _bars(left, ["Precision", "Recall", "F1"],
          {n: [data["deepset_test"][n][m] for m in metrics] for n in names}, label_values=True)
    rows = data["dataset"]["rows"]
    left.set_title(f"deepset/prompt-injections test split ({rows} texts)", fontsize=10.5, color=INK, loc="left")
    suite = data["agentic_suite"]["results"]
    transforms = list(suite[names[0]]["recall_by_transform"])
    _bars(right, transforms, {n: [suite[n]["recall_by_transform"][t] for t in transforms] for n in names},
          label_values=False)
    right.tick_params(axis="x", labelrotation=25)
    short = {names[0]: "rules", names[1]: "classifier", names[2]: "combined"}
    fprs = " · ".join(f"{short[n]} {suite[n]['benign_false_positive_rate']:.0%}" for n in names)
    right.set_title("Synthetic agentic suite: recall per obfuscation\n"
                    f"false-positive rate on 40 benign agent messages: {fprs}", fontsize=10.5, color=INK, loc="left")
    for ax in (left, right):
        _style(ax)
    handles, labels = left.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9.5, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    out = ASSETS / "injection_chart.png"
    fig.savefig(out, dpi=130, facecolor="white")
    plt.close(fig)
    return out


def pii_chart() -> Path:
    data = json.loads((RESULTS / "pii.json").read_text(encoding="utf-8"))
    per_type = data["results"]["regex + validators"]["per_type"]
    classes = list(per_type)
    fig, ax = plt.subplots(figsize=(10, 4.2), facecolor="white")
    _bars(ax, classes, {"precision": [per_type[c]["precision"] for c in classes],
                        "recall": [per_type[c]["recall"] for c in classes]}, label_values=True)
    corpus = data["corpus"]
    ax.set_title(f"PII detection per class, regex + validators (synthetic: {corpus['sentences']} sentences, "
                 f"{corpus['italian_share']:.0%} Italian, {corpus['hard_negatives']} hard negatives)\n"
                 f"recall misses are the {corpus['out_of_scope_variants']} values written in spellings the "
                 "detectors do not target",
                 fontsize=10.5, color=INK, loc="left")
    _style(ax)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=9.5, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    out = ASSETS / "pii_chart.png"
    fig.savefig(out, dpi=130, facecolor="white")
    plt.close(fig)
    return out


if __name__ == "__main__":
    for path in (injection_chart(), pii_chart()):
        print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size // 1024} KB)")
