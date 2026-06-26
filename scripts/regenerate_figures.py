#!/usr/bin/env python3
"""Regenerate the result figures from the corrected aggregate CSVs.

The committed PNGs were rendered from the broken run (every bar showed ~3.6%).
This script redraws the accuracy-bearing figures from the recomputed CSVs so
they match the real numbers. Run ``recompute_results.py`` first.

    python scripts/regenerate_figures.py
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "outputs" / "results"
PREFIX = "gqa_15000_sst_eval"
LLAVA_VISUAL_TOKENS = 576


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def load_summary():
    rows = read_csv(RESULTS_DIR / f"{PREFIX}_summary.csv")
    for r in rows:
        for k in ("exact_accuracy", "lenient_accuracy", "avg_tokens",
                  "avg_latency_ms", "token_reduction_pct"):
            r[k] = f(r[k])
    return rows


def llava_summary():
    p = RESULTS_DIR / "llava_1000_summary.csv"
    if not p.exists():
        return None
    return read_csv(p)[0]


COLORS = ["#2C3E50", "#3498DB", "#27AE60", "#16A085", "#E74C3C",
          "#E67E22", "#95A5A6", "#8E44AD"]


def fig_accuracy_vs_tokens(summary):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, xcol, xlabel in [
        (axes[0], "avg_tokens", "Average input tokens"),
        (axes[1], "token_reduction_pct", "Token reduction vs Full SST (%)"),
    ]:
        for row, color in zip(summary, COLORS):
            ax.scatter(row[xcol], row["exact_accuracy"] * 100, color=color, s=120, zorder=5)
            ax.annotate(row["method_name"], (row[xcol], row["exact_accuracy"] * 100),
                        textcoords="offset points", xytext=(6, 4), fontsize=7)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("Exact accuracy (%)", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Accuracy vs {xlabel}", fontsize=12)
    plt.tight_layout()
    out = RESULTS_DIR / f"{PREFIX}_accuracy_vs_tokens.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_tradeoff_with_llava(summary, llava):
    fig, ax = plt.subplots(figsize=(12, 6))
    for row, color in zip(summary, COLORS):
        ax.scatter(row["avg_tokens"], row["exact_accuracy"] * 100, color=color,
                   s=160, zorder=5, edgecolors="white", linewidths=1.5)
        ax.annotate(row["method_name"], (row["avg_tokens"], row["exact_accuracy"] * 100),
                    textcoords="offset points", xytext=(8, 4), fontsize=8.5,
                    color=color, fontweight="bold")
    if llava:
        lacc = f(llava["exact_accuracy"]) * 100
        ax.scatter(LLAVA_VISUAL_TOKENS, lacc, color="#E74C3C", s=220, marker="*",
                   zorder=6, edgecolors="black", linewidths=1.0, label="LLaVA (image)")
        ax.axhline(lacc, color="#E74C3C", linestyle="--", linewidth=1.2, alpha=0.6)
        ax.legend(fontsize=10)
    ax.set_xlabel("Average input tokens", fontsize=11)
    ax.set_ylabel("Exact accuracy (%)", fontsize=11)
    ax.set_title("Accuracy vs token cost — SST variants and LLaVA reference", fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = RESULTS_DIR / f"{PREFIX}_tradeoff_with_llava.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_token_vs_llava(summary, llava):
    rows = [("LLaVA (image)", LLAVA_VISUAL_TOKENS,
             f(llava["exact_accuracy"]) * 100 if llava else float("nan"))]
    for row in summary:
        rows.append((row["method_name"], row["avg_tokens"], row["exact_accuracy"] * 100))
    labels = [r[0] for r in rows]
    toks = [r[1] for r in rows]
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#E74C3C"] + ["#4A90D9"] * (len(rows) - 1)
    y = range(len(rows))
    ax.barh(list(y), toks, color=colors)
    for i, (lbl, t, acc) in enumerate(rows):
        ax.text(t + 5, i, f"{t:.0f} tok | {acc:.1f}%", va="center", fontsize=8)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Average input tokens", fontsize=11)
    ax.set_title("Input token budget: SST text vs LLaVA visual tokens", fontsize=12)
    plt.tight_layout()
    out = RESULTS_DIR / f"{PREFIX}_token_vs_llava.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_semantic_heatmap():
    rows = read_csv(RESULTS_DIR / f"{PREFIX}_per_type_analysis.csv")
    if not rows:
        return None
    types = [c for c in rows[0].keys() if c != "method_name"]
    methods = [r["method_name"] for r in rows]
    data = [[f(r[t]) for t in types] for r in rows]
    fig, ax = plt.subplots(figsize=(max(10, len(types) * 1.2), len(methods) * 0.7 + 1))
    im = ax.imshow(data, cmap="YlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(types)))
    ax.set_xticklabels(types, rotation=30, ha="right")
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)
    plt.colorbar(im, ax=ax, label="Exact accuracy")
    ax.set_title("Exact accuracy by method and semantic type", fontsize=12)
    for i in range(len(methods)):
        for j in range(len(types)):
            v = data[i][j]
            if not math.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=8, color="black" if v < 0.7 else "white")
    plt.tight_layout()
    out = RESULTS_DIR / f"{PREFIX}_semantic_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_latency_compression_floor(summary, llava):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Latency, Compression Ratio, and Performance Floor", fontsize=13, y=1.02)
    names = [r["method_name"] for r in summary]

    ax = axes[0]
    lats = [r["avg_latency_ms"] for r in summary]
    ax.bar(range(len(names)), lats, color=COLORS[:len(names)], edgecolor="white")
    if llava:
        llat = f(llava["avg_latency_sec"]) * 1000
        ax.axhline(llat, color="red", linestyle="--", linewidth=1.8, label=f"LLaVA ({llat:.0f} ms)")
        ax.legend(fontsize=9)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Avg latency (ms)")
    ax.set_title("Inference latency per method")

    ax = axes[1]
    ratios = [LLAVA_VISUAL_TOKENS / r["avg_tokens"] if r["avg_tokens"] else 0 for r in summary]
    ax.bar(range(len(names)), ratios, color=COLORS[:len(names)], edgecolor="white")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Compression ratio vs LLaVA (×)")
    ax.set_title("Token compression vs LLaVA (576 visual tokens)")

    ax = axes[2]
    accs = [r["exact_accuracy"] * 100 for r in summary]
    ax.bar(range(len(names)), accs, color=COLORS[:len(names)], edgecolor="white")
    if llava:
        ax.axhline(f(llava["exact_accuracy"]) * 100, color="red", linestyle="--",
                   linewidth=1.8, label="LLaVA accuracy")
        ax.legend(fontsize=9)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Exact accuracy (%)")
    ax.set_title("Accuracy vs LLaVA reference")

    plt.tight_layout()
    out = RESULTS_DIR / f"{PREFIX}_latency_compression_floor.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    summary = load_summary()
    llava = llava_summary()
    outs = [
        fig_accuracy_vs_tokens(summary),
        fig_tradeoff_with_llava(summary, llava),
        fig_token_vs_llava(summary, llava),
        fig_semantic_heatmap(),
        fig_latency_compression_floor(summary, llava),
    ]
    for o in outs:
        if o:
            print("  wrote", o.relative_to(ROOT))
    print("Figures regenerated from corrected CSVs.")


if __name__ == "__main__":
    main()
