"""Graphviz tree rendering and Matplotlib charts."""
from __future__ import annotations

import itertools

import graphviz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2",
           "#FF9DA6", "#9D755D", "#BAB0AC", "#EECA3B"]


def _blend(hex_color: str, weight: float) -> str:
    """Blend a colour with white (weight=1 -> full colour)."""
    rgb = [int(hex_color[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{int(255 - (255 - c) * weight):02x}" for c in rgb)


def tree_to_graphviz(tree, max_display_depth: int | None = None) -> graphviz.Digraph:
    dot = graphviz.Digraph(name=tree.name)
    dot.attr(rankdir="TB", fontname="Helvetica")
    dot.attr("node", shape="box", style="rounded,filled", fontname="Helvetica", fontsize="11")
    dot.attr("edge", fontname="Helvetica", fontsize="10")
    colors = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(tree.classes_)}
    ids = itertools.count()

    def add(node) -> str:
        nid = f"n{next(ids)}"
        purity = max(node.class_counts.values()) / max(node.n_samples, 1)
        fill = _blend(colors[node.prediction], 0.25 + 0.6 * purity)
        counts = ", ".join(f"{k}: {v}" for k, v in node.class_counts.items() if v)
        truncated = (max_display_depth is not None and node.depth >= max_display_depth
                     and not node.is_leaf)
        if node.is_leaf or truncated:
            head = f"class = {node.prediction}" if node.is_leaf else f"… subtree (majority: {node.prediction})"
            dot.node(nid, f"{head}\\nsamples = {node.n_samples}\\n[{counts}]", fillcolor=fill,
                     style="rounded,filled" + ("" if node.is_leaf else ",dashed"))
            return nid
        label = f"{node.feature}\\ngain = {node.gain:.3f}"
        if node.gain_ratio is not None:
            label += f"\\ngain ratio = {node.gain_ratio:.3f}"
        label += f"\\nsamples = {node.n_samples}"
        dot.node(nid, label, shape="box", style="filled", fillcolor="#F2F2F2")
        for key, child in node.children.items():
            cid = add(child)
            dot.edge(nid, cid, label=" " + node.branch_label(key)[:30])
        return nid

    add(tree.root_)
    return dot


def fig_confusion_matrix(cm: np.ndarray, labels: list, title: str):
    fig, ax = plt.subplots(figsize=(4.2 + 0.3 * len(labels), 3.8 + 0.25 * len(labels)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)
    thresh = cm.max() / 2 if cm.max() else 0.5
    for i, j in np.ndindex(cm.shape):
        ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def fig_metric_bars(table: pd.DataFrame):
    metrics = ["Accuracy", "Precision", "Recall", "F1-score"]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    x, w = np.arange(len(metrics)), 0.8 / max(len(table), 1)
    for i, (name, row) in enumerate(table.iterrows()):
        bars = ax.bar(x + i * w - 0.4 + w / 2, [row[m] for m in metrics], w,
                      label=name, color=PALETTE[i])
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    ax.set_xticks(x, labels=metrics); ax.set_ylim(0, 1.12); ax.set_ylabel("Score")
    ax.set_title("ID3 vs C4.5 on the test set"); ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def fig_importance(series: pd.Series, title: str):
    s = series.sort_values()
    fig, ax = plt.subplots(figsize=(5, max(2, 0.35 * len(s) + 1)))
    ax.barh(s.index.astype(str), s.values, color=PALETTE[0])
    ax.set_title(title); ax.set_xlabel("Relative importance")
    fig.tight_layout()
    return fig
