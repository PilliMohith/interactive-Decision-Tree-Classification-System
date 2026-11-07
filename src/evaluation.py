"""Metrics and comparison tables."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)


def evaluate_model(y_true, y_pred, labels: list, average: str = "macro") -> dict:
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=average, zero_division=0)
    pc, rc, fc, sc = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(p), "recall": float(r), "f1": float(f),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
        "per_class": pd.DataFrame(
            {"precision": pc, "recall": rc, "f1-score": fc, "support": sc}, index=labels),
        "average": average,
    }


def comparison_table(results: dict) -> pd.DataFrame:
    rows = {}
    for name, r in results.items():
        m, t = r["metrics"], r["model"]
        rows[name] = {
            "Accuracy": m["accuracy"], "Precision": m["precision"], "Recall": m["recall"],
            "F1-score": m["f1"], "Train time (s)": r["fit_time"],
            "Depth": t.get_depth(), "Nodes": t.get_n_nodes(), "Leaves": t.get_n_leaves(),
        }
    return pd.DataFrame(rows).T
