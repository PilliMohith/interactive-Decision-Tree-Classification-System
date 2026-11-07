"""ID3 and C4.5 decision-tree classifiers implemented with NumPy / Pandas.

ID3   - categorical attributes only, multiway splits, information gain.
C4.5  - categorical + continuous attributes, gain ratio (with the average-gain
        heuristic), binary threshold splits for numeric data and optional
        pessimistic error-based pruning.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np
import pandas as pd

EPS = 1e-12
# z-score for a 25% confidence factor (C4.5 default) used in pessimistic pruning
Z_CF25 = 0.6745


# --------------------------------------------------------------------------- #
# Information-theory helpers
# --------------------------------------------------------------------------- #
def entropy(counts) -> float:
    """Shannon entropy (base 2) of a vector of class counts."""
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())


def _entropy_rows(counts: np.ndarray) -> np.ndarray:
    """Row-wise entropy of a 2-D array of class counts."""
    counts = np.asarray(counts, dtype=float)
    total = counts.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(total > 0, counts / total, 0.0)
        logp = np.where(p > 0, np.log2(p), 0.0)
    return -(p * logp).sum(axis=1)


def _pessimistic_errors(n: float, errors: float, z: float = Z_CF25) -> float:
    """Upper confidence bound on the number of errors (C4.5 error estimate)."""
    if n <= 0:
        return 0.0
    f = errors / n
    upper = (f + z * z / (2 * n) + z * np.sqrt(f / n - f * f / n + z * z / (4 * n * n))) / (1 + z * z / n)
    return float(n * upper)


# --------------------------------------------------------------------------- #
# Tree node
# --------------------------------------------------------------------------- #
@dataclass
class Node:
    prediction: Any
    n_samples: int
    class_counts: dict
    depth: int
    feature: str | None = None
    threshold: float | None = None          # set only for numeric (binary) splits
    gain: float = 0.0
    gain_ratio: float | None = None
    children: dict = field(default_factory=dict)

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def branch_label(self, key) -> str:
        if self.threshold is None:
            return str(key)
        return f"≤ {self.threshold:.4g}" if key == "le" else f"> {self.threshold:.4g}"


# --------------------------------------------------------------------------- #
# Shared tree-building machinery
# --------------------------------------------------------------------------- #
class _BaseDecisionTree:
    name = "base"
    allow_numeric = True

    def __init__(self, max_depth: int | None = None, min_samples_split: int = 2, min_gain: float = 0.0):
        self.max_depth = max_depth
        self.min_samples_split = max(2, int(min_samples_split))
        self.min_gain = float(min_gain)

    # ---- training ---------------------------------------------------------
    def fit(self, X, y):
        X = pd.DataFrame(X).reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)
        if len(X) == 0 or len(X) != len(y):
            raise ValueError("X and y must be non-empty and of equal length.")
        if X.isna().any().any():
            raise ValueError("Missing values found. Impute or drop them before training.")

        self.classes_ = sorted(y.unique().tolist(), key=str)
        lookup = {c: i for i, c in enumerate(self.classes_)}
        self._y = y.map(lookup).to_numpy()
        self.features_ = list(X.columns)
        self.numeric_features_ = {
            c for c in X.columns
            if pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_bool_dtype(X[c])
        }
        if self.numeric_features_ and not self.allow_numeric:
            raise ValueError(
                f"{self.name} supports categorical attributes only. Discretise numeric columns first: "
                f"{sorted(self.numeric_features_)}"
            )
        self._cols = {
            c: X[c].to_numpy(dtype=float) if c in self.numeric_features_ else X[c].astype(str).to_numpy()
            for c in X.columns
        }
        self.root_ = self._build(np.arange(len(X)), list(self.features_), depth=0)
        self._after_build()
        del self._cols, self._y
        return self

    def _after_build(self) -> None:  # hook for pruning
        pass

    def _build(self, idx: np.ndarray, features: list[str], depth: int) -> Node:
        k = len(self.classes_)
        counts = np.bincount(self._y[idx], minlength=k)
        node = Node(
            prediction=self.classes_[int(counts.argmax())],
            n_samples=len(idx),
            class_counts={c: int(n) for c, n in zip(self.classes_, counts)},
            depth=depth,
        )
        pure = (counts > 0).sum() <= 1
        too_deep = self.max_depth is not None and depth >= self.max_depth
        if pure or not features or len(idx) < self.min_samples_split or too_deep:
            return node

        split = self._best_split(idx, features, counts)
        if split is None or split["gain"] <= max(self.min_gain, EPS) or len(split["groups"]) < 2:
            return node

        node.feature = split["feature"]
        node.threshold = split.get("threshold")
        node.gain = float(split["gain"])
        node.gain_ratio = split.get("gain_ratio")
        remaining = features if node.threshold is not None else [f for f in features if f != node.feature]
        for key, sub_idx in split["groups"].items():
            node.children[key] = self._build(sub_idx, remaining, depth + 1)
        return node

    def _best_split(self, idx, features, counts):  # pragma: no cover - abstract
        raise NotImplementedError

    # ---- shared candidate evaluation for categorical attributes -----------
    def _categorical_candidate(self, feature: str, idx: np.ndarray, base_h: float):
        k, n = len(self.classes_), len(idx)
        values, inv = np.unique(self._cols[feature][idx], return_inverse=True)
        if len(values) < 2:
            return None
        table = np.bincount(inv * k + self._y[idx], minlength=len(values) * k).reshape(len(values), k)
        sizes = table.sum(axis=1)
        gain = base_h - float((sizes / n * _entropy_rows(table)).sum())
        split_info = entropy(sizes)
        groups = {str(v): idx[inv == j] for j, v in enumerate(values)}
        return {
            "feature": feature, "gain": gain, "split_info": split_info,
            "gain_ratio": gain / split_info if split_info > EPS else 0.0, "groups": groups,
        }

    # ---- prediction -------------------------------------------------------
    def predict(self, X) -> np.ndarray:
        X = pd.DataFrame(X)
        records = X[self.features_].to_dict("records")
        return np.array([self._predict_one(r) for r in records])

    def _predict_one(self, rec: dict):
        node = self.root_
        while not node.is_leaf:
            v = rec[node.feature]
            if node.feature in self.numeric_features_:
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    break
                if np.isnan(v):
                    break
                key = "le" if v <= node.threshold else "gt"
            else:
                key = str(v)
                if key not in node.children:  # unseen category -> majority class here
                    break
            node = node.children[key]
        return node.prediction

    # ---- introspection ----------------------------------------------------
    def _walk(self, node: Node | None = None) -> Iterator[Node]:
        node = node or self.root_
        yield node
        for child in node.children.values():
            yield from self._walk(child)

    def get_depth(self) -> int:
        return max(n.depth for n in self._walk())

    def get_n_nodes(self) -> int:
        return sum(1 for _ in self._walk())

    def get_n_leaves(self) -> int:
        return sum(1 for n in self._walk() if n.is_leaf)

    def feature_importances(self) -> pd.Series:
        """Total sample-weighted information gain per feature, normalised to 1."""
        total = self.root_.n_samples
        imp = {f: 0.0 for f in self.features_}
        for n in self._walk():
            if not n.is_leaf:
                imp[n.feature] += n.gain * n.n_samples / total
        s = pd.Series(imp)
        return s / s.sum() if s.sum() > 0 else s

    def export_text(self) -> str:
        lines: list[str] = []

        def rec(node: Node, indent: str):
            for key, child in node.children.items():
                cond = f"{node.feature} {node.branch_label(key)}" if node.threshold is not None \
                    else f"{node.feature} = {key}"
                if child.is_leaf:
                    lines.append(f"{indent}|--- {cond}  ->  class: {child.prediction}  (n={child.n_samples})")
                else:
                    lines.append(f"{indent}|--- {cond}")
                    rec(child, indent + "|   ")

        if self.root_.is_leaf:
            lines.append(f"class: {self.root_.prediction}  (n={self.root_.n_samples})")
        else:
            rec(self.root_, "")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# ID3
# --------------------------------------------------------------------------- #
class ID3Classifier(_BaseDecisionTree):
    """ID3: choose the categorical attribute with the highest information gain."""

    name = "ID3"
    allow_numeric = False

    def _best_split(self, idx, features, counts):
        base_h = entropy(counts)
        best = None
        for f in features:
            cand = self._categorical_candidate(f, idx, base_h)
            if cand is not None and (best is None or cand["gain"] > best["gain"] + EPS):
                best = cand
        if best is None:
            return None
        best["gain_ratio"] = None  # not used by ID3
        return best


# --------------------------------------------------------------------------- #
# C4.5
# --------------------------------------------------------------------------- #
class C45Classifier(_BaseDecisionTree):
    """C4.5: gain ratio, continuous attributes, optional pessimistic pruning."""

    name = "C4.5"

    def __init__(self, max_depth=None, min_samples_split=2, min_gain=0.0, prune: bool = True):
        super().__init__(max_depth, min_samples_split, min_gain)
        self.prune = prune

    def _numeric_candidate(self, feature: str, idx: np.ndarray, base_h: float):
        k, n = len(self.classes_), len(idx)
        col, y = self._cols[feature][idx], self._y[idx]
        order = np.argsort(col, kind="stable")
        sv, sy = col[order], y[order]
        change = np.flatnonzero(sv[1:] != sv[:-1])       # positions between distinct values
        if change.size == 0:
            return None
        onehot = np.zeros((n, k))
        onehot[np.arange(n), sy] = 1
        cum = np.cumsum(onehot, axis=0)
        left = cum[change]
        right = cum[-1] - left
        n_left = change + 1
        n_right = n - n_left
        weighted = (n_left * _entropy_rows(left) + n_right * _entropy_rows(right)) / n
        gains = base_h - weighted
        j = int(np.argmax(gains))
        lo, hi = sv[change[j]], sv[change[j] + 1]
        thr = (lo + hi) / 2.0
        if thr >= hi:                                    # floating-point guard
            thr = lo
        mask = col <= thr
        split_info = entropy([n_left[j], n_right[j]])
        gain = float(gains[j])
        return {
            "feature": feature, "gain": gain, "split_info": split_info, "threshold": float(thr),
            "gain_ratio": gain / split_info if split_info > EPS else 0.0,
            "groups": {"le": idx[mask], "gt": idx[~mask]},
        }

    def _best_split(self, idx, features, counts):
        base_h = entropy(counts)
        cands = []
        for f in features:
            if f in self.numeric_features_:
                c = self._numeric_candidate(f, idx, base_h)
            else:
                c = self._categorical_candidate(f, idx, base_h)
            if c is not None and c["gain"] > EPS and c["split_info"] > EPS:
                cands.append(c)
        if not cands:
            return None
        # Quinlan's heuristic: only consider attributes with at least average gain,
        # then pick the one with the highest gain ratio.
        avg_gain = float(np.mean([c["gain"] for c in cands]))
        pool = [c for c in cands if c["gain"] >= avg_gain - EPS]
        return max(pool, key=lambda c: c["gain_ratio"])

    # ---- pessimistic error-based pruning ----------------------------------
    def _after_build(self) -> None:
        if self.prune:
            self._prune(self.root_)

    def _prune(self, node: Node) -> float:
        """Prune bottom-up; return the estimated error count of the (pruned) subtree."""
        leaf_errors = node.n_samples - max(node.class_counts.values())
        leaf_estimate = _pessimistic_errors(node.n_samples, leaf_errors)
        if node.is_leaf:
            return leaf_estimate
        subtree_estimate = sum(self._prune(ch) for ch in node.children.values())
        if leaf_estimate <= subtree_estimate + 0.1:
            node.children, node.feature, node.threshold = {}, None, None
            node.gain, node.gain_ratio = 0.0, None
            return leaf_estimate
        return subtree_estimate
