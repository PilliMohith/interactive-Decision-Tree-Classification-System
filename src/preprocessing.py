"""Data cleaning, imputation, discretisation and train/test splitting."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class PreprocessConfig:
    drop_columns: list[str] = field(default_factory=list)
    missing_strategy: str = "impute"      # "impute" | "drop_rows"
    numeric_impute: str = "median"        # "median" | "mean"
    test_size: float = 0.3
    random_state: int = 42
    stratify: bool = True
    n_bins: int = 4                       # bins used to discretise numeric columns for ID3
    binning: str = "quantile"             # "quantile" | "uniform"
    low_cardinality: int = 6              # numeric columns with <= this many unique values are
                                          # treated as categories by ID3


def _fmt(v: float) -> str:
    return f"{v:g}"


class DataPreparer:
    """Fitted on the training split only (avoids leakage); builds two views of the data:

    * C4.5 view - numeric columns stay numeric, categoricals are strings.
    * ID3 view  - every column is categorical (numeric columns are binned).
    """

    def __init__(self, config: PreprocessConfig):
        self.config = config

    def fit(self, X: pd.DataFrame) -> "DataPreparer":
        cfg = self.config
        self.columns_ = list(X.columns)
        self.numeric_cols_ = [c for c in X.columns
                              if pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_bool_dtype(X[c])]
        self.categorical_cols_ = [c for c in X.columns if c not in self.numeric_cols_]

        self.num_fill_ = {}
        for c in self.numeric_cols_:
            v = X[c].median() if cfg.numeric_impute == "median" else X[c].mean()
            self.num_fill_[c] = 0.0 if pd.isna(v) else float(v)
        self.cat_fill_ = {}
        for c in self.categorical_cols_:
            mode = X[c].dropna().astype(str).mode()
            self.cat_fill_[c] = mode.iloc[0] if len(mode) else "missing"

        filled = self._impute(X)
        self.bin_bounds_, self.bin_labels_, self.as_category_ = {}, {}, []
        for c in self.numeric_cols_:
            col = filled[c]
            if col.nunique() <= cfg.low_cardinality:
                self.as_category_.append(c)
                continue
            if cfg.binning == "uniform":
                inner = np.linspace(col.min(), col.max(), cfg.n_bins + 1)[1:-1]
            else:
                inner = np.quantile(col, np.linspace(0, 1, cfg.n_bins + 1)[1:-1])
            inner = np.unique(inner)
            bounds = [-np.inf, *inner.tolist(), np.inf]
            labels = []
            for i in range(len(bounds) - 1):
                lo, hi = bounds[i], bounds[i + 1]
                if len(bounds) == 2:
                    labels.append("all values")
                elif i == 0:
                    labels.append(f"≤ {hi:.4g}")
                elif i == len(bounds) - 2:
                    labels.append(f"> {lo:.4g}")
                else:
                    labels.append(f"({lo:.4g}, {hi:.4g}]")
            self.bin_bounds_[c], self.bin_labels_[c] = bounds, labels
        return self

    def _impute(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X[self.columns_].copy()
        for c in self.numeric_cols_:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(self.num_fill_[c]).astype(float)
        for c in self.categorical_cols_:
            X[c] = X[c].astype(object).where(X[c].notna(), self.cat_fill_[c]).astype(str)
        return X

    def transform(self, X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        c45 = self._impute(X)
        id3 = c45.copy()
        for c in self.numeric_cols_:
            if c in self.as_category_:
                id3[c] = id3[c].map(_fmt)
            else:
                id3[c] = pd.cut(id3[c], bins=self.bin_bounds_[c], labels=self.bin_labels_[c]).astype(str)
        return c45, id3


@dataclass
class PreparedData:
    config: PreprocessConfig
    target: str
    preparer: DataPreparer
    raw_X_train: pd.DataFrame
    X_train_c45: pd.DataFrame
    X_test_c45: pd.DataFrame
    X_train_id3: pd.DataFrame
    X_test_id3: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    classes: list
    notes: list[str]


def suspicious_id_columns(df: pd.DataFrame, target: str) -> list[str]:
    """Columns that look like row identifiers (useless / harmful for trees)."""
    out, n = [], len(df)
    for c in df.columns:
        if c == target or n < 20:
            continue
        nun = df[c].nunique(dropna=True)
        name = str(c).lower()
        if not pd.api.types.is_numeric_dtype(df[c]) and nun > 0.5 * n:
            out.append(c)
        elif pd.api.types.is_integer_dtype(df[c]) and nun == n and (
                "id" in name.replace("_", " ").split() or "index" in name or name.startswith("unnamed")):
            out.append(c)
    return out


def prepare_data(df: pd.DataFrame, target: str, config: PreprocessConfig) -> PreparedData:
    notes: list[str] = []
    data = df.copy()

    before = len(data)
    data = data.dropna(subset=[target])
    if len(data) < before:
        notes.append(f"Removed {before - len(data)} rows with a missing target value.")

    data = data.drop(columns=[c for c in config.drop_columns if c in data.columns])
    X, y = data.drop(columns=[target]), data[target].astype(str).str.strip()
    if X.shape[1] == 0:
        raise ValueError("No feature columns left after dropping columns.")

    for c in X.select_dtypes(include=["object", "string"]).columns:
        X[c] = X[c].astype("object").map(lambda v: v.strip() if isinstance(v, str) else v).replace("", np.nan)

    if config.missing_strategy == "drop_rows":
        mask = X.notna().all(axis=1)
        if (~mask).sum():
            notes.append(f"Dropped {int((~mask).sum())} rows containing missing values.")
        X, y = X[mask], y[mask]
    elif X.isna().any().any():
        notes.append(f"Imputed {int(X.isna().sum().sum())} missing cells using training-set statistics "
                     f"({config.numeric_impute} for numeric, mode for categorical).")

    if len(X) < 10:
        raise ValueError("Too few rows remain after preprocessing (need at least 10).")
    if y.nunique() < 2:
        raise ValueError("The target needs at least two distinct classes.")
    if y.nunique() > 20:
        notes.append(f"Target has {y.nunique()} classes - is it really a classification target?")

    strat = y if config.stratify and y.value_counts().min() >= 2 else None
    if config.stratify and strat is None:
        notes.append("Stratified split skipped: a class has fewer than 2 samples.")
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=config.test_size, random_state=config.random_state, stratify=strat)
    except ValueError:
        notes.append("Stratified split not possible with this test size; used a random split instead.")
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=config.test_size, random_state=config.random_state)

    preparer = DataPreparer(config).fit(X_tr)
    tr_c45, tr_id3 = preparer.transform(X_tr)
    te_c45, te_id3 = preparer.transform(X_te)
    if preparer.bin_bounds_:
        notes.append(f"Binned {len(preparer.bin_bounds_)} numeric column(s) for ID3 "
                     f"({config.n_bins} {config.binning} bins).")
    return PreparedData(config, target, preparer, X_tr, tr_c45, te_c45, tr_id3, te_id3,
                        y_tr, y_te, sorted(y.unique().tolist()), notes)
