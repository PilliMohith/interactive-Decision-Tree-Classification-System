"""Interactive Decision Tree Classification System (ID3 vs C4.5) - Streamlit UI."""
from __future__ import annotations

import io
import time
from pathlib import Path

import graphviz
import numpy as np
import pandas as pd
import streamlit as st

from src.evaluation import comparison_table, evaluate_model
from src.preprocessing import PreprocessConfig, prepare_data, suspicious_id_columns
from src.trees import C45Classifier, ID3Classifier
from src.visualization import (fig_confusion_matrix, fig_importance,
                               fig_metric_bars, tree_to_graphviz)

st.set_page_config(page_title="Decision Tree Classification System", page_icon="🌳", layout="wide")

DATA_DIR = Path(__file__).parent / "data"
SAMPLES = {
    "Sample: Play Tennis (categorical)": "play_tennis.csv",
    "Sample: Iris (numeric)": "iris.csv",
    "Sample: Credit approval (mixed + missing)": "credit_approval_synthetic.csv",
}


# --------------------------------------------------------------------------- #
# Sidebar - dataset + hyper-parameters
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("🌳 Decision Trees")
    source = st.radio("Dataset", ["Upload CSV", *SAMPLES])
    uploaded = st.file_uploader("CSV file", type=["csv"]) if source == "Upload CSV" else None

    st.subheader("Tree settings")
    no_limit = st.checkbox("No depth limit", value=False)
    max_depth = None if no_limit else st.slider("Max depth", 1, 20, 5)
    min_split = st.number_input("Min samples to split", 2, 1000, 2)
    min_gain = st.number_input("Min gain", 0.0, 1.0, 0.0, step=0.005, format="%.3f")
    prune = st.checkbox("C4.5 pessimistic pruning", value=True)

    st.subheader("Evaluation")
    average = st.selectbox("Averaging for precision / recall / F1", ["macro", "weighted"])


@st.cache_data(show_spinner=False)
def read_csv_bytes(raw: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(raw), sep=None, engine="python")  # auto-detect delimiter


if source == "Upload CSV":
    if uploaded is None:
        st.title("Decision Tree Classification System")
        st.info("Upload a classification dataset (CSV) in the sidebar, or pick a bundled sample.")
        st.stop()
    try:
        df = read_csv_bytes(uploaded.getvalue())
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read the CSV: {exc}")
        st.stop()
    dataset_key = f"upload:{uploaded.name}:{uploaded.size}"
else:
    df = pd.read_csv(DATA_DIR / SAMPLES[source])
    dataset_key = source

if st.session_state.get("dataset_key") != dataset_key:      # new dataset -> reset pipeline
    for k in ("prep", "results"):
        st.session_state.pop(k, None)
    st.session_state["dataset_key"] = dataset_key

st.title("Decision Tree Classification System")
st.caption("Upload → select target → preprocess → train ID3 & C4.5 → inspect tree → evaluate → compare")

tab_data, tab_prep, tab_train, tab_eval, tab_cmp = st.tabs(
    ["1 · Data", "2 · Preprocess", "3 · Train & Tree", "4 · Evaluate & Predict", "5 · Compare"])

# --------------------------------------------------------------------------- #
# 1 - Data
# --------------------------------------------------------------------------- #
with tab_data:
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing cells", int(df.isna().sum().sum()))

    target = st.selectbox("Target attribute (class label)", list(df.columns),
                          index=len(df.columns) - 1, key=f"target::{dataset_key}")
    st.dataframe(df.head(50))

    left, right = st.columns(2)
    with left:
        st.markdown("**Column summary**")
        st.dataframe(pd.DataFrame({
            "dtype": df.dtypes.astype(str), "unique": df.nunique(), "missing": df.isna().sum()}))
    with right:
        st.markdown(f"**Class distribution of `{target}`**")
        st.bar_chart(df[target].astype(str).value_counts())
    if df[target].nunique() > 20:
        st.warning("The target has more than 20 distinct values - check that it is a class label.")

# --------------------------------------------------------------------------- #
# 2 - Preprocess
# --------------------------------------------------------------------------- #
with tab_prep:
    feature_cols = [c for c in df.columns if c != target]
    suspects = suspicious_id_columns(df, target)
    if suspects:
        st.warning(f"Possible identifier column(s) pre-selected for removal: {suspects}")

    a, b, c = st.columns(3)
    with a:
        drop_cols = st.multiselect("Columns to drop", feature_cols, default=suspects,
                                   key=f"drop::{dataset_key}::{target}")
        missing = st.radio("Missing values", ["Impute", "Drop rows"], horizontal=True)
        num_imp = st.radio("Numeric imputation", ["median", "mean"], horizontal=True)
    with b:
        test_size = st.slider("Test set size", 0.1, 0.5, 0.3, 0.05)
        seed = st.number_input("Random seed", 0, 10_000, 42)
        stratify = st.checkbox("Stratified split", value=True)
    with c:
        n_bins = st.slider("Bins for numeric columns (ID3)", 2, 10, 4)
        binning = st.radio("Binning method", ["quantile", "uniform"], horizontal=True)
        low_card = st.number_input("Treat numeric columns with ≤ N unique values as categories (ID3)", 2, 50, 6)

    if st.button("Run preprocessing", type="primary"):
        cfg = PreprocessConfig(drop_cols, "drop_rows" if missing == "Drop rows" else "impute", num_imp,
                               test_size, int(seed), stratify, n_bins, binning, int(low_card))
        try:
            st.session_state["prep"] = prepare_data(df, target, cfg)
            st.session_state.pop("results", None)
        except ValueError as exc:
            st.session_state.pop("prep", None)
            st.error(str(exc))

    prep = st.session_state.get("prep")
    if prep is None:
        st.info("Configure the options above and click **Run preprocessing**.")
    else:
        st.success(f"Train: {len(prep.y_train)} rows · Test: {len(prep.y_test)} rows · "
                   f"Features: {len(prep.preparer.columns_)} · Classes: {len(prep.classes)}")
        for note in prep.notes:
            st.write("• " + note)
        v1, v2 = st.columns(2)
        v1.markdown("**ID3 view** (all attributes categorical)")
        v1.dataframe(prep.X_train_id3.head(20))
        v2.markdown("**C4.5 view** (numeric attributes kept continuous)")
        v2.dataframe(prep.X_train_c45.head(20))
        if prep.preparer.bin_bounds_:
            with st.expander("Discretisation bins used for ID3"):
                for col, labels in prep.preparer.bin_labels_.items():
                    st.write(f"**{col}**: " + " | ".join(labels))

# --------------------------------------------------------------------------- #
# 3 - Train & tree
# --------------------------------------------------------------------------- #
with tab_train:
    prep = st.session_state.get("prep")
    if prep is None:
        st.info("Run preprocessing first (tab 2).")
    else:
        if st.button("Train ID3 and C4.5", type="primary"):
            results = {}
            specs = {
                "ID3": (ID3Classifier(max_depth, min_split, min_gain), prep.X_train_id3, prep.X_test_id3),
                "C4.5": (C45Classifier(max_depth, min_split, min_gain, prune), prep.X_train_c45, prep.X_test_c45),
            }
            for name, (model, Xtr, Xte) in specs.items():
                t0 = time.perf_counter()
                model.fit(Xtr, prep.y_train)
                fit_time = time.perf_counter() - t0
                pred = model.predict(Xte)
                results[name] = {
                    "model": model, "fit_time": fit_time, "pred": pred,
                    "metrics": evaluate_model(prep.y_test, pred, prep.classes, average),
                }
            st.session_state["results"] = results

        results = st.session_state.get("results")
        if results is None:
            st.info("Click **Train ID3 and C4.5**.")
        else:
            algo = st.radio("Show tree for", list(results), horizontal=True)
            model = results[algo]["model"]
            s1, s2, s3 = st.columns(3)
            s1.metric("Depth", model.get_depth())
            s2.metric("Nodes", model.get_n_nodes())
            s3.metric("Leaves", model.get_n_leaves())

            depth_view = st.slider("Levels to display", 1, max(model.get_depth(), 1),
                                   min(model.get_depth(), 6) or 1,
                                   help="Deeper levels are collapsed into dashed nodes.")
            dot = tree_to_graphviz(model, depth_view)
            st.graphviz_chart(dot)

            d1, d2, _ = st.columns([1, 1, 3])
            d1.download_button("Download DOT", dot.source, f"{algo}_tree.dot")
            try:
                d2.download_button("Download PNG", tree_to_graphviz(model, depth_view).pipe(format="png"),
                                   f"{algo}_tree.png", mime="image/png")
            except graphviz.ExecutableNotFound:
                d2.caption("Install the Graphviz binary for PNG export.")

            with st.expander("Decision rules (text)"):
                st.code(model.export_text(), language="text")
            with st.expander("Feature importance"):
                st.pyplot(fig_importance(model.feature_importances(), f"{algo} feature importance"))

# --------------------------------------------------------------------------- #
# 4 - Evaluate & predict
# --------------------------------------------------------------------------- #
with tab_eval:
    results, prep = st.session_state.get("results"), st.session_state.get("prep")
    if not results:
        st.info("Train the models first (tab 3).")
    else:
        algo = st.radio("Algorithm", list(results), horizontal=True, key="eval_algo")
        res = results[algo]
        m = res["metrics"]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Accuracy", f"{m['accuracy']:.3f}")
        k2.metric(f"Precision ({m['average']})", f"{m['precision']:.3f}")
        k3.metric(f"Recall ({m['average']})", f"{m['recall']:.3f}")
        k4.metric(f"F1-score ({m['average']})", f"{m['f1']:.3f}")

        cm_col, pc_col = st.columns([1, 1])
        with cm_col:
            st.pyplot(fig_confusion_matrix(m["confusion_matrix"], prep.classes, f"{algo} - confusion matrix"))
        with pc_col:
            st.markdown("**Per-class metrics**")
            st.dataframe(m["per_class"].style.format(
                {"precision": "{:.3f}", "recall": "{:.3f}", "f1-score": "{:.3f}", "support": "{:.0f}"}))

        st.markdown("**Test-set predictions**")
        pred_df = prep.X_test_c45.copy()
        pred_df.insert(0, "actual", prep.y_test.to_numpy())
        pred_df.insert(1, "predicted", res["pred"])
        pred_df.insert(2, "correct", pred_df["actual"] == pred_df["predicted"])
        st.dataframe(pred_df)
        st.download_button("Download predictions (CSV)", pred_df.to_csv(index=False),
                           f"{algo}_predictions.csv", mime="text/csv")

        st.divider()
        st.markdown("**Predict a new record**")
        raw = prep.raw_X_train
        with st.form("single_prediction"):
            cols = st.columns(3)
            record = {}
            for i, col in enumerate(prep.preparer.columns_):
                with cols[i % 3]:
                    if col in prep.preparer.numeric_cols_:
                        s = raw[col].dropna()
                        record[col] = st.number_input(col, value=float(s.median()) if len(s) else 0.0,
                                                      min_value=None, max_value=None)
                    else:
                        opts = sorted(raw[col].dropna().astype(str).unique())
                        record[col] = st.selectbox(col, opts)
            go = st.form_submit_button("Predict")
        if go:
            c45_row, id3_row = prep.preparer.transform(pd.DataFrame([record]))
            p1, p2 = st.columns(2)
            p1.success(f"ID3 → **{results['ID3']['model'].predict(id3_row)[0]}**")
            p2.success(f"C4.5 → **{results['C4.5']['model'].predict(c45_row)[0]}**")

# --------------------------------------------------------------------------- #
# 5 - Compare
# --------------------------------------------------------------------------- #
with tab_cmp:
    results, prep = st.session_state.get("results"), st.session_state.get("prep")
    if not results:
        st.info("Train the models first (tab 3).")
    else:
        table = comparison_table(results)
        st.dataframe(table.style.format({
            "Accuracy": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}", "F1-score": "{:.4f}",
            "Train time (s)": "{:.4f}"}).highlight_max(
            subset=["Accuracy", "Precision", "Recall", "F1-score"], color="#c6efce"))

        best = table["F1-score"].idxmax()
        tied = table["F1-score"].nunique() == 1
        st.markdown(("**Both algorithms tie on F1-score.**" if tied else
                     f"**{best}** achieves the higher F1-score "
                     f"({table.loc[best, 'F1-score']:.3f}) on this test split.")
                    + f" Prediction agreement between ID3 and C4.5: "
                      f"**{np.mean(results['ID3']['pred'] == results['C4.5']['pred']):.1%}**.")

        g1, g2 = st.columns(2)
        g1.pyplot(fig_metric_bars(table))
        with g2:
            st.markdown("**Tree size**")
            st.bar_chart(table[["Depth", "Nodes", "Leaves"]])

        cm1, cm2 = st.columns(2)
        for col, name in zip((cm1, cm2), results):
            col.pyplot(fig_confusion_matrix(results[name]["metrics"]["confusion_matrix"],
                                            prep.classes, f"{name} - confusion matrix"))
        st.caption("ID3 uses information gain on binned attributes; C4.5 uses gain ratio, "
                   "keeps numeric attributes continuous and (optionally) prunes. "
                   "A single split can be noisy - try several seeds / test sizes before drawing conclusions.")
