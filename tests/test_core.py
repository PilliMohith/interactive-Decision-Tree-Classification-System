import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation import evaluate_model
from src.preprocessing import PreprocessConfig, prepare_data
from src.trees import C45Classifier, ID3Classifier, entropy

DATA = Path(__file__).resolve().parents[1] / "data"


def test_entropy_values():
    assert entropy([5, 5]) == pytest.approx(1.0)
    assert entropy([10, 0]) == 0.0
    assert entropy([9, 5]) == pytest.approx(0.940, abs=1e-3)


def test_id3_play_tennis_matches_textbook():
    df = pd.read_csv(DATA / "play_tennis.csv")
    X, y = df.drop(columns="PlayTennis"), df["PlayTennis"]
    tree = ID3Classifier().fit(X, y)
    assert tree.root_.feature == "Outlook"                       # IG = 0.247 (Mitchell, 1997)
    assert tree.root_.gain == pytest.approx(0.2467, abs=1e-3)
    assert (tree.predict(X) == y).all()
    assert tree.root_.children["Overcast"].is_leaf


def test_id3_rejects_numeric_columns():
    with pytest.raises(ValueError):
        ID3Classifier().fit(pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]}), ["x", "x", "y", "y"])


def test_c45_numeric_threshold_split():
    X = pd.DataFrame({"v": [1, 2, 3, 10, 11, 12]})
    y = ["a", "a", "a", "b", "b", "b"]
    tree = C45Classifier().fit(X, y)
    assert tree.root_.feature == "v" and 3 < tree.root_.threshold < 10
    assert list(tree.predict(pd.DataFrame({"v": [0, 20]}))) == ["a", "b"]


def test_c45_pruning_never_grows_tree():
    df = pd.read_csv(DATA / "credit_approval_synthetic.csv")
    prep = prepare_data(df, "decision", PreprocessConfig())
    full = C45Classifier(prune=False).fit(prep.X_train_c45, prep.y_train)
    pruned = C45Classifier(prune=True).fit(prep.X_train_c45, prep.y_train)
    assert pruned.get_n_nodes() <= full.get_n_nodes()


def test_unseen_category_falls_back_to_majority():
    df = pd.read_csv(DATA / "play_tennis.csv")
    tree = ID3Classifier().fit(df.drop(columns="PlayTennis"), df["PlayTennis"])
    row = df.drop(columns="PlayTennis").iloc[[0]].copy()
    row["Outlook"] = "Hurricane"
    assert tree.predict(row)[0] in {"Yes", "No"}


def test_pipeline_handles_missing_and_mixed_types():
    df = pd.read_csv(DATA / "credit_approval_synthetic.csv")
    prep = prepare_data(df, "decision", PreprocessConfig(test_size=0.25))
    assert not prep.X_train_id3.isna().any().any()
    assert not any(pd.api.types.is_numeric_dtype(prep.X_train_id3[c]) for c in prep.X_train_id3)  # ID3: all categorical
    id3 = ID3Classifier(max_depth=5).fit(prep.X_train_id3, prep.y_train)
    c45 = C45Classifier(max_depth=5).fit(prep.X_train_c45, prep.y_train)
    for model, Xte in [(id3, prep.X_test_id3), (c45, prep.X_test_c45)]:
        m = evaluate_model(prep.y_test, model.predict(Xte), prep.classes)
        assert m["accuracy"] > 0.6
        assert m["confusion_matrix"].sum() == len(prep.y_test)


def test_metrics_binary():
    m = evaluate_model(["a", "a", "b", "b"], ["a", "b", "b", "b"], ["a", "b"])
    assert m["accuracy"] == 0.75
    assert m["confusion_matrix"].tolist() == [[1, 1], [0, 2]]
    assert m["recall"] == pytest.approx((0.5 + 1.0) / 2)
