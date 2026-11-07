# Decision Tree Classification System - Documentation

## 1. Objective
Provide an interactive tool to upload a classification dataset, preprocess it, build **ID3** and **C4.5**
decision trees, visualise them, generate predictions, evaluate them (confusion matrix, accuracy, precision,
recall, F1-score) and compare the two algorithms.

## 2. Technology stack
| Library | Role |
|---|---|
| Python 3.10+ | language |
| Pandas, NumPy | data handling; vectorised entropy / split search |
| Scikit-learn | `train_test_split` and evaluation metrics |
| Matplotlib | confusion matrices, metric and importance charts |
| Graphviz | tree drawing (`graphviz.Digraph`, DOT/PNG export) |
| Streamlit | web UI |

The trees themselves are written from scratch (`src/trees.py`) so that the algorithm logic is visible; scikit-learn is
used only for splitting and metrics (its `DecisionTreeClassifier` implements CART, not ID3/C4.5).

## 3. Architecture
```
CSV ─► app.py ─► preprocessing.prepare_data ─► PreparedData ─┬─► ID3 view (all categorical) ─► ID3Classifier
                                                              └─► C4.5 view (numeric kept)    ─► C45Classifier
                                       predictions ─► evaluation.evaluate_model ─► charts / comparison
```
State is kept in `st.session_state` (`prep`, `results`) and reset automatically when the dataset changes.

## 4. Preprocessing (`src/preprocessing.py`)
1. Rows with a missing target are removed; chosen columns are dropped (identifier-like columns are suggested automatically).
2. Missing features: **impute** (median/mean for numeric, mode for categorical) or **drop rows**. Statistics are computed on the **training split only** to avoid leakage.
3. Stratified train/test split (falls back to random if a class is too small).
4. Two views are produced from the same split:
   * **C4.5 view** - numeric attributes stay continuous.
   * **ID3 view** - numeric attributes are discretised (quantile or uniform bins fitted on the training data); numeric columns with few unique values become categories.
5. New records (single-record prediction) pass through the same fitted `DataPreparer`.

## 5. Algorithms (`src/trees.py`)
Let S be the samples at a node, p_i the class proportions.

* Entropy: `H(S) = -Σ p_i log2 p_i`
* Information gain: `Gain(S,A) = H(S) - Σ_v |S_v|/|S| · H(S_v)`
* Split information: `SplitInfo(S,A) = -Σ_v |S_v|/|S| · log2(|S_v|/|S|)`
* Gain ratio: `GainRatio = Gain / SplitInfo`

### ID3
* Categorical attributes only (raises an error for numeric columns).
* Multiway split on the attribute with the maximum information gain; the attribute is removed from the candidate list below that node.
* Stops when the node is pure, no attributes remain, `min_samples_split` is not met, `max_depth` is reached, or gain ≤ `min_gain`.
* Unseen category at prediction time → majority class of the current node.

### C4.5
* **Numeric attributes**: sort values, evaluate midpoints between distinct neighbours using cumulative class counts (vectorised), choose the threshold with the best gain → binary split `≤ t` / `> t`. Numeric attributes may be reused deeper in the tree.
* **Categorical attributes**: multiway split like ID3.
* **Split selection**: among attributes whose gain ≥ the average gain, choose the highest **gain ratio** (Quinlan's heuristic, avoids the bias towards many-valued attributes).
* **Pruning** (optional): bottom-up pessimistic error-based pruning with confidence factor 25 %; a subtree is replaced by a leaf if the leaf's estimated error ≤ the subtree's.
* Missing values must be imputed beforehand (fractional-instance handling from the original C4.5 is not implemented).

## 6. Evaluation (`src/evaluation.py`)
* Confusion matrix (rows = actual, columns = predicted)
* Accuracy = correct / total
* Precision = TP / (TP + FP), Recall = TP / (TP + FN), F1 = 2PR / (P + R)
* Multi-class: **macro** (unweighted mean over classes) or **weighted** (by support), selectable in the sidebar; per-class values are always shown.

## 7. Using the app
1. **Data** - upload a CSV (or choose a sample) and select the target column.
2. **Preprocess** - choose columns to drop, missing-value handling, split and binning options → *Run preprocessing*.
3. **Train & Tree** - set depth/min-split/min-gain/pruning in the sidebar → *Train ID3 and C4.5*; inspect the tree, rules and feature importance; export DOT/PNG.
4. **Evaluate & Predict** - metrics, confusion matrix, per-class table, test predictions (CSV), single-record prediction.
5. **Compare** - metrics table, bar charts, tree sizes, training time, prediction agreement.

Bundled samples: *Play Tennis* (categorical, reproduces the textbook ID3 tree - root = Outlook, gain 0.247), *Iris* (numeric), *Credit approval* (synthetic, mixed types + missing values).

## 8. Testing
`python -m pytest -q` covers: entropy values, the textbook ID3 tree, numeric thresholds, pruning, unseen categories, the full preprocessing → training → evaluation pipeline and metric correctness.

## 9. Git / GitHub workflow
```bash
git init -b main
git add .gitignore requirements.txt packages.txt
git commit -m "chore: project skeleton and dependencies"
git add src/trees.py tests/
git commit -m "feat: ID3 and C4.5 implementations with unit tests"
git add src/preprocessing.py src/evaluation.py src/visualization.py data/
git commit -m "feat: preprocessing, evaluation, visualisation and sample data"
git add app.py
git commit -m "feat: Streamlit interface"
git add README.md docs/
git commit -m "docs: README and project documentation"

git remote add origin https://github.com/<your-username>/decision-tree-classifier.git
git push -u origin main
```
Recommended practice: feature branches (`git switch -c feature/<name>`) merged through pull requests, Conventional-Commit
messages (`feat:`, `fix:`, `docs:`, `test:`), small focused commits, and a release tag (`git tag v1.0 && git push --tags`).
To deploy for free: push to GitHub and create an app on Streamlit Community Cloud pointing at `app.py`
(`packages.txt` installs the Graphviz binary there).

## 10. Limitations and future work
* No handling of missing values inside the trees (imputation is done up front).
* Pruning is pessimistic-error based; reduced-error or cost-complexity pruning could be added.
* Evaluation uses one hold-out split; k-fold cross-validation would give more stable comparisons.
* Very large datasets (>100k rows) will be slow because the search is pure Python/NumPy.
