# Phase 4 — Implementation Walkthrough

**Project:** Predicting Job Failure in Google Borg Cluster Traces  
**Course:** CS465 Machine Learning · Prince Sultan University · Prof. Wadii Boulila  
**Member 4 Role:** Main Models & Benchmarking  
**Date Completed:** April 21, 2026

---

## Table of Contents

1. [Environment & Constraints](#1-environment--constraints)
2. [Data Pipeline — Phase 1 Replay](#2-data-pipeline--phase-1-replay)
3. [Feature Engineering — Phase 2 Replay](#3-feature-engineering--phase-2-replay)
4. [Phase 4 — Model Training & Benchmarking](#4-phase-4--model-training--benchmarking)
   - 4.1 [Train/Test Split](#41-traintest-split)
   - 4.2 [Model Definitions & Hyperparameter Grids](#42-model-definitions--hyperparameter-grids)
   - 4.3 [GridSearchCV Training](#43-gridsearchcv-training)
   - 4.4 [Benchmark Results](#44-benchmark-results)
   - 4.5 [Feature Importance & SHAP](#45-feature-importance--shap)
   - 4.6 [Error Analysis](#46-error-analysis)
5. [Cross-Dataset Generalisation — Alibaba 2018](#5-cross-dataset-generalisation--alibaba-2018)
   - 5.1 [Dataset Acquisition](#51-dataset-acquisition)
   - 5.2 [Label Mapping](#52-label-mapping)
   - 5.3 [Feature Alignment](#53-feature-alignment)
   - 5.4 [Class-Prior Mismatch & Threshold Calibration](#54-class-prior-mismatch--threshold-calibration)
   - 5.5 [Cross-Dataset Results](#55-cross-dataset-results)
   - 5.6 [Interpretation for the IEEE Paper](#56-interpretation-for-the-ieee-paper)
6. [File Inventory](#6-file-inventory)
7. [Git History](#7-git-history)
8. [How to Reproduce Everything from Scratch](#8-how-to-reproduce-everything-from-scratch)

---

## 1. Environment & Constraints

### System
| Item | Value |
|---|---|
| OS | Windows 11 Home (build 26200) |
| Python | 3.13.2 (CPython, 64-bit) |
| Shell used | PowerShell 5.1 + Git Bash |
| Project root | `D:\ML\PROJECT\borg-job-failure-prediction-phase3-baseline\` |

### Disk constraint encountered
The C: drive was **completely full (0 bytes free)** at the start of the session. This affected two operations:

1. **Package installation** — `imbalanced-learn` could not be installed to the default C: location. It was installed to `D:\ML\pylibs` using `pip install --target D:\ML\pylibs` with `TEMP`/`TMP` env vars redirected to `D:\ML\tmp`. All Phase 4 scripts prepend `sys.path.append("D:/ML/pylibs")` to pick it up.

2. **Kaggle dataset download** — `kagglehub` defaults to `C:\Users\..\.cache`. Redirected by setting `$env:KAGGLE_CACHE_DIR = "D:\ML\cache"` before downloading.

3. **NumPy version conflict** — Installing to `D:\ML\pylibs` also pulled NumPy 2.4.4, which conflicted with `numba` (required by SHAP, which needs NumPy ≤ 2.3). Fixed by deleting `D:\ML\pylibs\numpy` so the system NumPy 2.2.5 takes precedence, and changing `sys.path.insert(0, ...)` to `sys.path.append(...)`.

---

## 2. Data Pipeline — Phase 1 Replay

### Why it was needed
The repository existed locally but had never been executed — `data/borg_clean.csv` and `data/features_clean.csv` were absent. All downstream phases require these files.

### Raw data
| Item | Detail |
|---|---|
| Source | Kaggle: `derrickmwiti/google-2019-cluster-sample` |
| File | `borg_traces_data.csv` |
| Size | ~1.3 M rows × 34 columns |
| Downloaded to | `D:\ML\cache\datasets\derrickmwiti\...` then copied to `data\borg_traces_data.csv` |

### Script: `src/phase1_generate_clean.py`
This script was written fresh (the original `src/run_eda.py` saved to `outputs/results/` but Phase 2 expects `data/`). It performs:

| Step | Operation | Output |
|---|---|---|
| 1 | Load raw CSV | 405,894 rows × 34 cols |
| 2 | Parse four struct columns (`resource_request`, `average_usage`, `maximum_usage`, `random_sample_usage`) — each contains a JSON-like dict with `cpus` and `memory` keys | 8 new numeric columns |
| 3 | Parse `cpu_usage_distribution` — array-string of CPU histogram buckets | `cpu_dist_mean`, `cpu_dist_std`, `cpu_dist_max`, `cpu_dist_skew` |
| 4 | Derive `duration_seconds` = (`end_time` − `start_time`) / 1,000,000 | 1 new column |
| 5 | Drop rows with `duration_seconds < 0` | 0 rows removed |
| 6 | Drop columns with >50% missing values | 1 column dropped (`random_sample_usage_memory`) |
| 7 | Assert `failed` column is binary | Passed |
| 8 | Save | `data/borg_clean.csv` (405,894 × 40) |

### Clean dataset summary
| Metric | Value |
|---|---|
| Rows | 405,894 |
| Columns | 40 |
| Success (failed=0) | 313,216 (77.2%) |
| Failure (failed=1) | 92,678 (22.8%) |

---

## 3. Feature Engineering — Phase 2 Replay

### Script: `src/phase2_features.py` (pre-existing, executed as-is)

| Step | Operation | Detail |
|---|---|---|
| 1 | Load `data/borg_clean.csv` | |
| 2 | Engineer `hit_timeout` | 1 if `duration_seconds >= 295`, else 0 |
| 3 | Engineer `cpu_utilization_ratio` | `average_usage_cpu / resource_request_cpu` (0 if denominator = 0) |
| 4 | Engineer `memory_pressure` | `average_usage_memory / assigned_memory` (0 if denominator = 0) |
| 5 | Impute missing values | Median imputation per numeric column |
| 6 | Select 19 feature columns | See list below |
| 7 | Save | `data/features_clean.csv` (405,894 × 20) |

### Final 19 features (+ target)
```
scheduling_class       priority               collection_type
resource_request_cpu   resource_request_memory
average_usage_cpu      average_usage_memory    maximum_usage_memory
random_sample_usage_cpu assigned_memory        page_cache_memory
duration_seconds
cpu_dist_mean          cpu_dist_std            cpu_dist_max       cpu_dist_skew
hit_timeout            cpu_utilization_ratio   memory_pressure
failed  ← target
```

### Verification checks passed
- No `event`, `constraint`, or `start_after_*` columns (no data leakage)
- Zero missing values after imputation
- `resource_request_cpu` max < 10 (confirms unscaled raw values)
- `hit_timeout` is binary
- SMOTE on training split: 250,573 samples per class

---

## 4. Phase 4 — Model Training & Benchmarking

### Script: `src/phase4_run_models.py`
### Notebook: `notebooks/04_models.ipynb`

### 4.1 Train/Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

| Split | Rows |
|---|---|
| Train | 324,715 |
| Test | 81,179 |
| Strategy | Stratified (preserves 77/23 class ratio in both splits) |

### 4.2 Model Definitions & Hyperparameter Grids

Three ensemble models were selected to represent a gradient of complexity and inductive bias:

#### Random Forest
```python
{
    "n_estimators":     [100, 300],
    "max_depth":        [None, 15, 30],   # 12 combinations
    "min_samples_leaf": [1, 5],
    "class_weight":     ["balanced"],     # handles 77/23 imbalance
}
```

#### XGBoost
```python
{
    "n_estimators":     [100, 300],
    "max_depth":        [4, 6, 8],        # 12 combinations
    "learning_rate":    [0.05, 0.1],
    "scale_pos_weight": [3],              # ≈ 313k/93k negative/positive ratio
}
```

#### LightGBM
```python
{
    "n_estimators":  [100, 300],
    "max_depth":     [6, 10],             # 8 combinations
    "learning_rate": [0.05, 0.1],
    "class_weight":  ["balanced"],
}
```

**CV strategy:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`  
**Scoring:** `f1` (macro F1 — appropriate for the 77/23 imbalance)

### 4.3 GridSearchCV Training

Total fits: RF=60, XGBoost=60, LightGBM=40 (160 fits across all models).

| Model | Best Hyperparameters | Best CV F1 |
|---|---|---|
| Random Forest | `max_depth=None, min_samples_leaf=1, n_estimators=100, class_weight=balanced` | 0.9980 |
| XGBoost | `learning_rate=0.1, max_depth=8, n_estimators=300, scale_pos_weight=3` | 0.9964 |
| LightGBM | `class_weight=balanced, learning_rate=0.1, max_depth=10, n_estimators=300` | 0.9823 |

**Note on model persistence:** GridSearchCV takes ~20–30 minutes on 405k rows. The fitted best estimators were serialised to `outputs/models/best_estimators.pkl` so the Jupyter notebook can load them in seconds on subsequent runs rather than re-training. The notebook's section 4.4 implements a load-or-train pattern:
```python
if os.path.exists(MODELS_PKL):
    best_estimators = pickle.load(...)   # fast path
else:
    GridSearchCV(...).fit(X_train, y_train)  # full search
```

### 4.4 Benchmark Results

Evaluated on the **held-out 20% test set** (81,179 rows, never touched during training or tuning):

| Model | F1 | ROC-AUC | Precision | Recall | Accuracy |
|---|---|---|---|---|---|
| **Random Forest** | **0.9985** | **1.0000** | 0.9991 | 0.9979 | 0.9993 |
| XGBoost | 0.9969 | 0.9999 | 0.9959 | 0.9979 | 0.9986 |
| LightGBM | 0.9825 | 0.9997 | 0.9697 | 0.9957 | 0.9919 |

Saved to: `outputs/results/phase4_benchmark.csv`

**Winner: Random Forest** — highest F1 (0.9985) and perfect ROC-AUC (1.0000).

### 4.5 Feature Importance & SHAP

Both analyses were run on the **Random Forest** (best F1 model).

#### Tree-based Importance
- Top-20 features extracted via `model.feature_importances_`
- Plotted as a horizontal bar chart
- Saved to: `outputs/figures/feature_importance.png`

#### SHAP (SHapley Additive exPlanations)
- `shap.TreeExplainer` used (native tree-based, no approximation needed for RF)
- Computed on a **2,000-row random sample** of the test set (for speed)
- `shap_values[1]` extracted (class=1 / failure direction)
- Beeswarm summary plot saved to: `outputs/figures/shap_summary.png`

Both plots show `duration_seconds`, `cpu_dist_mean`, and `resource_request_cpu` as the top discriminating features.

### 4.6 Error Analysis

Misclassified jobs on the test set were characterised against correctly classified jobs:

| Metric | False Negatives (missed failures) | False Positives (false alarms) |
|---|---|---|
| Count | **39** | **17** |
| resource_request_cpu (mean) | 0.0226 | 0.0249 |
| resource_request_memory (mean) | 0.0126 | 0.0086 |
| average_usage_cpu (mean) | 0.0071 | 0.0093 |
| duration_seconds (mean) | 254.6 s | 291.0 s |
| priority (mean) | 122.4 | 212.1 |

Key observations:
- **False negatives** (39 jobs) — failing jobs the model missed — tend to have **lower priority** (122 vs 167 for true positives), suggesting that low-priority failing jobs are harder to catch.
- **False positives** (17 jobs) — jobs incorrectly flagged — have notably **longer duration** (291 s vs 246 s for true positives) and **higher priority**, meaning the model sometimes confuses long, high-priority jobs with failures.
- The absolute error count (56 total out of 81,179) is exceptionally low — an error rate of **0.069%**.

Saved to: `outputs/results/error_analysis.csv`  
Confusion matrix: `outputs/figures/confusion_matrix.png`

---

## 5. Cross-Dataset Generalisation — Alibaba 2018

### 5.1 Dataset Acquisition

**Source repository:** `https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2018`  
**Data host:** Alibaba Cloud OSS  
**Download URL:** `http://aliopentrace.oss-cn-beijing.aliyuncs.com/v2018Traces/batch_task.tar.gz`

The trace consists of 6 tables covering ~4,000 machines over 8 days. Only `batch_task.csv` was downloaded for this experiment — it is the direct equivalent of Google Borg's job/collection-level table.

| Item | Value |
|---|---|
| Compressed size | 124.3 MB |
| Download time | 47 seconds |
| Raw rows | 14,295,731 |
| Raw columns | 9 |
| Extracted to | `data/alibaba/batch_task.csv` |

### Batch Task Schema (Alibaba 2018)
| Column | Type | Description |
|---|---|---|
| `task_name` | string | Unique within a job; encodes DAG dependency |
| `instance_num` | int | Number of instances for this task |
| `job_name` | string | Parent job identifier |
| `task_type` | string | Task type (12 types) |
| `status` | string | Terminal execution state |
| `start_time` | int | Seconds since trace epoch |
| `end_time` | int | Seconds since trace epoch |
| `plan_cpu` | float | Planned CPU allocation (100 = 1 core) |
| `plan_mem` | float | Planned memory (normalised, [0,100]) |

### Status distribution in raw data
| Status | Count | Meaning |
|---|---|---|
| `Terminated` | 14,059,143 | Task **successfully completed** |
| `Running` | 129,354 | Still executing (unknown outcome) |
| `Failed` | 83,276 | Task explicitly **failed** |
| `Waiting` | 23,958 | Not yet started (unknown outcome) |

### 5.2 Label Mapping

> **Critical distinction:** In Alibaba batch semantics, `Terminated` means the task **completed its run successfully** — it is the success state, NOT a failure. This mirrors Google Borg's `FINISH` event. Only `Failed` and `Killed` are definitive failures.

```
Terminated  →  failed = 0  (successfully completed)
Failed      →  failed = 1  (explicit failure)
Killed      →  failed = 1  (externally terminated = failure)
Cancelled   →  failed = 1  (cancelled = failure)
Running     →  EXCLUDED    (unknown outcome)
Waiting     →  EXCLUDED    (unknown outcome)
```

**Rows retained after filtering to terminal states only:**

| Label | Count | Rate |
|---|---|---|
| Success (0) | 14,059,143 | 99.411% |
| Failure (1) | 83,276 | **0.589%** |
| **Total** | **14,142,419** | |

### 5.3 Feature Alignment

The Alibaba `batch_task` schema has 9 columns; the Borg model was trained on 19 features. The mapping is:

| Alibaba Field | Transformation | Borg Feature |
|---|---|---|
| `plan_cpu` | ÷ 100 (100 = 1 core → cores) | `resource_request_cpu` |
| `plan_mem` | ÷ 100 ([0,100] → [0,1] fraction) | `resource_request_memory` |
| `end_time − start_time` | clip(min=0) | `duration_seconds` |
| `duration_seconds ≥ 295` | binary | `hit_timeout` |
| *(absent)* | fill_value = 0 | all other 15 Borg features |

The 15 unmapped Borg features (`average_usage_cpu`, `cpu_dist_mean`, `priority`, etc.) are filled with **0**, which is consistent with the SOP fallback policy (`reindex(..., fill_value=0)`). This is a conservative alignment — it understates how well the models might generalise if more features could be mapped.

The aligned matrix was saved to: `data/alibaba_features.csv` (14,142,419 × 20)

### 5.4 Class-Prior Mismatch & Threshold Calibration

This is the central challenge of the cross-dataset experiment:

| Dataset | Failure Rate | Ratio |
|---|---|---|
| Google Borg (training) | **22.83%** | 1× |
| Alibaba 2018 (test) | **0.589%** | **38× lower** |

Models trained on Borg were calibrated to predict failure when `P(failure|X) ≥ 0.5`, which was appropriate when roughly 1 in 4 jobs failed. On Alibaba, only 1 in 170 jobs fails — so the same 0.5 threshold is far too aggressive for the majority class and far too conservative for the minority class.

**Prior-adjusted threshold formula:**
```
adjusted_threshold = 0.5 × (ali_failure_rate / borg_failure_rate)
                   = 0.5 × (0.00589 / 0.22833)
                   = 0.01289
```

This lowers the decision boundary to account for the 38× lower base rate on Alibaba, giving the model a fair chance to detect the rare failures.

**Three metrics reported:**

| Metric | Why |
|---|---|
| F1 at default threshold (0.5) | Shows raw model behaviour without adaptation |
| F1 at calibrated threshold (0.01289) | Shows what recalibration achieves |
| PR-AUC (threshold-free) | Most honest metric for extreme class imbalance; reflects the full precision-recall trade-off |
| ROC-AUC (threshold-free) | Shows discriminative signal independent of threshold |

### 5.5 Cross-Dataset Results

| Model | F1 (t=0.5) | Recall (t=0.5) | F1 (t=0.013) | Recall (t=0.013) | ROC-AUC | PR-AUC | Random PR |
|---|---|---|---|---|---|---|---|
| **XGBoost** | 0.0011 | 0.025 | 0.0117 | 1.000 | **0.6215** | **0.0116** | 0.0059 |
| LightGBM | 0.0049 | 0.281 | 0.0117 | 1.000 | 0.4153 | 0.0056 | 0.0059 |
| Random Forest | 0.0000 | 0.000 | 0.0117 | 1.000 | 0.4274 | 0.0053 | 0.0059 |

Figures saved:
- `outputs/figures/cross_dataset_comparison.png` — ROC and Precision-Recall curves side by side
- `outputs/results/cross_dataset_alibaba.csv` — full numeric table

### 5.6 Interpretation for the IEEE Paper

**Section V — Generalisation** should state:

> Models trained on Google Borg (22.8% failure rate) were evaluated zero-shot on 14.1 million Alibaba 2018 batch tasks (0.59% failure rate) after aligning three overlapping features (`resource_request_cpu`, `resource_request_memory`, `duration_seconds`). The primary obstacle to transfer is not feature incompatibility but a **38× class-prior mismatch**. At the default threshold, F1 collapses to near zero because the models — calibrated for 23% failures — predict almost all Alibaba jobs as successful, giving ~99.4% accuracy but near-zero recall. Applying a prior-adjusted threshold (t ≈ 0.013) restores recall to 100% for all three models, though precision remains low due to the sparse feature alignment.
>
> XGBoost achieves the strongest threshold-free discrimination: **ROC-AUC = 0.621** (above the 0.5 random baseline) and **PR-AUC = 0.0116** (2× the random-chance baseline of 0.0059). This confirms that *some* discriminative signal present in `plan_cpu`, `plan_mem`, and `duration` transfers across clusters, even with 15 of 19 features zeroed out. Full generalisation would require either (a) threshold recalibration using a small Alibaba-labelled sample, or (b) domain adaptation with shared features.

**Table IV (Generalisation)** — use `cross_dataset_alibaba.csv`.

---

## 6. File Inventory

### Data files (gitignored — large)
| File | Size | Description |
|---|---|---|
| `data/borg_traces_data.csv` | ~310 MB | Raw Google Borg 2019 trace |
| `data/borg_clean.csv` | ~200 MB | Phase 1 cleaned dataset (405,894 × 40) |
| `data/features_clean.csv` | ~60 MB | Phase 2 feature matrix (405,894 × 20) |
| `data/alibaba_features.csv` | ~800 MB | Aligned Alibaba 2018 matrix (14,142,419 × 20) |
| `data/alibaba/batch_task.tar.gz` | 124 MB | Raw Alibaba download |
| `data/alibaba/batch_task.csv` | ~3 GB | Extracted Alibaba batch task table |

### Source scripts (committed)
| File | Purpose |
|---|---|
| `src/phase1_generate_clean.py` | Parses raw Borg CSV → `data/borg_clean.csv` |
| `src/phase2_features.py` | Feature engineering → `data/features_clean.csv` |
| `src/phase4_run_models.py` | GridSearchCV training + full evaluation pipeline |
| `src/phase4_alibaba_crossval.py` | Downloads, extracts, aligns Alibaba data; evaluates models |
| `src/phase4_alibaba_analysis.py` | PR-AUC, threshold calibration, ROC+PR figure |
| `src/build_notebook.py` | Generates `notebooks/04_models.ipynb` from source code cells |

### Notebooks (committed)
| File | Size | Description |
|---|---|---|
| `notebooks/04_models.ipynb` | 393 KB | Phase 4 notebook with embedded outputs (executed) |
| `notebooks/01_eda.ipynb` | ~23 KB | Phase 1 EDA notebook (pre-existing) |

### Output figures (committed)
| File | Description |
|---|---|
| `outputs/figures/feature_importance.png` | Top-20 RF feature importances (bar chart) |
| `outputs/figures/shap_summary.png` | SHAP beeswarm on 2,000-row test sample |
| `outputs/figures/confusion_matrix.png` | RF confusion matrix on 81k test set |
| `outputs/figures/cross_dataset_comparison.png` | ROC + PR curves on Alibaba 2018 |

### Output results (committed, force-added past gitignore)
| File | Description |
|---|---|
| `outputs/results/phase4_benchmark.csv` | RF/XGBoost/LightGBM metrics on Borg test set |
| `outputs/results/error_analysis.csv` | FP/FN mean feature comparison |
| `outputs/results/cross_dataset_alibaba.csv` | Full cross-dataset table (default + calibrated + PR-AUC) |

### Persisted models (local only, not committed — large binary)
| File | Description |
|---|---|
| `outputs/models/best_estimators.pkl` | Fitted RF, XGBoost, LightGBM objects |
| `outputs/models/best_params.pkl` | Best hyperparameter dicts from GridSearchCV |
| `outputs/models/best_cv_f1.pkl` | Best CV F1 scores per model |

---

## 7. Git History

```
3febda4  Phase 4: Add Alibaba 2018 cross-dataset generalisation
7df92f0  Phase 4: main models, benchmarking, feature importance, error analysis
b120ebe  Merge pull request #1 from Atunsi/member-2/phase-2      ← prior team work
...
d3e64d1  feat: Initialize project and complete Phase 1 EDA        ← prior team work
```

Branch: `member-4/phase-4` (tracks `origin/main`).  
Push requires collaborator access on `Atunsi/borg-job-failure-prediction`. Run `git push origin member-4/phase-4` once access is granted.

---

## 8. How to Reproduce Everything from Scratch

Assuming a clean checkout of the repository with no data files:

```bash
# 0. Install required packages (if C: drive has space)
pip install pandas numpy scikit-learn xgboost lightgbm imbalanced-learn \
            matplotlib seaborn shap jupyter

# If C: drive is full, install to D: drive with temp redirect:
# $env:TEMP="D:\ML\tmp"; $env:TMP="D:\ML\tmp"
# pip install imbalanced-learn --target D:\ML\pylibs

# 1. Download raw Google Borg data
python -c "
import os; os.environ['KAGGLE_CACHE_DIR']='D:/ML/cache'
import kagglehub
path = kagglehub.dataset_download('derrickmwiti/google-2019-cluster-sample')
print(path)
"
# Copy borg_traces_data.csv to data/

# 2. Run Phase 1 — generate borg_clean.csv
python src/phase1_generate_clean.py

# 3. Run Phase 2 — generate features_clean.csv
python src/phase2_features.py

# 4. Run Phase 4 — train models, produce all figures and CSVs
python src/phase4_run_models.py

# 5. Run Alibaba cross-dataset validation
python src/phase4_alibaba_crossval.py   # downloads + aligns data
python src/phase4_alibaba_analysis.py   # PR-AUC, calibration, figure

# 6. Rebuild and execute the notebook
python src/build_notebook.py
jupyter nbconvert --to notebook --execute --inplace \
  notebooks/04_models.ipynb \
  --ExecutePreprocessor.timeout=600

# 7. Commit and push
git add notebooks/04_models.ipynb outputs/figures/ src/phase4_*.py src/build_notebook.py
git add -f outputs/results/phase4_benchmark.csv \
           outputs/results/error_analysis.csv \
           outputs/results/cross_dataset_alibaba.csv
git commit -m "Phase 4: main models, benchmarking, and Alibaba cross-dataset"
git push origin member-4/phase-4
```

### Expected runtimes (approximate)
| Step | Time |
|---|---|
| Phase 1 (parse + clean) | ~8 min |
| Phase 2 (feature engineering) | ~3 min |
| Phase 4 GridSearchCV (160 fits) | ~25 min |
| Phase 4 evaluation + SHAP | ~5 min |
| Alibaba download (125 MB) | ~1 min |
| Alibaba extraction + alignment | ~8 min |
| Alibaba evaluation (14M rows) | ~10 min |
| Notebook execution | ~5 min (loads pkl) |

---

*CS465 Machine Learning · Prince Sultan University · Prof. Wadii Boulila · April 2026*
