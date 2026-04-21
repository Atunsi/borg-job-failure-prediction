"""
Phase 3 — Baseline Models
CS465 ML Project · Google Borg Job Failure Prediction

Trains and evaluates four baselines with stratified 5-fold cross-validation:
  1. Persistence          (DummyClassifier, majority class)
  2. Logistic Regression  (class_weight='balanced')
  3. Gaussian Naive Bayes (sample_weight for balance — GaussianNB has no
                           class_weight parameter in sklearn)
  4. Decision Tree        (class_weight='balanced')

Metrics reported per fold (mean ± std across 5 folds):
  F1, ROC-AUC, Precision, Recall, Accuracy

Outputs:
  outputs/results/baseline_benchmark.csv
  outputs/results/baseline_benchmark.md
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score,
    recall_score, accuracy_score,
)

# ── Config ─────────────────────────────────────────────────────────────────
# Change this to wherever Phase 2 saved the final feature matrix.

FEATURES_PATH = "data/features_clean.csv"
RESULTS_DIR   = "output/"
FIG_DIR       = "output/figures/"
RANDOM_STATE  = 42
N_SPLITS      = 5

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ── Load ───────────────────────────────────────────────────────────────────
print("=" * 60)
print("Phase 3 — Baseline Models")
print("=" * 60)
print(f"Loading feature matrix from: {FEATURES_PATH}")

df = pd.read_csv(FEATURES_PATH)
print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

# Drop accidental index column that `to_csv(..., index=False)` sometimes misses
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])
    print("Dropped column 'Unnamed: 0' (CSV index artifact).")
    

# Target
assert "failed" in df.columns, "ERROR: target column 'failed' not found."
y = df["failed"].astype(int).values
X_df = df.drop(columns=["failed"])
feature_names = X_df.columns.tolist()
X = X_df.values.astype(np.float64)

print(f"\nTarget 'failed':")
print(f"  Success (0): {(y == 0).sum():,}  ({(y == 0).mean():.1%})")
print(f"  Failure (1): {(y == 1).sum():,}  ({(y == 1).mean():.1%})")
print(f"Features:    {X.shape[1]}")

# Safety net: any residual NaNs will break LogReg / NB
n_nan = int(np.isnan(X).sum())
if n_nan > 0:
    print(f"Warning: {n_nan:,} NaN cells remaining — imputing with 0.")
    X = np.nan_to_num(X, nan=0.0)

# ── Model factory ──────────────────────────────────────────────────────────
# Re-created each fold so state (coefficients, tree) does not carry over.
def build_model(name):
    if name == "Persistence (majority)":
        return DummyClassifier(strategy="most_frequent",
                               random_state=RANDOM_STATE)
    if name == "Logistic Regression":
        return LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="liblinear",
            random_state=RANDOM_STATE,
        )
    if name == "Gaussian Naive Bayes":
        return GaussianNB()
    if name == "Decision Tree":
        return DecisionTreeClassifier(
            class_weight="balanced",
            max_depth=10,         # shallow baseline; Phase 4 tunes this
            random_state=RANDOM_STATE,
        )
    raise ValueError(name)


MODEL_NAMES = [
    "Persistence (majority)",
    "Logistic Regression",
    "Gaussian Naive Bayes",
    "Decision Tree",
]

# ── CV loop ──
# StratifiedKFold ensures each fold has similar class balance, which is crucial
cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                     random_state=RANDOM_STATE)
results = []

for name in MODEL_NAMES:
    print(f"\n── {name} ─────────────────────────────────────────")
    fold_metrics = {"f1": [], "roc_auc": [], "precision": [],
                    "recall": [], "accuracy": []}
    fold_times = []

    for fold, (tr_idx, te_idx) in enumerate(cv.split(X, y), start=1):
        model = build_model(name)
        t0 = time.time()

        # GaussianNB needs sample_weight because it lacks class_weight param
        if name == "Gaussian Naive Bayes":
            sw = compute_sample_weight("balanced", y[tr_idx])
            model.fit(X[tr_idx], y[tr_idx], sample_weight=sw)
        else:
            model.fit(X[tr_idx], y[tr_idx])

        fit_time = time.time() - t0

        y_pred = model.predict(X[te_idx])
        try:
            y_prob = model.predict_proba(X[te_idx])[:, 1]
        except Exception:
            y_prob = y_pred.astype(float)

        # AUC undefined when model outputs a single constant (e.g. majority baseline)
        auc = (roc_auc_score(y[te_idx], y_prob)
               if len(np.unique(y_prob)) > 1 else 0.5)

        fold_metrics["f1"].append(f1_score(y[te_idx], y_pred, zero_division=0))
        fold_metrics["roc_auc"].append(auc)
        fold_metrics["precision"].append(
            precision_score(y[te_idx], y_pred, zero_division=0))
        fold_metrics["recall"].append(
            recall_score(y[te_idx], y_pred, zero_division=0))
        fold_metrics["accuracy"].append(
            accuracy_score(y[te_idx], y_pred))
        fold_times.append(fit_time)

        print(f"  fold {fold}:  F1={fold_metrics['f1'][-1]:.4f}  "
              f"AUC={auc:.4f}  "
              f"Prec={fold_metrics['precision'][-1]:.4f}  "
              f"Rec={fold_metrics['recall'][-1]:.4f}  "
              f"fit={fit_time:.1f}s")

    row = {"Model": name}
    for m, vals in fold_metrics.items():
        row[f"{m}_mean"] = float(np.mean(vals))
        row[f"{m}_std"]  = float(np.std(vals))
    row["fit_time_mean_s"] = float(np.mean(fold_times))
    results.append(row)

# ── Benchmark table ────────────────────────────────────────────────────────
bench = pd.DataFrame(results)

# Pretty display copy
display = pd.DataFrame({"Model": bench["Model"]})
for m in ["f1", "roc_auc", "precision", "recall", "accuracy"]:
    display[m.upper()] = [
        f"{row[f'{m}_mean']:.4f} ± {row[f'{m}_std']:.4f}"
        for _, row in bench.iterrows()
    ]
display["Fit Time (s)"] = bench["fit_time_mean_s"].round(2).values

print("\n" + "=" * 80)
print("BASELINE BENCHMARK RESULTS (stratified 5-fold CV, mean ± std)")
print("=" * 80)
print(display.to_string(index=False))
print("=" * 80)


# ══════════════════════════════════════════════════════════════════════════
# CSV (machine-readable, full precision)
# ══════════════════════════════════════════════════════════════════════════
csv_path = os.path.join(RESULTS_DIR, "baseline_benchmark.csv")
bench.to_csv(csv_path, index=False)
print(f"\nSaved: {csv_path}")

# ══════════════════════════════════════════════════════════════════════════
# Markdown (paste into IEEE paper / eda_summary)
# ══════════════════════════════════════════════════════════════════════════
md_path = os.path.join(RESULTS_DIR, "baseline_benchmark.md")
with open(md_path, "w") as f:
    f.write("# Baseline Benchmark — Phase 3\n\n")
    f.write(f"Stratified {N_SPLITS}-fold cross-validation on "
            f"{X.shape[0]:,} jobs x {X.shape[1]} features.  \n")
    f.write(f"Failure rate: {y.mean():.1%}.\n\n")
    try:
        f.write(display.to_markdown(index=False))
    except ImportError:
        # tabulate not installed — fall back to a plain table
        f.write(display.to_string(index=False))
    f.write("\n")
print(f"Saved: {md_path}")


# ══════════════════════════════════════════════════════════════════════════
# PART 4 — Plot figure comparing the four baselines across all five metrics (bar chart with error bars)
# ══════════════════════════════════════════════════════════════════════════
# Same style as run_eda.py
bench_plot = pd.DataFrame({
    "Model":     bench["Model"],
    "F1":        bench["f1_mean"],
    "ROC_AUC":   bench["roc_auc_mean"],
    "Precision": bench["precision_mean"],
    "Recall":    bench["recall_mean"],
    "Accuracy":  bench["accuracy_mean"],
})

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
 
models        = bench_plot["Model"].tolist()
metrics       = ["F1", "ROC_AUC", "Precision", "Recall", "Accuracy"]
metric_labels = ["F1", "ROC-AUC", "Precision", "Recall", "Accuracy"]
values        = bench_plot[metrics].values
 
n_models  = len(models)
n_metrics = len(metrics)
bar_width = 0.15
x = np.arange(n_models)
 
# Distinct, readable colors for the five metrics
colors = ["#2471A3", "#E74C3C", "#28B463", "#F39C12", "#8E44AD"]
 
fig, ax = plt.subplots(figsize=(14, 7))
 
# Grid behind the bars
ax.set_axisbelow(True)
ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
 
for i, (label, color) in enumerate(zip(metric_labels, colors)):
    offset = (i - (n_metrics - 1) / 2) * bar_width
    bars = ax.bar(x + offset, values[:, i], width=bar_width,
                  label=label, color=color, edgecolor="white",
                  linewidth=0.8)
 
    # Exact value on top of each bar (rotated so labels don't overlap)
    for bar, val in zip(bars, values[:, i]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{val:.3f}",
                ha="center", va="bottom",
                fontsize=7.5, rotation=90)
 
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=10)
ax.set_ylabel("Score")
ax.set_ylim(0, 1.15)
ax.set_yticks(np.arange(0, 1.01, 0.1))
ax.set_title("Figure 11 — Baseline Model Comparison (stratified 5-fold CV)",
             fontsize=11)
ax.legend(loc="upper left", ncol=5, frameon=False, fontsize=9)
 
plt.tight_layout()
fig_path = os.path.join(FIG_DIR, "figure_baseline_comparison.png")
plt.savefig(fig_path, bbox_inches="tight", dpi=150)
plt.close()
print(f"Saved: {fig_path}")

print("\nPhase 3 complete.")
