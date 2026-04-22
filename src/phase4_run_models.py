"""
Phase 4 - Main Models & Benchmarking
CS465 ML Project - Google Borg Job Failure Prediction
Member 4

Trains RF, XGBoost, LightGBM with GridSearchCV, evaluates on held-out test set,
produces feature importance, SHAP summary, confusion matrix, and error analysis.
"""
import sys, os
sys.path.append("D:/ML/pylibs")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (f1_score, roc_auc_score, precision_score,
                             recall_score, accuracy_score, ConfusionMatrixDisplay)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(ROOT, "data")
FIG_DIR    = os.path.join(ROOT, "outputs", "figures")
RES_DIR    = os.path.join(ROOT, "outputs", "results")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

# ── 4.1  Load data ──────────────────────────────────────────────────────────
feat_path  = os.path.join(DATA_DIR, "features_clean.csv")
clean_path = os.path.join(DATA_DIR, "borg_clean.csv")

try:
    df = pd.read_csv(feat_path)
    print(f"Loaded features_clean.csv  shape: {df.shape}")
except FileNotFoundError:
    df = pd.read_csv(clean_path)
    print(f"Fallback: loaded borg_clean.csv  shape: {df.shape}")

TARGET = "failed"
X = df.drop(columns=[TARGET])
y = df[TARGET]

print(f"Class balance:\n{y.value_counts(normalize=True).round(3).to_string()}")

# ── 4.2  Train / test split ─────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"\nTrain: {X_train.shape}  |  Test: {X_test.shape}")

# ── 4.3  Model definitions & grids ─────────────────────────────────────────
rf_param_grid = {
    "n_estimators": [100, 300],
    "max_depth":    [None, 15, 30],
    "min_samples_leaf": [1, 5],
    "class_weight": ["balanced"],
}

xgb_param_grid = {
    "n_estimators": [100, 300],
    "max_depth":    [4, 6, 8],
    "learning_rate": [0.05, 0.1],
    "scale_pos_weight": [3],
}

lgbm_param_grid = {
    "n_estimators": [100, 300],
    "max_depth":    [6, 10],
    "learning_rate": [0.05, 0.1],
    "class_weight": ["balanced"],
}

cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
SCORING = "f1"

models = {
    "Random Forest": (
        RandomForestClassifier(random_state=42),
        rf_param_grid,
    ),
    "XGBoost": (
        XGBClassifier(eval_metric="logloss", random_state=42,
                      use_label_encoder=False),
        xgb_param_grid,
    ),
    "LightGBM": (
        LGBMClassifier(random_state=42, verbose=-1),
        lgbm_param_grid,
    ),
}

# ── 4.4  GridSearchCV training ───────────────────────────────────────────────
best_estimators = {}
search_results  = {}

for name, (estimator, grid) in models.items():
    print(f"\nTuning {name}...")
    search = GridSearchCV(
        estimator, grid, cv=cv, scoring=SCORING,
        n_jobs=-1, verbose=1, refit=True,
    )
    search.fit(X_train, y_train)
    best_estimators[name] = search.best_estimator_
    search_results[name]  = search
    print(f"  Best params : {search.best_params_}")
    print(f"  Best CV F1  : {search.best_score_:.4f}")

# ── 4.5  Evaluation & benchmark table ───────────────────────────────────────
def evaluate(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "Model":     name,
        "F1":        round(f1_score(y_test, y_pred), 4),
        "ROC-AUC":   round(roc_auc_score(y_test, y_prob), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall":    round(recall_score(y_test, y_pred), 4),
        "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
    }

phase4_results = [evaluate(n, m, X_test, y_test) for n, m in best_estimators.items()]
phase4_df = pd.DataFrame(phase4_results).sort_values("F1", ascending=False)
print("\n" + "=" * 60)
print("Phase 4 Benchmark Table")
print("=" * 60)
print(phase4_df.to_string(index=False))

bench_path = os.path.join(RES_DIR, "phase4_benchmark.csv")
phase4_df.to_csv(bench_path, index=False)
print(f"\nSaved benchmark -> {bench_path}")

# ── 4.6  Feature importance + SHAP ─────────────────────────────────────────
best_model_name = phase4_df.iloc[0]["Model"]
best_model      = best_estimators[best_model_name]
print(f"\nBest model: {best_model_name}")

if hasattr(best_model, "feature_importances_"):
    importances = pd.Series(best_model.feature_importances_, index=X.columns)
    top20 = importances.nlargest(20)

    fig, ax = plt.subplots(figsize=(10, 7))
    top20.sort_values().plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title(f"Top 20 Feature Importances - {best_model_name}")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    fi_path = os.path.join(FIG_DIR, "feature_importance.png")
    plt.savefig(fi_path, dpi=150)
    plt.close()
    print(f"Saved feature_importance.png")

# SHAP on 2000-row sample
sample_idx = X_test.sample(2000, random_state=42).index
X_sample   = X_test.loc[sample_idx]

print("Computing SHAP values (2000-row sample)...")
explainer   = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_sample)

# Handle multi-output (RF returns list)
sv = shap_values[1] if isinstance(shap_values, list) else shap_values

shap.summary_plot(sv, X_sample, show=False)
shap_path = os.path.join(FIG_DIR, "shap_summary.png")
plt.savefig(shap_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved shap_summary.png")

# ── 4.7  Error analysis ──────────────────────────────────────────────────────
y_pred_best = best_model.predict(X_test)

errors = X_test.copy()
errors["true_label"] = y_test.values
errors["predicted"]  = y_pred_best

fn = errors[(errors["true_label"] == 1) & (errors["predicted"] == 0)]
fp = errors[(errors["true_label"] == 0) & (errors["predicted"] == 1)]
tp = errors[(errors["true_label"] == 1) & (errors["predicted"] == 1)]

print(f"\nFalse Negatives (missed failures): {len(fn):,}")
print(f"False Positives (false alarms):    {len(fp):,}")

key_features = ["resource_request_cpu", "resource_request_memory",
                "average_usage_cpu", "duration_seconds", "priority"]
key_features = [f for f in key_features if f in X_test.columns]

comparison = pd.DataFrame({
    "True Positives (mean)":  tp[key_features].mean(),
    "False Negatives (mean)": fn[key_features].mean(),
    "False Positives (mean)": fp[key_features].mean(),
})
print("\nError analysis comparison:")
print(comparison.round(4))

ea_path = os.path.join(RES_DIR, "error_analysis.csv")
comparison.to_csv(ea_path)
print(f"Saved error_analysis.csv")

# Confusion matrix
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_best, ax=ax,
    display_labels=["Success", "Failure"],
)
ax.set_title(f"Confusion Matrix - {best_model_name}")
cm_path = os.path.join(FIG_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=150)
plt.close()
print(f"Saved confusion_matrix.png")

# ── 4.8  Cross-dataset (Alibaba) ─────────────────────────────────────────────
alibaba_path = os.path.join(DATA_DIR, "alibaba_features.csv")
if os.path.exists(alibaba_path):
    df_ali = pd.read_csv(alibaba_path)
    assert "failed" in df_ali.columns
    common_cols = [c for c in X.columns if c in df_ali.columns]
    X_ali = df_ali[common_cols].reindex(columns=X.columns, fill_value=0)
    y_ali = df_ali["failed"]

    ali_results = []
    for name, model in best_estimators.items():
        y_pred = model.predict(X_ali)
        y_prob = model.predict_proba(X_ali)[:, 1]
        ali_results.append({
            "Model":          name,
            "F1 (Alibaba)":   round(f1_score(y_ali, y_pred), 4),
            "ROC-AUC (Alibaba)": round(roc_auc_score(y_ali, y_prob), 4),
        })

    ali_df = pd.DataFrame(ali_results)
    print("\nCross-dataset results (Alibaba):")
    print(ali_df.to_string(index=False))
    ali_df.to_csv(os.path.join(RES_DIR, "cross_dataset_alibaba.csv"), index=False)
else:
    print("\nAlibaba dataset not found. Cross-dataset validation skipped.")
    print("Document this omission in Section 5 of the IEEE paper.")
    # Save a placeholder so Step 5 checklist can note the omission
    pd.DataFrame([{"Note": "Alibaba dataset unavailable. Section 5 documents omission."}]
                 ).to_csv(os.path.join(RES_DIR, "cross_dataset_alibaba.csv"), index=False)

print("\n" + "=" * 60)
print("Phase 4 complete. All deliverables saved.")
print("=" * 60)
