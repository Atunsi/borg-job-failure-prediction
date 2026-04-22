"""
Phase 4 — Cross-Dataset Generalisation (Alibaba 2018 batch_task)
================================================================
Downloads, extracts, and aligns the Alibaba 2018 batch_task table to
the Borg feature schema, then evaluates the pre-trained Random Forest,
XGBoost, and LightGBM models on it.

Borg feature matrix (features_clean.csv columns used for training):
  scheduling_class, priority, collection_type,
  resource_request_cpu, resource_request_memory,
  average_usage_cpu, average_usage_memory,
  maximum_usage_memory, random_sample_usage_cpu,
  assigned_memory, page_cache_memory,
  duration_seconds,
  cpu_dist_mean, cpu_dist_std, cpu_dist_max, cpu_dist_skew,
  hit_timeout, cpu_utilization_ratio, memory_pressure

Alibaba batch_task columns:
  task_name, instance_num, job_name, task_type,
  status,                       → failed
  start_time, end_time,         → duration_seconds
  plan_cpu,                     → resource_request_cpu  (÷100 → cores)
  plan_mem                      → resource_request_memory (÷100 → fraction)

Unmapped Borg features are filled with 0 (column median = 0 after imputation
is consistent with the SOP fallback: fill_value=0).
"""
import sys, os, tarfile, pickle
sys.path.append("D:/ML/pylibs")

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.metrics import (f1_score, roc_auc_score, precision_score,
                             recall_score, accuracy_score)

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(ROOT, "data")
ALI_DIR     = os.path.join(DATA_DIR, "alibaba")
RES_DIR     = os.path.join(ROOT, "outputs", "results")
MODELS_PKL  = os.path.join(ROOT, "outputs", "models", "best_estimators.pkl")
BORG_FEAT   = os.path.join(DATA_DIR, "features_clean.csv")

# ── Borg training feature order ─────────────────────────────────────────────
borg_cols = [c for c in pd.read_csv(BORG_FEAT, nrows=0).columns if c != "failed"]
print(f"Borg feature columns ({len(borg_cols)}): {borg_cols}")

# ── Extract batch_task.tar.gz ────────────────────────────────────────────────
tarball    = os.path.join(ALI_DIR, "batch_task.tar.gz")
csv_path   = os.path.join(ALI_DIR, "batch_task.csv")

if not os.path.exists(csv_path):
    print(f"Extracting {tarball} ...")
    with tarfile.open(tarball, "r:gz") as tf:
        tf.extractall(path=ALI_DIR)
    # Find extracted file (may be in a subdirectory)
    for root, dirs, files in os.walk(ALI_DIR):
        for fn in files:
            if fn == "batch_task.csv":
                src = os.path.join(root, fn)
                if src != csv_path:
                    os.rename(src, csv_path)
                break
    print("Extraction done.")
else:
    print(f"Already extracted: {csv_path}")

# ── Load batch_task ──────────────────────────────────────────────────────────
print("Loading batch_task.csv ...")
COLS = ["task_name", "instance_num", "job_name", "task_type",
        "status", "start_time", "end_time", "plan_cpu", "plan_mem"]
df = pd.read_csv(csv_path, header=None, names=COLS, low_memory=False)
print(f"  Shape: {df.shape}")
print(f"  status values: {df['status'].value_counts().to_dict()}")

# ── Build failure label ──────────────────────────────────────────────────────
# Alibaba batch_task status semantics (from trace documentation and prior work):
#   "Terminated" = task completed successfully (finished its run)
#   "Failed"     = task explicitly failed
#   "Killed"     = task was killed (treated as failure)
#   "Cancelled"  = task was cancelled (treated as failure)
#   "Running"    = still executing — unknown final outcome → EXCLUDE
#   "Waiting"    = not yet started — unknown final outcome → EXCLUDE
#
# Only rows with a known terminal state are used for evaluation.
status_counts = df["status"].value_counts()
print("\nStatus distribution:")
print(status_counts)

SUCCESS_STATUSES = {"Terminated", "terminated", "Succeeded", "succeeded"}
FAIL_STATUSES    = {"Failed", "failed", "Killed", "killed", "Cancelled", "cancelled"}
TERMINAL_STATUSES = SUCCESS_STATUSES | FAIL_STATUSES

df = df[df["status"].isin(TERMINAL_STATUSES)].copy()
df["failed"] = df["status"].apply(lambda s: 1 if s in FAIL_STATUSES else 0)

print(f"\nRows with known terminal state: {len(df):,}")
print(f"Failed label distribution:")
print(df["failed"].value_counts())
print(f"Failure rate: {df['failed'].mean():.3%}")

# ── Feature engineering ──────────────────────────────────────────────────────
# plan_cpu: 100 = 1 core → divide by 100 to get cores
df["resource_request_cpu"]    = pd.to_numeric(df["plan_cpu"], errors="coerce") / 100.0
# plan_mem: already normalised [0,100] → divide by 100 for [0,1] fraction
df["resource_request_memory"] = pd.to_numeric(df["plan_mem"], errors="coerce") / 100.0

# Duration in seconds (timestamps already in seconds per schema)
df["start_time"] = pd.to_numeric(df["start_time"], errors="coerce")
df["end_time"]   = pd.to_numeric(df["end_time"],   errors="coerce")
df["duration_seconds"] = (df["end_time"] - df["start_time"]).clip(lower=0)

# hit_timeout proxy: duration >= 295 s
df["hit_timeout"] = (df["duration_seconds"] >= 295).astype(int)

# instance_num as proxy for scheduling_class / priority
df["scheduling_class"] = 0
df["priority"]         = 0
df["collection_type"]  = 0

# ── Build aligned feature matrix ─────────────────────────────────────────────
X_ali = df.reindex(columns=borg_cols, fill_value=0)

# Fill NaN from failed conversions
for col in X_ali.columns:
    if X_ali[col].isna().any():
        X_ali[col] = X_ali[col].fillna(0)

y_ali = df["failed"]

print(f"\nAligned feature matrix: {X_ali.shape}")
print(f"Class balance:\n{y_ali.value_counts(normalize=True).round(3)}")

# Sanity: ensure no NaN
assert X_ali.isna().sum().sum() == 0, "NaN values remain in feature matrix!"
assert y_ali.isna().sum() == 0,       "NaN values in labels!"

# ── Save aligned feature matrix ───────────────────────────────────────────────
out_path = os.path.join(DATA_DIR, "alibaba_features.csv")
out_df = X_ali.copy()
out_df["failed"] = y_ali.values
out_df.to_csv(out_path, index=False)
print(f"\nSaved aligned feature matrix -> {out_path}  ({out_df.shape})")

# ── Load trained models ───────────────────────────────────────────────────────
print(f"\nLoading trained models from {MODELS_PKL} ...")
with open(MODELS_PKL, "rb") as f:
    best_estimators = pickle.load(f)
print(f"  Models available: {list(best_estimators.keys())}")

# ── Evaluate each model on Alibaba data ──────────────────────────────────────
ali_results = []
for name, model in best_estimators.items():
    y_pred = model.predict(X_ali)
    y_prob = model.predict_proba(X_ali)[:, 1]
    row = {
        "Model":             name,
        "F1 (Alibaba)":      round(f1_score(y_ali, y_pred), 4),
        "ROC-AUC (Alibaba)": round(roc_auc_score(y_ali, y_prob), 4),
        "Precision (Alibaba)": round(precision_score(y_ali, y_pred), 4),
        "Recall (Alibaba)":    round(recall_score(y_ali, y_pred), 4),
        "Accuracy (Alibaba)":  round(accuracy_score(y_ali, y_pred), 4),
    }
    ali_results.append(row)
    print(f"\n  {name}:")
    for k, v in row.items():
        if k != "Model":
            print(f"    {k}: {v}")

ali_df = pd.DataFrame(ali_results)
print("\n" + "=" * 60)
print("Cross-Dataset Generalisation Results (Alibaba 2018)")
print("=" * 60)
print(ali_df.to_string(index=False))

out_csv = os.path.join(RES_DIR, "cross_dataset_alibaba.csv")
ali_df.to_csv(out_csv, index=False)
print(f"\nSaved -> {out_csv}")
print("Cross-dataset validation complete.")
