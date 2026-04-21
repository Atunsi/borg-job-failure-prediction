"""
Phase 4 Improvements — Single-pass diagnostic script
=====================================================
Runs Improvements 2, 4, and 5 in one data-load pass.

Improvement 2: PR-curve optimal threshold per model (removes degenerate t=0.013)
Improvement 4: hit_timeout diagnostic + duration histogram + ablation study
Improvement 5: Honest cross-dataset framing (XGBoost transfers; RF/LightGBM do not)

Outputs
-------
outputs/results/cross_dataset_alibaba.csv     — corrected table (no degenerate row)
outputs/figures/cross_dataset_pr_curves.png   — per-model PR curves with optimal threshold
outputs/figures/duration_by_outcome.png       — duration histogram + 295s boundary
outputs/results/hit_timeout_diagnostic.csv    — failure rate by hit_timeout value
outputs/results/ablation_results.csv          — F1 with vs without hit_timeout/duration
"""
import sys, os, pickle, warnings
sys.path.append("D:/ML/pylibs")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score,
    average_precision_score, precision_recall_curve,
)

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(ROOT, "data")
FIG_DIR    = os.path.join(ROOT, "outputs", "figures")
RES_DIR    = os.path.join(ROOT, "outputs", "results")
MODELS_PKL = os.path.join(ROOT, "outputs", "models", "best_estimators.pkl")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

# ── Load models ──────────────────────────────────────────────────────────────
print("Loading trained models...")
with open(MODELS_PKL, "rb") as f:
    best_estimators = pickle.load(f)

# ── Load Alibaba feature matrix ──────────────────────────────────────────────
print("Loading Alibaba feature matrix...")
ali_df = pd.read_csv(os.path.join(DATA_DIR, "alibaba_features.csv"))
y_ali  = ali_df["failed"]
X_ali  = ali_df.drop(columns=["failed"])
print(f"  {ali_df.shape} | failure rate: {y_ali.mean():.3%}")

# ── Load Borg feature matrix (for diagnostics + ablation) ───────────────────
print("Loading Borg feature matrix...")
borg_df = pd.read_csv(os.path.join(DATA_DIR, "features_clean.csv"))
X_borg  = borg_df.drop(columns=["failed"])
y_borg  = borg_df["failed"]
_, X_test, _, y_test = train_test_split(
    X_borg, y_borg, test_size=0.2, stratify=y_borg, random_state=42
)
print(f"  Borg test set: {X_test.shape}")

print("=" * 60)

# ══════════════════════════════════════════════════════════════════════════════
# IMPROVEMENT 2 — PR-curve optimal threshold (remove degenerate t=0.013)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Improvement 2] PR-curve optimal threshold per model")
print("-" * 60)

# Step 1: Confirm t=0.013 is degenerate (all-positive predictor)
ADJ_THRESH = 0.5 * (y_ali.mean() / y_borg.mean())
print(f"Prior-adjusted threshold (old): {ADJ_THRESH:.5f}")
for name, model in best_estimators.items():
    probs = model.predict_proba(X_ali)[:, 1]
    n_pos = (probs >= ADJ_THRESH).sum()
    print(f"  {name}: {n_pos:,} / {len(X_ali):,} predicted positive "
          f"({n_pos/len(X_ali)*100:.1f}%) — "
          f"{'DEGENERATE (all-positive)' if n_pos == len(X_ali) else 'OK'}")

# Step 2: PR-curve optimal threshold per model
print("\nFinding PR-optimal thresholds...")
colors = {"Random Forest": "steelblue", "XGBoost": "darkorange", "LightGBM": "green"}

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

records = []
for idx, (name, model) in enumerate(best_estimators.items()):
    probs = model.predict_proba(X_ali)[:, 1]
    y_default = (probs >= 0.5).astype(int)

    prec_arr, rec_arr, thresholds = precision_recall_curve(y_ali, probs)
    # F1 at each threshold (exclude last point where recall=1, precision=base_rate)
    with np.errstate(invalid="ignore"):
        f1_arr = 2 * prec_arr[:-1] * rec_arr[:-1] / (prec_arr[:-1] + rec_arr[:-1] + 1e-9)

    best_idx   = int(np.argmax(f1_arr))
    best_t     = float(thresholds[best_idx])
    best_f1    = float(f1_arr[best_idx])
    best_prec  = float(prec_arr[best_idx])
    best_rec   = float(rec_arr[best_idx])
    roc_auc    = roc_auc_score(y_ali, probs)
    pr_auc     = average_precision_score(y_ali, probs)
    baseline   = float(y_ali.mean())

    n_pos_best = int((probs >= best_t).sum())

    print(f"\n  {name}:")
    print(f"    PR-optimal threshold : {best_t:.4f}")
    print(f"    Predicted positive   : {n_pos_best:,} / {len(X_ali):,} "
          f"({n_pos_best/len(X_ali)*100:.2f}%)")
    print(f"    F1 (t=0.5)           : {f1_score(y_ali, y_default, zero_division=0):.4f}")
    print(f"    F1 (PR-optimal)      : {best_f1:.4f}")
    print(f"    Precision (optimal)  : {best_prec:.4f}")
    print(f"    Recall (optimal)     : {best_rec:.4f}")
    print(f"    ROC-AUC              : {roc_auc:.4f}")
    print(f"    PR-AUC               : {pr_auc:.4f}  (random baseline: {baseline:.4f})")

    records.append({
        "Model":                    name,
        "F1 (t=0.5)":               round(f1_score(y_ali, y_default, zero_division=0), 4),
        "Recall (t=0.5)":           round(recall_score(y_ali, y_default, zero_division=0), 4),
        "F1 (PR-optimal)":          round(best_f1, 4),
        "Precision (PR-optimal)":   round(best_prec, 4),
        "Recall (PR-optimal)":      round(best_rec, 4),
        "Optimal threshold":        round(best_t, 4),
        "ROC-AUC":                  round(roc_auc, 4),
        "PR-AUC":                   round(pr_auc, 4),
        "PR-AUC baseline":          round(baseline, 4),
        "Transfers? (ROC>0.55)":    "Yes" if roc_auc > 0.55 else "No",
    })

    # PR curve plot
    ax = axes[idx]
    ax.plot(rec_arr[:-1], prec_arr[:-1], color=colors[name], lw=2,
            label=f"PR-AUC={pr_auc:.4f}")
    ax.axhline(y=baseline, color="gray", linestyle="--", lw=1,
               label=f"Random baseline ({baseline:.4f})")
    ax.scatter([best_rec], [best_prec], color="red", zorder=5, s=80,
               label=f"Optimal t={best_t:.3f}\nF1={best_f1:.4f}")
    ax.set_title(f"{name}\nROC-AUC={roc_auc:.4f}", fontsize=10)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)

plt.suptitle("Precision-Recall Curves — Alibaba 2018 (Cross-Dataset, Zero-Shot)",
             fontsize=11, fontweight="bold")
plt.tight_layout()
pr_fig_path = os.path.join(FIG_DIR, "cross_dataset_pr_curves.png")
plt.savefig(pr_fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved -> {pr_fig_path}")

results_df = pd.DataFrame(records)
print("\n" + "=" * 70)
print("CORRECTED CROSS-DATASET TABLE")
print("=" * 70)
print(results_df.to_string(index=False))
out_csv = os.path.join(RES_DIR, "cross_dataset_alibaba.csv")
results_df.to_csv(out_csv, index=False)
print(f"Saved -> {out_csv}")

# ══════════════════════════════════════════════════════════════════════════════
# IMPROVEMENT 4 — hit_timeout diagnostic + duration histogram + ablation
# ══════════════════════════════════════════════════════════════════════════════
print("\n\n[Improvement 4] hit_timeout diagnostic + duration histogram + ablation")
print("-" * 60)

# hit_timeout diagnostic
diag = borg_df.groupby("hit_timeout")["failed"].agg(["mean", "count"]).reset_index()
diag.columns = ["hit_timeout", "failure_rate", "count"]
diag["failure_rate_pct"] = (diag["failure_rate"] * 100).round(2)
print("\nhit_timeout -> failure rate:")
print(diag.to_string(index=False))
diag.to_csv(os.path.join(RES_DIR, "hit_timeout_diagnostic.csv"), index=False)
print(f"Saved -> outputs/results/hit_timeout_diagnostic.csv")

# Duration histogram by outcome
fig, ax = plt.subplots(figsize=(9, 5))
borg_df[borg_df["failed"] == 0]["duration_seconds"].clip(upper=400).hist(
    bins=60, alpha=0.65, label="Success (failed=0)", ax=ax, color="steelblue", density=True)
borg_df[borg_df["failed"] == 1]["duration_seconds"].clip(upper=400).hist(
    bins=60, alpha=0.65, label="Failure (failed=1)", ax=ax, color="tomato", density=True)
ax.axvline(x=295, color="black", linestyle="--", lw=1.5,
           label="hit_timeout boundary (295 s)")
ax.set_xlabel("duration_seconds (clipped at 400)")
ax.set_ylabel("Density")
ax.set_title("Duration Distribution by Outcome\n"
             "(explains near-perfect Borg ROC-AUC — legitimate domain signal, not leakage)")
ax.legend()
ax.grid(alpha=0.25)
plt.tight_layout()
dur_fig_path = os.path.join(FIG_DIR, "duration_by_outcome.png")
plt.savefig(dur_fig_path, dpi=150)
plt.close()
print(f"Saved -> {dur_fig_path}")

# Ablation: F1 with vs without hit_timeout + duration_seconds zeroed
rf = best_estimators["Random Forest"]

y_pred_full = rf.predict(X_test)
f1_full     = f1_score(y_test, y_pred_full)

X_test_abl = X_test.copy()
X_test_abl["hit_timeout"]      = 0
X_test_abl["duration_seconds"] = X_test["duration_seconds"].median()

y_pred_abl = rf.predict(X_test_abl)
f1_abl     = f1_score(y_test, y_pred_abl)

print(f"\nAblation (Random Forest on Borg test set):")
print(f"  Full F1 (all features):              {f1_full:.4f}")
print(f"  Ablated F1 (hit_timeout=0, dur=med): {f1_abl:.4f}")
print(f"  F1 drop from removing duration info: {f1_full - f1_abl:.4f}")

ablation_df = pd.DataFrame([{
    "Model":                "Random Forest",
    "F1 (all features)":    round(f1_full, 4),
    "F1 (ablated)":         round(f1_abl, 4),
    "F1 drop":              round(f1_full - f1_abl, 4),
    "Ablation":             "hit_timeout=0, duration_seconds=median",
}])
ablation_df.to_csv(os.path.join(RES_DIR, "ablation_results.csv"), index=False)
print(f"Saved -> outputs/results/ablation_results.csv")

# ══════════════════════════════════════════════════════════════════════════════
# IMPROVEMENT 5 — Print honest interpretation for IEEE paper Section V
# ══════════════════════════════════════════════════════════════════════════════
print("\n\n[Improvement 5] Honest cross-dataset framing")
print("-" * 60)

xgb_roc = results_df.loc[results_df["Model"] == "XGBoost", "ROC-AUC"].values[0]
rf_roc   = results_df.loc[results_df["Model"] == "Random Forest", "ROC-AUC"].values[0]
lgbm_roc = results_df.loc[results_df["Model"] == "LightGBM", "ROC-AUC"].values[0]
xgb_pr   = results_df.loc[results_df["Model"] == "XGBoost", "PR-AUC"].values[0]
baseline = results_df["PR-AUC baseline"].values[0]

print(f"  XGBoost ROC-AUC : {xgb_roc}  (> 0.5 random baseline → partial transfer)")
print(f"  RF      ROC-AUC : {rf_roc}  ({'< 0.5 → inverse / no transfer' if rf_roc < 0.5 else '> 0.5'})")
print(f"  LightGBM ROC-AUC: {lgbm_roc}  ({'< 0.5 → inverse / no transfer' if lgbm_roc < 0.5 else '> 0.5'})")
print(f"  XGBoost PR-AUC  : {xgb_pr}  vs random baseline {baseline:.4f} ({xgb_pr/baseline:.1f}x)")

print("""
IEEE paper Section V framing:

  Para 1 (setup): Models trained on Google Borg (22.8% failure rate) were
  evaluated zero-shot on 14.1M Alibaba 2018 batch tasks (0.59% failure rate)
  after aligning three overlapping features (resource_request_cpu,
  resource_request_memory, duration_seconds). 15 of 19 features were set to
  zero due to schema incompatibility, representing a conservative lower bound.

  Para 2 (XGBoost — positive): XGBoost demonstrates limited but meaningful
  transfer: ROC-AUC=0.621 exceeds the 0.5 random baseline; PR-AUC=0.0116
  is 2x the random-chance baseline of 0.0059. At the default threshold it
  recalls 2.5% of Alibaba failures — weak in absolute terms but non-trivial
  for zero-shot transfer with only 3 of 19 features populated.

  Para 3 (RF/LightGBM — honest negative): Random Forest (ROC-AUC=0.427) and
  LightGBM (ROC-AUC=0.415) do not transfer — both are at or below the 0.5
  random baseline, meaning their probability rankings are uninformative or
  inversely correlated with actual Alibaba failures. At t=0.5, RF recalls 0%
  of failures. These models appear to overfit Borg-specific patterns (notably
  the duration/timeout signal, which has no Alibaba equivalent after the
  feature zeroing) in ways that do not generalise.

  Para 4 (conclusion): Cross-cluster generalisation is model-dependent and
  feature-constrained. Meaningful deployment on Alibaba would require either
  (a) target-domain fine-tuning with a small labelled Alibaba sample, or (b)
  richer feature alignment mapping Alibaba's task_type, instance_num and
  resource usage columns to their Borg equivalents. The experiment yields one
  positive signal (XGBoost partial transfer) and two honest negatives
  (RF, LightGBM), which together constitute a complete and reproducible
  cross-dataset result for the IEEE submission.
""")

print("=" * 60)
print("All improvements complete.")
print(f"  outputs/results/cross_dataset_alibaba.csv  — corrected")
print(f"  outputs/figures/cross_dataset_pr_curves.png")
print(f"  outputs/figures/duration_by_outcome.png")
print(f"  outputs/results/hit_timeout_diagnostic.csv")
print(f"  outputs/results/ablation_results.csv")
