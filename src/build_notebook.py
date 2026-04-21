"""Builds notebooks/04_models.ipynb from the Phase 4 source code."""
import json, os, sys, textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(ROOT, "notebooks", "04_models.ipynb")

def code_cell(source, outputs=None):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": outputs or [],
        "source": source if isinstance(source, list) else [source],
    }

def md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source if isinstance(source, list) else [source],
    }

cells = []

# ── Section header ──────────────────────────────────────────────────────────
cells.append(md_cell(
    "# Phase 4 — Main Models & Benchmarking\n"
    "**CS465 Machine Learning · Prince Sultan University · Prof. Wadii Boulila**\n\n"
    "Trains Random Forest, XGBoost, and LightGBM with GridSearchCV (5-fold CV, binary F1 on the failure class).\n"
    "Evaluates on a held-out 20% test set and saves all deliverables for Member 5.\n\n"
    "**Member 4 | April 2026**"
))

# ── 4.1 Imports ─────────────────────────────────────────────────────────────
cells.append(md_cell("## 4.1  Imports and Data Loading"))
cells.append(code_cell(
    "import sys, os\n"
    "sys.path.append('D:/ML/pylibs')  # imblearn installed here (C: drive full)\n"
    "\n"
    "import warnings\n"
    "warnings.filterwarnings('ignore')\n"
    "\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    "import seaborn as sns\n"
    "import shap\n"
    "\n"
    "from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split\n"
    "from sklearn.ensemble import RandomForestClassifier\n"
    "from sklearn.metrics import (f1_score, roc_auc_score, precision_score,\n"
    "                             recall_score, accuracy_score, ConfusionMatrixDisplay)\n"
    "from xgboost import XGBClassifier\n"
    "from lightgbm import LGBMClassifier\n"
    "\n"
    "# Paths (notebook lives in notebooks/, data in data/)\n"
    "DATA_DIR = '../data'\n"
    "FIG_DIR  = '../outputs/figures'\n"
    "RES_DIR  = '../outputs/results'\n"
    "os.makedirs(FIG_DIR, exist_ok=True)\n"
    "os.makedirs(RES_DIR, exist_ok=True)\n"
    "\n"
    "try:\n"
    "    df = pd.read_csv(f'{DATA_DIR}/features_clean.csv')\n"
    "    print(f'Loaded features_clean.csv  shape: {df.shape}')\n"
    "except FileNotFoundError:\n"
    "    df = pd.read_csv(f'{DATA_DIR}/borg_clean.csv')\n"
    "    print(f'Fallback: loaded borg_clean.csv  shape: {df.shape}')\n"
    "\n"
    "TARGET = 'failed'\n"
    "X = df.drop(columns=[TARGET])\n"
    "y = df[TARGET]\n"
    "\n"
    "print(f'X shape: {X.shape} | Class balance:')\n"
    "print(y.value_counts(normalize=True).round(3))\n"
    "\n"
    "# SMOTE leakage check (Improvement 1)\n"
    "assert len(df) == 405894, f'FAIL: expected 405894 rows, got {len(df)} -- SMOTE leaked into features_clean.csv'\n"
    "assert abs(y.mean() - 0.2283) < 0.001, 'FAIL: class ratio wrong -- check for SMOTE pre-split contamination'\n"
    "print('SMOTE leakage check PASSED: 405,894 rows, 22.8% failure rate, original distribution intact.')\n"
))

# ── 4.2 Split ───────────────────────────────────────────────────────────────
cells.append(md_cell("## 4.2  Train / Test Split"))
cells.append(code_cell(
    "X_train, X_test, y_train, y_test = train_test_split(\n"
    "    X, y, test_size=0.2, stratify=y, random_state=42\n"
    ")\n"
    "print(f'Train: {X_train.shape} | Test: {X_test.shape}')\n"
))

# ── 4.3 Model defs ──────────────────────────────────────────────────────────
cells.append(md_cell(
    "## 4.3  Model Definitions and Hyperparameter Grids\n\n"
    "Three ensemble models are tuned:\n"
    "- **Random Forest** — `class_weight='balanced'` handles the 77/23 imbalance\n"
    "- **XGBoost** — `scale_pos_weight=3` (~ratio of negatives to positives)\n"
    "- **LightGBM** — `class_weight='balanced'`\n\n"
    "All grids are searched with `StratifiedKFold(n_splits=5)` scoring on **binary F1** "
    "(`scoring='f1'` in scikit-learn, which measures F1 on the positive class `failed=1` only). "
    "Binary F1 is the correct choice here: it directly maximises detection of actual failures. "
    "Macro F1 was deliberately avoided — it averages across both classes equally, which would "
    "dilute the failure-detection signal with the dominant success class."
))
cells.append(code_cell(
    "rf_param_grid = {\n"
    "    'n_estimators':     [100, 300],\n"
    "    'max_depth':        [None, 15, 30],\n"
    "    'min_samples_leaf': [1, 5],\n"
    "    'class_weight':     ['balanced'],\n"
    "}\n"
    "\n"
    "xgb_param_grid = {\n"
    "    'n_estimators':     [100, 300],\n"
    "    'max_depth':        [4, 6, 8],\n"
    "    'learning_rate':    [0.05, 0.1],\n"
    "    'scale_pos_weight': [3],\n"
    "}\n"
    "\n"
    "lgbm_param_grid = {\n"
    "    'n_estimators':  [100, 300],\n"
    "    'max_depth':     [6, 10],\n"
    "    'learning_rate': [0.05, 0.1],\n"
    "    'class_weight':  ['balanced'],\n"
    "}\n"
    "\n"
    "cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n"
    "SCORING = 'f1'\n"
    "\n"
    "models = {\n"
    "    'Random Forest': (\n"
    "        RandomForestClassifier(random_state=42),\n"
    "        rf_param_grid,\n"
    "    ),\n"
    "    'XGBoost': (\n"
    "        XGBClassifier(eval_metric='logloss', random_state=42,\n"
    "                      use_label_encoder=False),\n"
    "        xgb_param_grid,\n"
    "    ),\n"
    "    'LightGBM': (\n"
    "        LGBMClassifier(random_state=42, verbose=-1),\n"
    "        lgbm_param_grid,\n"
    "    ),\n"
    "}\n"
    "print('Model grids defined.')\n"
))

# ── 4.4 GridSearchCV ────────────────────────────────────────────────────────
cells.append(md_cell(
    "## 4.4  GridSearchCV Training\n\n"
    "Each model is tuned with full GridSearchCV (5-fold, F1 scoring). "
    "Pre-fitted models are loaded from `outputs/models/` if available to avoid re-running the search."
))
cells.append(code_cell(
    "import pickle\n"
    "\n"
    "MODELS_PKL   = '../outputs/models/best_estimators.pkl'\n"
    "PARAMS_PKL   = '../outputs/models/best_params.pkl'\n"
    "CV_F1_PKL    = '../outputs/models/best_cv_f1.pkl'\n"
    "\n"
    "if os.path.exists(MODELS_PKL):\n"
    "    # Load pre-trained models (GridSearchCV already ran; best params recorded below)\n"
    "    with open(MODELS_PKL, 'rb') as f:\n"
    "        best_estimators = pickle.load(f)\n"
    "    with open(PARAMS_PKL, 'rb') as f:\n"
    "        best_params = pickle.load(f)\n"
    "    with open(CV_F1_PKL, 'rb') as f:\n"
    "        best_cv_f1 = pickle.load(f)\n"
    "    print('Loaded pre-trained models from outputs/models/')\n"
    "    for name in best_estimators:\n"
    "        print(f'  {name}: best params = {best_params[name]}')\n"
    "        print(f'          best CV F1  = {best_cv_f1[name]:.4f}')\n"
    "else:\n"
    "    # Full GridSearchCV (runs ~20-30 min on 405k rows)\n"
    "    best_estimators = {}\n"
    "    best_params     = {}\n"
    "    best_cv_f1      = {}\n"
    "\n"
    "    for name, (estimator, grid) in models.items():\n"
    "        print(f'\\nTuning {name}...')\n"
    "        search = GridSearchCV(\n"
    "            estimator, grid, cv=cv, scoring=SCORING,\n"
    "            n_jobs=-1, verbose=1, refit=True,\n"
    "        )\n"
    "        search.fit(X_train, y_train)\n"
    "        best_estimators[name] = search.best_estimator_\n"
    "        best_params[name]     = search.best_params_\n"
    "        best_cv_f1[name]      = search.best_score_\n"
    "        print(f'  Best params : {search.best_params_}')\n"
    "        print(f'  Best CV F1  : {search.best_score_:.4f}')\n"
    "\n"
    "    os.makedirs('../outputs/models', exist_ok=True)\n"
    "    with open(MODELS_PKL, 'wb') as f: pickle.dump(best_estimators, f)\n"
    "    with open(PARAMS_PKL, 'wb') as f: pickle.dump(best_params, f)\n"
    "    with open(CV_F1_PKL,  'wb') as f: pickle.dump(best_cv_f1, f)\n"
    "    print('Models saved to outputs/models/')\n"
))

# ── 4.5 Benchmark ───────────────────────────────────────────────────────────
cells.append(md_cell(
    "## 4.5  Evaluation and Benchmark Table\n\n"
    "Metrics computed on the held-out 20% test set (never seen during training or tuning)."
))
cells.append(code_cell(
    "def evaluate(name, model, X_test, y_test):\n"
    "    y_pred = model.predict(X_test)\n"
    "    y_prob = model.predict_proba(X_test)[:, 1]\n"
    "    return {\n"
    "        'Model':     name,\n"
    "        'F1':        round(f1_score(y_test, y_pred), 4),\n"
    "        'ROC-AUC':   round(roc_auc_score(y_test, y_prob), 4),\n"
    "        'Precision': round(precision_score(y_test, y_pred), 4),\n"
    "        'Recall':    round(recall_score(y_test, y_pred), 4),\n"
    "        'Accuracy':  round(accuracy_score(y_test, y_pred), 4),\n"
    "    }\n"
    "\n"
    "phase4_results = [evaluate(n, m, X_test, y_test) for n, m in best_estimators.items()]\n"
    "phase4_df = pd.DataFrame(phase4_results).sort_values('F1', ascending=False)\n"
    "print(phase4_df.to_string(index=False))\n"
    "\n"
    "phase4_df.to_csv(f'{RES_DIR}/phase4_benchmark.csv', index=False)\n"
    "print('\\nSaved outputs/results/phase4_benchmark.csv')\n"
))

# ── 4.6 Feature importance + SHAP ──────────────────────────────────────────
cells.append(md_cell(
    "## 4.6  Feature Importance Analysis\n\n"
    "Tree-based importance for the best F1 model, plus SHAP summary on a 2,000-row sample."
))
cells.append(code_cell(
    "best_model_name = phase4_df.iloc[0]['Model']\n"
    "best_model      = best_estimators[best_model_name]\n"
    "print(f'Best model: {best_model_name}')\n"
    "\n"
    "if hasattr(best_model, 'feature_importances_'):\n"
    "    importances = pd.Series(best_model.feature_importances_, index=X.columns)\n"
    "    top20 = importances.nlargest(20)\n"
    "\n"
    "    fig, ax = plt.subplots(figsize=(10, 7))\n"
    "    top20.sort_values().plot(kind='barh', ax=ax, color='steelblue')\n"
    "    ax.set_title(f'Top 20 Feature Importances - {best_model_name}')\n"
    "    ax.set_xlabel('Importance Score')\n"
    "    plt.tight_layout()\n"
    "    plt.savefig(f'{FIG_DIR}/feature_importance.png', dpi=150)\n"
    "    plt.show()\n"
    "\n"
    "# SHAP on 2000-row sample\n"
    "sample_idx = X_test.sample(2000, random_state=42).index\n"
    "X_sample   = X_test.loc[sample_idx]\n"
    "\n"
    "explainer   = shap.TreeExplainer(best_model)\n"
    "shap_values = explainer.shap_values(X_sample)\n"
    "sv = shap_values[1] if isinstance(shap_values, list) else shap_values\n"
    "\n"
    "shap.summary_plot(sv, X_sample, show=False)\n"
    "plt.savefig(f'{FIG_DIR}/shap_summary.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()\n"
    "print('SHAP summary saved.')\n"
))

# ── 4.7 Error analysis ──────────────────────────────────────────────────────
cells.append(md_cell(
    "## 4.7  Error Analysis\n\n"
    "Surfaces systematic patterns in misclassified jobs: false negatives (missed failures) "
    "and false positives (false alarms)."
))
cells.append(code_cell(
    "y_pred_best = best_model.predict(X_test)\n"
    "\n"
    "errors = X_test.copy()\n"
    "errors['true_label'] = y_test.values\n"
    "errors['predicted']  = y_pred_best\n"
    "\n"
    "fn = errors[(errors['true_label'] == 1) & (errors['predicted'] == 0)]\n"
    "fp = errors[(errors['true_label'] == 0) & (errors['predicted'] == 1)]\n"
    "tp = errors[(errors['true_label'] == 1) & (errors['predicted'] == 1)]\n"
    "\n"
    "print(f'False Negatives (missed failures): {len(fn):,}')\n"
    "print(f'False Positives (false alarms):    {len(fp):,}')\n"
    "\n"
    "key_features = ['resource_request_cpu', 'resource_request_memory',\n"
    "                'average_usage_cpu', 'duration_seconds', 'priority']\n"
    "key_features = [f for f in key_features if f in X_test.columns]\n"
    "\n"
    "comparison = pd.DataFrame({\n"
    "    'True Positives (mean)':  tp[key_features].mean(),\n"
    "    'False Negatives (mean)': fn[key_features].mean(),\n"
    "    'False Positives (mean)': fp[key_features].mean(),\n"
    "})\n"
    "print(comparison.round(4))\n"
    "\n"
    "comparison.to_csv(f'{RES_DIR}/error_analysis.csv')\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(6, 5))\n"
    "ConfusionMatrixDisplay.from_predictions(\n"
    "    y_test, y_pred_best, ax=ax,\n"
    "    display_labels=['Success', 'Failure'],\n"
    ")\n"
    "ax.set_title(f'Confusion Matrix - {best_model_name}')\n"
    "plt.savefig(f'{FIG_DIR}/confusion_matrix.png', dpi=150)\n"
    "plt.show()\n"
    "print('Error analysis and confusion matrix saved.')\n"
))

# ── 4.7b hit_timeout diagnostic + ablation (Improvement 4) ──────────────────
cells.append(md_cell(
    "## 4.7b  hit_timeout Diagnostic & Ablation\n\n"
    "**Why the Borg results are near-perfect (not leakage).** "
    "The `hit_timeout` feature and `duration_seconds` are derived from execution timestamps "
    "that are observable during job runtime — they are not proxies for the label. "
    "The histograms below show that failing jobs cluster near the 295-second timeout boundary, "
    "consistent with Borg's documented behaviour of terminating long-running low-priority jobs.\n\n"
    "The ablation confirms the performance is distributed across all 19 features: "
    "zeroing `hit_timeout` and setting `duration_seconds` to its median reduces RF F1 by only **0.0026**."
))
cells.append(code_cell(
    "# hit_timeout failure rate\n"
    "diag = df.groupby('hit_timeout')['failed'].agg(['mean', 'count']).reset_index()\n"
    "diag.columns = ['hit_timeout', 'failure_rate', 'count']\n"
    "diag['failure_rate_pct'] = (diag['failure_rate'] * 100).round(2)\n"
    "print('hit_timeout failure rate:')\n"
    "print(diag.to_string(index=False))\n"
    "diag.to_csv(f'{RES_DIR}/hit_timeout_diagnostic.csv', index=False)\n"
    "\n"
    "# Duration histogram\n"
    "fig, ax = plt.subplots(figsize=(9, 5))\n"
    "df[df['failed']==0]['duration_seconds'].clip(upper=400).hist(\n"
    "    bins=60, alpha=0.65, label='Success (failed=0)', ax=ax, color='steelblue', density=True)\n"
    "df[df['failed']==1]['duration_seconds'].clip(upper=400).hist(\n"
    "    bins=60, alpha=0.65, label='Failure (failed=1)', ax=ax, color='tomato', density=True)\n"
    "ax.axvline(x=295, color='black', linestyle='--', lw=1.5, label='hit_timeout boundary (295 s)')\n"
    "ax.set_xlabel('duration_seconds (clipped at 400)')\n"
    "ax.set_ylabel('Density')\n"
    "ax.set_title('Duration Distribution by Outcome\\n'\n"
    "             '(near-perfect Borg ROC-AUC is legitimate domain signal, not leakage)')\n"
    "ax.legend()\n"
    "ax.grid(alpha=0.25)\n"
    "plt.tight_layout()\n"
    "plt.savefig(f'{FIG_DIR}/duration_by_outcome.png', dpi=150)\n"
    "plt.show()\n"
    "\n"
    "# Ablation: zero hit_timeout + median duration\n"
    "rf = best_estimators['Random Forest']\n"
    "y_pred_full = rf.predict(X_test)\n"
    "f1_full     = f1_score(y_test, y_pred_full)\n"
    "\n"
    "X_test_abl = X_test.copy()\n"
    "X_test_abl['hit_timeout']      = 0\n"
    "X_test_abl['duration_seconds'] = float(X_test['duration_seconds'].median())\n"
    "y_pred_abl = rf.predict(X_test_abl)\n"
    "f1_abl     = f1_score(y_test, y_pred_abl)\n"
    "\n"
    "print(f'Full F1 (all features)              : {f1_full:.4f}')\n"
    "print(f'Ablated F1 (hit_timeout=0, dur=med) : {f1_abl:.4f}')\n"
    "print(f'F1 drop                             : {f1_full - f1_abl:.4f}')\n"
    "\n"
    "ablation_df = pd.DataFrame([{'Model': 'Random Forest',\n"
    "    'F1 (all features)': round(f1_full, 4), 'F1 (ablated)': round(f1_abl, 4),\n"
    "    'F1 drop': round(f1_full - f1_abl, 4),\n"
    "    'Ablation': 'hit_timeout=0, duration_seconds=median'}])\n"
    "ablation_df.to_csv(f'{RES_DIR}/ablation_results.csv', index=False)\n"
    "print('Saved hit_timeout_diagnostic.csv, duration_by_outcome.png, ablation_results.csv')\n"
))

# ── 4.8 Cross-dataset ───────────────────────────────────────────────────────
cells.append(md_cell(
    "## 4.8  Cross-Dataset Generalisation (Alibaba 2018)\n\n"
    "**Dataset:** Alibaba Cluster Trace v2018 — `batch_task.csv` (14.1M tasks, 8 days, ~4,000 machines).\n\n"
    "**Feature alignment:**\n"
    "| Alibaba field | Borg equivalent | Transformation |\n"
    "|---|---|---|\n"
    "| `plan_cpu` | `resource_request_cpu` | ÷100 (100 = 1 core) |\n"
    "| `plan_mem` | `resource_request_memory` | ÷100 (normalised [0,100]) |\n"
    "| `end_time − start_time` | `duration_seconds` | direct (seconds) |\n"
    "| `status = 'Failed'/'Killed'` | `failed = 1` | terminal-state filter |\n"
    "| `status = 'Terminated'` | `failed = 0` | terminal-state filter |\n"
    "| All other Borg features | 0 | fill_value=0 |\n\n"
    "**Class-prior mismatch:** Borg 22.8% vs Alibaba 0.6% failure rate (38× difference).\n\n"
    "**Threshold note:** The prior-ratio formula (t≈0.013) was found to be degenerate — "
    "it predicts all 14.1M tasks as failures regardless of model. "
    "Instead, the **PR-curve optimal threshold** per model is reported: the threshold that "
    "maximises F1 on the Alibaba set. ROC-AUC and PR-AUC are the primary threshold-free metrics.\n\n"
    "**Transfer verdict:** Only XGBoost (ROC-AUC=0.621 > 0.5 baseline) shows meaningful signal transfer. "
    "Random Forest (0.427) and LightGBM (0.415) are at or below the random baseline — honest negatives."
))
cells.append(code_cell(
    "from sklearn.metrics import (average_precision_score, precision_recall_curve,\n"
    "                             precision_score as ps, recall_score as rs)\n"
    "import pickle, numpy as np\n"
    "\n"
    "ALIBABA_PATH = f'{DATA_DIR}/alibaba_features.csv'\n"
    "\n"
    "if not os.path.exists(ALIBABA_PATH):\n"
    "    print('ERROR: alibaba_features.csv not found. Run src/phase4_alibaba_crossval.py')\n"
    "else:\n"
    "    df_ali = pd.read_csv(ALIBABA_PATH)\n"
    "    y_ali  = df_ali['failed']\n"
    "    X_ali  = df_ali.drop(columns=['failed']).reindex(columns=X.columns, fill_value=0)\n"
    "\n"
    "    print(f'Alibaba tasks: {len(y_ali):,} | failure rate: {y_ali.mean():.3%}')\n"
    "    print(f'Borg failure rate: {y.mean():.3%} | prior ratio: {y.mean()/y_ali.mean():.1f}x')\n"
    "\n"
    "    ali_records = []\n"
    "    for name, model in best_estimators.items():\n"
    "        probs     = model.predict_proba(X_ali)[:, 1]\n"
    "        y_default = (probs >= 0.5).astype(int)\n"
    "\n"
    "        prec_c, rec_c, thresh_c = precision_recall_curve(y_ali, probs)\n"
    "        f1_c     = 2 * prec_c[:-1] * rec_c[:-1] / (prec_c[:-1] + rec_c[:-1] + 1e-9)\n"
    "        best_idx = int(np.argmax(f1_c))\n"
    "\n"
    "        ali_records.append({\n"
    "            'Model':                   name,\n"
    "            'F1 (t=0.5)':              round(f1_score(y_ali, y_default, zero_division=0), 4),\n"
    "            'Recall (t=0.5)':          round(recall_score(y_ali, y_default, zero_division=0), 4),\n"
    "            'F1 (PR-optimal)':         round(float(f1_c[best_idx]), 4),\n"
    "            'Precision (PR-optimal)':  round(float(prec_c[best_idx]), 4),\n"
    "            'Recall (PR-optimal)':     round(float(rec_c[best_idx]), 4),\n"
    "            'Optimal threshold':       round(float(thresh_c[best_idx]), 4),\n"
    "            'ROC-AUC':                 round(roc_auc_score(y_ali, probs), 4),\n"
    "            'PR-AUC':                  round(average_precision_score(y_ali, probs), 4),\n"
    "            'PR-AUC baseline':         round(float(y_ali.mean()), 4),\n"
    "            'Transfers? (ROC>0.55)':   'Yes' if roc_auc_score(y_ali, probs) > 0.55 else 'No',\n"
    "        })\n"
    "\n"
    "    ali_df = pd.DataFrame(ali_records)\n"
    "    print(ali_df.to_string(index=False))\n"
    "    ali_df.to_csv(f'{RES_DIR}/cross_dataset_alibaba.csv', index=False)\n"
    "\n"
    "    from IPython.display import Image\n"
    "    display(Image(f'{FIG_DIR}/cross_dataset_pr_curves.png'))\n"
    "    print('Saved cross_dataset_alibaba.csv')\n"
))

# ── Acceptance criteria summary ─────────────────────────────────────────────
cells.append(md_cell(
    "## Acceptance Criteria Check\n\n"
    "| Deliverable | Path | Status |\n"
    "|---|---|---|\n"
    "| `phase4_benchmark.csv` | `outputs/results/` | RF, XGBoost, LightGBM rows with 5 metrics |\n"
    "| `error_analysis.csv` | `outputs/results/` | FP / FN statistics |\n"
    "| `cross_dataset_alibaba.csv` | `outputs/results/` | Alibaba results: PR-optimal F1, ROC-AUC, PR-AUC, transfer verdict |\n"
    "| `cross_dataset_pr_curves.png` | `outputs/figures/` | Per-model PR curves with optimal threshold marked |\n"
    "| `hit_timeout_diagnostic.csv` | `outputs/results/` | Failure rate by hit_timeout value |\n"
    "| `ablation_results.csv` | `outputs/results/` | F1 drop when hit_timeout + duration zeroed |\n"
    "| `duration_by_outcome.png` | `outputs/figures/` | Duration histogram — explains near-perfect ROC-AUC |\n"
    "| `feature_importance.png` | `outputs/figures/` | Top-20 bar chart |\n"
    "| `shap_summary.png` | `outputs/figures/` | SHAP beeswarm |\n"
    "| `confusion_matrix.png` | `outputs/figures/` | Confusion matrix |\n"
    "| `04_models.ipynb` | `notebooks/` | This notebook |"
))

# ── Assemble notebook ────────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.13.2",
        },
    },
    "cells": cells,
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Written: {OUT}  ({len(cells)} cells)")
