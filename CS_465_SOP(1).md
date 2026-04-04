# Member 1 — Standard Operating Procedure
## CS465 Machine Learning Project: Google Borg Job Failure Prediction
**Role:** Data Pipeline · Loading · Parsing · Cleaning · EDA  
**Deadline:** Week 7 deliverable  
**Hands-off to:** Member 2 (Feature Engineering) with a clean, saved DataFrame

---

## Your Mission

You are the foundation of the entire project. Every other member depends on
what you produce. Your job is to take the raw 328 MB CSV file and deliver:

1. A **clean, parsed DataFrame** saved as `data/borg_clean.csv`
2. A **Jupyter notebook** (`notebooks/01_eda.ipynb`) with 10+ publication-quality figures
3. A **written summary** (`outputs/eda_summary.md`) of your key findings for Member 5 to use in the IEEE paper

If your data is messy, every downstream model will be wrong. Do this carefully.

---

## Table of Contents

1. [Setup](#step-1-environment-setup)
2. [Project Structure](#step-2-create-project-structure)
3. [Download the Dataset](#step-3-download-the-dataset)
4. [Load & Inspect](#step-4-load-and-inspect-the-raw-data)
5. [Parse Struct Columns](#step-5-parse-struct-columns)
6. [Parse CPU Histograms](#step-6-parse-cpu-histogram-columns)
7. [Clean the Data](#step-7-clean-the-data)
8. [EDA — Required Figures](#step-8-eda-required-figures)
9. [Save Outputs](#step-9-save-outputs)
10. [Handoff Checklist](#step-10-handoff-checklist)

---

## Step 1: Environment Setup

Open your terminal and run the following commands exactly.

```bash
# Create and activate a virtual environment
python -m venv cs465_env
source cs465_env/bin/activate        # Mac/Linux
# cs465_env\Scripts\activate         # Windows

# Install all required libraries
pip install pandas numpy matplotlib seaborn scikit-learn ast-tools jupyter
```

Verify everything installed correctly:

```bash
python -c "import pandas, numpy, matplotlib, seaborn, sklearn; print('All good.')"
```

If you see `All good.` you are ready to proceed.

---

## Step 2: Create Project Structure

Run this once to create the shared folder structure the whole team will use:

```bash
mkdir -p cs465-project/data
mkdir -p cs465-project/notebooks
mkdir -p cs465-project/outputs/figures
mkdir -p cs465-project/outputs/results
mkdir -p cs465-project/src
cd cs465-project
```

Your folder should look like this when done:

```
cs465-project/
├── data/
│   └── borg_traces_data.csv        ← place the downloaded file here
├── notebooks/
│   └── 01_eda.ipynb                ← your main notebook
├── outputs/
│   ├── figures/                    ← all EDA plots saved here
│   └── results/                    ← clean CSV saved here
├── src/
│   └── (empty for now)
└── README.md
```

Create the README:

```bash
echo "# CS465 — Google Borg Job Failure Prediction" > README.md
echo "Team of 5 | Prince Sultan University | Prof. Wadii Boulila" >> README.md
```

---

## Step 3: Download the Dataset

1. Go to [https://www.kaggle.com](https://www.kaggle.com)
2. Search for: **borg traces data**
3. Download `borg_traces_data.csv` (328 MB)
4. Place it in `cs465-project/data/borg_traces_data.csv`

> **Important:** Do NOT rename the file. The rest of the team's code will
> reference it by this exact name.

---

## Step 4: Load and Inspect the Raw Data

Open Jupyter and create `notebooks/01_eda.ipynb`. Start with this cell:

```python
import pandas as pd
import numpy as np
import ast
import re
import os
import warnings
warnings.filterwarnings('ignore')

# ── Load raw CSV ───────────────────────────────────────────────────────────────
RAW_PATH   = "../data/borg_traces_data.csv"
CLEAN_PATH = "../outputs/results/borg_clean.csv"
FIG_PATH   = "../outputs/figures/"

df = pd.read_csv(RAW_PATH, low_memory=False)

# ── Basic inspection ───────────────────────────────────────────────────────────
print("=" * 60)
print(f"Shape:        {df.shape[0]:,} rows  x  {df.shape[1]} columns")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
print("=" * 60)

print("\nColumn names:")
for col in df.columns:
    print(f"  {col}")

print("\nData types:")
print(df.dtypes)

print("\nFirst 3 rows:")
df.head(3)
```

**What to check:**
- Confirm you have 34 columns
- Note which columns look like dict strings (e.g. `{'cpus': 0.02, 'memory': 0.01}`)
- Note which columns look like array strings (e.g. `[0.003 0.004 0.005 ...]`)

---

## Step 5: Parse Struct Columns

Four columns store CPU and memory values encoded as Python dict strings.
You must parse each into two separate numeric columns.

```python
# ── Helper function ─────────────────────────────────────────────────────────
def parse_struct(value, key):
    """
    Extracts a numeric value from a dict-string column.

    Why this is needed: pandas reads {'cpus': 0.02, 'memory': 0.01} as a
    plain string. ast.literal_eval converts it back to a real Python dict
    so we can index it with the key we want.

    Args:
        value : the raw string from the DataFrame cell
        key   : 'cpus' or 'memory'

    Returns:
        float if parsing succeeds, np.nan if it fails
    """
    if pd.isna(value) or str(value).strip() in ("", "None", "nan"):
        return np.nan
    try:
        parsed = ast.literal_eval(str(value))
        result = parsed.get(key, np.nan)
        # Some memory values are stored as None inside the dict
        return np.nan if result is None else float(result)
    except (ValueError, SyntaxError):
        return np.nan


# ── Parse all four struct columns ───────────────────────────────────────────
struct_cols = [
    "resource_request",
    "average_usage",
    "maximum_usage",
    "random_sample_usage",
]

for col in struct_cols:
    print(f"Parsing {col}...")
    df[f"{col}_cpu"]    = df[col].apply(lambda x: parse_struct(x, "cpus"))
    df[f"{col}_memory"] = df[col].apply(lambda x: parse_struct(x, "memory"))
    # Drop the original string column — it is no longer needed
    df.drop(columns=[col], inplace=True)

print("\nDone. New numeric columns created:")
new_cols = [c for c in df.columns if any(s in c for s in struct_cols)]
for c in new_cols:
    print(f"  {c}  —  non-null: {df[c].notna().sum():,}")
```

**Expected output:** 8 new columns named like `resource_request_cpu`,
`resource_request_memory`, etc. Each should have tens of thousands of
non-null values.

---

## Step 6: Parse CPU Histogram Columns

The `cpu_usage_distribution` column stores an 11-point array per job.
Extract 4 summary statistics from it — these become novel engineered features.

```python
# ── Helper function ─────────────────────────────────────────────────────────
def parse_array_string(value):
    """
    Converts a numpy-array string into a Python list of floats.

    Example input:  "[0.003 0.004 0.005 0.006 0.007]"
    Example output: [0.003, 0.004, 0.005, 0.006, 0.007]

    The tricky part: numpy prints arrays with spaces, not commas.
    re.sub collapses all whitespace into single spaces before splitting.
    """
    if pd.isna(value):
        return []
    try:
        cleaned = re.sub(r'\s+', ' ', str(value).strip())
        cleaned = cleaned.replace('[', '').replace(']', '').strip()
        return [float(x) for x in cleaned.split() if x]
    except Exception:
        return []


# ── Extract summary statistics from the distribution ────────────────────────
print("Parsing cpu_usage_distribution (this may take a minute)...")
dist_series = df["cpu_usage_distribution"].apply(parse_array_string)

# Mean: average CPU usage across the distribution
df["cpu_dist_mean"] = dist_series.apply(
    lambda x: float(np.mean(x)) if len(x) > 0 else np.nan
)

# Std: variability of CPU usage — high std = bursty job
df["cpu_dist_std"] = dist_series.apply(
    lambda x: float(np.std(x)) if len(x) > 0 else np.nan
)

# Max: peak CPU sample in the distribution
df["cpu_dist_max"] = dist_series.apply(
    lambda x: float(np.max(x)) if len(x) > 0 else np.nan
)

# Skewness: are most samples low with occasional spikes? (positive skew)
df["cpu_dist_skew"] = dist_series.apply(
    lambda x: float(pd.Series(x).skew()) if len(x) > 2 else np.nan
)

# Drop the original array string columns — not needed downstream
df.drop(columns=["cpu_usage_distribution", "tail_cpu_usage_distribution"],
        inplace=True, errors="ignore")

print("Done. CPU histogram features created:")
for col in ["cpu_dist_mean", "cpu_dist_std", "cpu_dist_max", "cpu_dist_skew"]:
    print(f"  {col}  —  non-null: {df[col].notna().sum():,}")
```

---

## Step 7: Clean the Data

```python
# ── 7.1  Create the target variable ─────────────────────────────────────────
# 'failed' should already exist as a 0/1 column — verify it
assert "failed" in df.columns, "ERROR: 'failed' column not found!"
assert df["failed"].isin([0, 1]).all(), "ERROR: 'failed' has values other than 0/1!"

print(f"Target variable 'failed':")
print(f"  Success (0): {(df['failed']==0).sum():,}  ({(df['failed']==0).mean():.1%})")
print(f"  Failure (1): {(df['failed']==1).sum():,}  ({(df['failed']==1).mean():.1%})")


# ── 7.2  Create duration feature ─────────────────────────────────────────────
# Borg timestamps are in microseconds — convert to seconds
# Duration = how long the job ran before succeeding or failing
df["duration_seconds"] = (df["end_time"] - df["start_time"]) / 1_000_000

# Remove impossible durations (negative = data error)
n_before = len(df)
df = df[df["duration_seconds"] >= 0].copy()
print(f"\nRemoved {n_before - len(df):,} rows with negative duration.")


# ── 7.3  Drop columns with >50% missing values ───────────────────────────────
missing_rate = df.isnull().mean()
cols_to_drop = missing_rate[missing_rate > 0.50].index.tolist()

print(f"\nDropping {len(cols_to_drop)} columns with >50% missing values:")
for c in cols_to_drop:
    print(f"  {c}  ({missing_rate[c]:.1%} missing)")

df.drop(columns=cols_to_drop, inplace=True)


# ── 7.4  Final shape report ──────────────────────────────────────────────────
print(f"\nFinal clean dataset shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"Remaining missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
```

---

## Step 8: EDA — Required Figures

Run each block below. Each one saves a figure to `outputs/figures/`.
You need **all 10 figures** for the IEEE paper's Dataset Exploration section.

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Shared plot style
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
BLUE = "#2471A3"
RED  = "#E74C3C"


# ── Figure 1: Class Distribution ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

counts = df["failed"].value_counts().sort_index()
labels = ["Success (0)", "Failure (1)"]
axes[0].bar(labels, counts.values, color=[BLUE, RED], width=0.5, edgecolor="white")
axes[0].set_title("Class Distribution")
axes[0].set_ylabel("Number of Jobs")
for i, v in enumerate(counts.values):
    axes[0].text(i, v * 1.02, f"{v:,}\n({v/len(df):.1%})",
                 ha="center", fontsize=9)

axes[1].pie(counts.values, labels=["Success", "Failure"],
            colors=[BLUE, RED], autopct="%1.1f%%",
            startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 1.5})
axes[1].set_title("Class Balance")

plt.suptitle("Figure 1 — Target Variable Distribution", fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig(FIG_PATH + "fig01_class_distribution.png",
            bbox_inches="tight", dpi=150)
plt.show()
print("Figure 1 saved.")


# ── Figure 2: Missing Values ─────────────────────────────────────────────────
missing = df.isnull().mean().sort_values(ascending=True)
missing = missing[missing > 0]

fig, ax = plt.subplots(figsize=(10, max(4, len(missing) * 0.35)))
bars = ax.barh(missing.index, missing.values * 100, color=BLUE, alpha=0.8)
ax.axvline(x=30, color=RED, linestyle="--", linewidth=1, label="30% threshold")
ax.set_xlabel("Missing Values (%)")
ax.set_title("Figure 2 — Missing Value Analysis by Column")
ax.legend()
for bar, val in zip(bars, missing.values):
    ax.text(val * 100 + 0.5, bar.get_y() + bar.get_height()/2,
            f"{val:.1%}", va="center", fontsize=8)
plt.tight_layout()
plt.savefig(FIG_PATH + "fig02_missing_values.png",
            bbox_inches="tight", dpi=150)
plt.show()
print("Figure 2 saved.")


# ── Figure 3: CPU Request Distribution by Outcome ───────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
df[df["failed"]==0]["resource_request_cpu"].dropna().hist(
    ax=ax, bins=60, alpha=0.6, color=BLUE, label="Success", density=True)
df[df["failed"]==1]["resource_request_cpu"].dropna().hist(
    ax=ax, bins=60, alpha=0.6, color=RED, label="Failure", density=True)
ax.set_xlabel("Requested CPU")
ax.set_ylabel("Density")
ax.set_title("Figure 3 — CPU Request Distribution: Success vs Failure")
ax.legend()
ax.set_yscale("log")
plt.tight_layout()
plt.savefig(FIG_PATH + "fig03_cpu_request_dist.png",
            bbox_inches="tight", dpi=150)
plt.show()
print("Figure 3 saved.")


# ── Figure 4: Memory Request Distribution by Outcome ────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
df[df["failed"]==0]["resource_request_memory"].dropna().hist(
    ax=ax, bins=60, alpha=0.6, color=BLUE, label="Success", density=True)
df[df["failed"]==1]["resource_request_memory"].dropna().hist(
    ax=ax, bins=60, alpha=0.6, color=RED, label="Failure", density=True)
ax.set_xlabel("Requested Memory")
ax.set_ylabel("Density")
ax.set_title("Figure 4 — Memory Request Distribution: Success vs Failure")
ax.legend()
ax.set_yscale("log")
plt.tight_layout()
plt.savefig(FIG_PATH + "fig04_memory_request_dist.png",
            bbox_inches="tight", dpi=150)
plt.show()
print("Figure 4 saved.")


# ── Figure 5: Failure Rate by Scheduling Class ──────────────────────────────
fail_by_sched = (df.groupby("scheduling_class")["failed"]
                   .agg(["mean", "count"])
                   .reset_index())
fail_by_sched.columns = ["scheduling_class", "failure_rate", "count"]
fail_by_sched = fail_by_sched.sort_values("failure_rate", ascending=False)

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(fail_by_sched["scheduling_class"].astype(str),
              fail_by_sched["failure_rate"] * 100,
              color=BLUE, width=0.5)
ax.set_xlabel("Scheduling Class")
ax.set_ylabel("Failure Rate (%)")
ax.set_title("Figure 5 — Failure Rate by Scheduling Class")
for bar, (_, row) in zip(bars, fail_by_sched.iterrows()):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.3,
            f"{row['failure_rate']:.1%}\n(n={row['count']:,})",
            ha="center", fontsize=8)
plt.tight_layout()
plt.savefig(FIG_PATH + "fig05_failure_by_scheduling_class.png",
            bbox_inches="tight", dpi=150)
plt.show()
print("Figure 5 saved.")


# ── Figure 6: Failure Rate by Priority Bin ──────────────────────────────────
df["priority_bin"] = pd.cut(
    df["priority"],
    bins=5,
    labels=["Very Low", "Low", "Medium", "High", "Very High"]
)
fail_by_priority = (df.groupby("priority_bin", observed=True)["failed"]
                      .mean()
                      .reset_index())

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(fail_by_priority["priority_bin"].astype(str),
       fail_by_priority["failed"] * 100,
       color=RED, alpha=0.8, width=0.5)
ax.set_xlabel("Priority Bin")
ax.set_ylabel("Failure Rate (%)")
ax.set_title("Figure 6 — Failure Rate by Job Priority")
plt.tight_layout()
plt.savefig(FIG_PATH + "fig06_failure_by_priority.png",
            bbox_inches="tight", dpi=150)
plt.show()
print("Figure 6 saved.")


# ── Figure 7: Job Duration Distribution ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
df[df["failed"]==0]["duration_seconds"].clip(0, 3600).dropna().hist(
    ax=ax, bins=60, alpha=0.6, color=BLUE, label="Success", density=True)
df[df["failed"]==1]["duration_seconds"].clip(0, 3600).dropna().hist(
    ax=ax, bins=60, alpha=0.6, color=RED, label="Failure", density=True)
ax.set_xlabel("Job Duration (seconds, clipped at 1 hour)")
ax.set_ylabel("Density")
ax.set_title("Figure 7 — Job Duration Distribution: Success vs Failure")
ax.legend()
plt.tight_layout()
plt.savefig(FIG_PATH + "fig07_duration_distribution.png",
            bbox_inches="tight", dpi=150)
plt.show()
print("Figure 7 saved.")


# ── Figure 8: CPU Histogram Feature Distributions ───────────────────────────
hist_features = ["cpu_dist_mean", "cpu_dist_std", "cpu_dist_max", "cpu_dist_skew"]
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for ax, feat in zip(axes, hist_features):
    df[df["failed"]==0][feat].dropna().hist(
        ax=ax, bins=50, alpha=0.6, color=BLUE, label="Success", density=True)
    df[df["failed"]==1][feat].dropna().hist(
        ax=ax, bins=50, alpha=0.6, color=RED, label="Failure", density=True)
    ax.set_title(feat, fontsize=9)
    ax.set_yscale("log")
    ax.legend(fontsize=7)

plt.suptitle("Figure 8 — CPU Histogram Engineered Features: Success vs Failure",
             fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig(FIG_PATH + "fig08_cpu_histogram_features.png",
            bbox_inches="tight", dpi=150)
plt.show()
print("Figure 8 saved.")


# ── Figure 9: Correlation Heatmap ───────────────────────────────────────────
corr_cols = [
    "resource_request_cpu", "resource_request_memory",
    "average_usage_cpu", "average_usage_memory",
    "maximum_usage_cpu", "maximum_usage_memory",
    "duration_seconds", "priority",
    "cpu_dist_mean", "cpu_dist_std", "cpu_dist_skew",
    "failed"
]
# Only keep columns that actually exist after cleaning
corr_cols = [c for c in corr_cols if c in df.columns]
corr_matrix = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix, mask=mask, annot=True, fmt=".2f",
    cmap="coolwarm", center=0, linewidths=0.4,
    annot_kws={"size": 7}, ax=ax
)
ax.set_title("Figure 9 — Feature Correlation Matrix")
plt.tight_layout()
plt.savefig(FIG_PATH + "fig09_correlation_heatmap.png",
            bbox_inches="tight", dpi=150)
plt.show()
print("Figure 9 saved.")


# ── Figure 10: Average CPU Usage vs Average Memory Usage (scatter) ───────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (cpu_col, mem_col, title) in zip(axes, [
    ("resource_request_cpu", "resource_request_memory", "Resource Request"),
    ("average_usage_cpu",    "average_usage_memory",    "Actual Average Usage"),
]):
    sample = df.sample(min(5000, len(df)), random_state=42)
    ax.scatter(
        sample[sample["failed"]==0][cpu_col],
        sample[sample["failed"]==0][mem_col],
        alpha=0.3, s=5, color=BLUE, label="Success"
    )
    ax.scatter(
        sample[sample["failed"]==1][cpu_col],
        sample[sample["failed"]==1][mem_col],
        alpha=0.4, s=5, color=RED, label="Failure"
    )
    ax.set_xlabel(f"CPU ({title})")
    ax.set_ylabel(f"Memory ({title})")
    ax.set_title(f"Figure 10a/b — {title}: CPU vs Memory")
    ax.legend(markerscale=3, fontsize=8)

plt.suptitle("Figure 10 — CPU vs Memory: Success vs Failure (5000-sample)",
             fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig(FIG_PATH + "fig10_cpu_vs_memory_scatter.png",
            bbox_inches="tight", dpi=150)
plt.show()
print("Figure 10 saved.")

print("\nAll 10 figures saved to outputs/figures/")
```

---

## Step 9: Save Outputs

```python
# ── 9.1  Save the clean DataFrame ────────────────────────────────────────────
# This is what Members 2, 3, and 4 will load — do not skip this step
df.to_csv(CLEAN_PATH, index=False)
print(f"Clean dataset saved: {CLEAN_PATH}")
print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")


# ── 9.2  Print final summary for eda_summary.md ──────────────────────────────
print("\n" + "="*60)
print("EDA SUMMARY — paste this into outputs/eda_summary.md")
print("="*60)
print(f"Total jobs:          {len(df):,}")
print(f"Successful jobs:     {(df['failed']==0).sum():,}  ({(df['failed']==0).mean():.1%})")
print(f"Failed jobs:         {(df['failed']==1).sum():,}  ({(df['failed']==1).mean():.1%})")
print(f"Features retained:   {df.shape[1] - 1}")
print(f"Avg duration (succ): {df[df['failed']==0]['duration_seconds'].mean():.1f}s")
print(f"Avg duration (fail): {df[df['failed']==1]['duration_seconds'].mean():.1f}s")
print(f"Avg CPU request:     {df['resource_request_cpu'].mean():.5f}")
print(f"Avg memory request:  {df['resource_request_memory'].mean():.5f}")
print(f"Most failure-prone scheduling_class: "
      f"{df.groupby('scheduling_class')['failed'].mean().idxmax()}")
```

---

## Step 10: Handoff Checklist

Before you tell the team you are done, verify every item below:

```
[ ] data/borg_traces_data.csv        — raw file in place (do not modify)
[ ] outputs/results/borg_clean.csv   — clean file saved and opens correctly
[ ] notebooks/01_eda.ipynb           — runs top to bottom without errors
[ ] outputs/figures/fig01_*.png      — class distribution
[ ] outputs/figures/fig02_*.png      — missing values
[ ] outputs/figures/fig03_*.png      — CPU request distribution
[ ] outputs/figures/fig04_*.png      — memory request distribution
[ ] outputs/figures/fig05_*.png      — failure by scheduling class
[ ] outputs/figures/fig06_*.png      — failure by priority
[ ] outputs/figures/fig07_*.png      — duration distribution
[ ] outputs/figures/fig08_*.png      — CPU histogram features
[ ] outputs/figures/fig09_*.png      — correlation heatmap
[ ] outputs/figures/fig10_*.png      — CPU vs memory scatter
[ ] outputs/eda_summary.md           — key numbers written up for Member 5
```

Once all boxes are checked, message the group chat with:
- The shape of `borg_clean.csv`
- The failure rate
- The top 2-3 findings from EDA (e.g. which scheduling class fails most)

Member 2 cannot begin feature engineering until `borg_clean.csv` exists.

---

## Common Errors & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `FileNotFoundError` on CSV | Wrong path | Confirm file is in `data/` and path uses `../data/` |
| `ValueError` in `ast.literal_eval` | Non-standard dict string | The `parse_struct` function handles this — check you copied it exactly |
| All histogram values are `NaN` | Array string has newlines | The `re.sub` in `parse_array_string` handles this — check you copied it exactly |
| `KeyError: 'failed'` | Column named differently | Run `print(df.columns.tolist())` to check the actual column name |
| Figure not showing | Running as script not notebook | Use Jupyter, not a plain `.py` file |
| Memory error on 328 MB file | Not enough RAM | Add `nrows=100000` to `read_csv` temporarily for testing |

---

## Key Numbers to Report to the Team

After running Step 9, fill in these values and post them to the group:

| Metric | Value |
|--------|-------|
| Total rows after cleaning | _______ |
| Failure rate | _______ % |
| Number of features retained | _______ |
| Most failure-prone scheduling class | _______ |
| Average duration of failed jobs (seconds) | _______ |
| Average duration of successful jobs (seconds) | _______ |

---

*CS465 Machine Learning · Prince Sultan University · Prof. Wadii Boulila*  
*Member 1 SOP — Data Pipeline, Loading, Parsing, Cleaning, EDA*
