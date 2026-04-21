"""
Phase 1 — Data Cleaning
Reads raw borg_traces_data.csv, parses struct/histogram columns,
derives duration_seconds and failed target, drops high-missing cols,
saves data/borg_clean.csv.
"""
import sys, os
sys.path.insert(0, "D:/ML/pylibs")   # imblearn / extra libs on D drive

import ast, re, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH  = os.path.join(ROOT, "data", "borg_traces_data.csv")
OUT_PATH  = os.path.join(ROOT, "data", "borg_clean.csv")

print("=" * 60)
print("Phase 1 — Data Cleaning")
print("=" * 60)

df = pd.read_csv(RAW_PATH, low_memory=False)
print(f"Raw shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

# ── Parse struct columns ────────────────────────────────────────────────────
def parse_struct(value, key):
    if pd.isna(value) or str(value).strip() in ("", "None", "nan"):
        return np.nan
    try:
        parsed = ast.literal_eval(str(value))
        result = parsed.get(key, np.nan)
        return np.nan if result is None else float(result)
    except (ValueError, SyntaxError):
        return np.nan

for col in ["resource_request", "average_usage", "maximum_usage", "random_sample_usage"]:
    if col not in df.columns:
        continue
    print(f"  Parsing {col}...")
    df[f"{col}_cpu"]    = df[col].apply(lambda x: parse_struct(x, "cpus"))
    df[f"{col}_memory"] = df[col].apply(lambda x: parse_struct(x, "memory"))
    df.drop(columns=[col], inplace=True)

# ── Parse CPU histogram ─────────────────────────────────────────────────────
def parse_array_string(value):
    if pd.isna(value):
        return []
    try:
        cleaned = re.sub(r'\s+', ' ', str(value).strip())
        cleaned = cleaned.replace('[', '').replace(']', '').strip()
        return [float(x) for x in cleaned.split() if x]
    except Exception:
        return []

if "cpu_usage_distribution" in df.columns:
    print("  Parsing cpu_usage_distribution...")
    dist = df["cpu_usage_distribution"].apply(parse_array_string)
    df["cpu_dist_mean"] = dist.apply(lambda x: float(np.mean(x))          if len(x) > 0 else np.nan)
    df["cpu_dist_std"]  = dist.apply(lambda x: float(np.std(x))           if len(x) > 0 else np.nan)
    df["cpu_dist_max"]  = dist.apply(lambda x: float(np.max(x))           if len(x) > 0 else np.nan)
    df["cpu_dist_skew"] = dist.apply(lambda x: float(pd.Series(x).skew()) if len(x) > 2 else np.nan)
    df.drop(columns=["cpu_usage_distribution", "tail_cpu_usage_distribution"],
            inplace=True, errors="ignore")

# ── Target + duration ────────────────────────────────────────────────────────
assert "failed" in df.columns, "ERROR: 'failed' column missing"
df["duration_seconds"] = (df["end_time"] - df["start_time"]) / 1_000_000
n_before = len(df)
df = df[df["duration_seconds"] >= 0].copy()
print(f"  Removed {n_before - len(df):,} rows with negative duration.")

# ── Drop high-missing columns (>50%) ────────────────────────────────────────
missing_rate = df.isnull().mean()
cols_to_drop = missing_rate[missing_rate > 0.50].index.tolist()
print(f"  Dropping {len(cols_to_drop)} columns with >50% missing: {cols_to_drop}")
df.drop(columns=cols_to_drop, inplace=True)

print(f"\nClean shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"Target: success={( df['failed']==0).sum():,}  failure={(df['failed']==1).sum():,}")
for col in ["cpu_dist_mean", "cpu_dist_std", "cpu_dist_max", "cpu_dist_skew"]:
    assert col in df.columns, f"Missing: {col}"

df.to_csv(OUT_PATH, index=False)
print(f"Saved → {OUT_PATH}")
