import pandas as pd
import numpy as np
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def main():
    # 1.4 Validate Member 1 Outputs
    assert os.path.exists("data/borg_clean.csv"), "FAIL: borg_clean.csv not found."
    df = pd.read_csv("data/borg_clean.csv")

    required_columns = [
        "scheduling_class", "priority", "collection_type",
        "resource_request_cpu", "resource_request_memory",
        "average_usage_cpu", "average_usage_memory", "maximum_usage_memory",
        "assigned_memory", "page_cache_memory",
        "duration_seconds",
        "cpu_dist_mean", "cpu_dist_std", "cpu_dist_max", "cpu_dist_skew",
        "failed"
    ]
    missing = [col for col in required_columns if col not in df.columns]
    assert not missing, f"FAIL: Missing columns from Member 1: {missing}"

    assert df["failed"].isin([0, 1]).all(), "FAIL: Target column 'failed' is not binary."
    assert len(df) > 100_000, f"WARN: Row count ({len(df)}) seems low. Expected ~405,894."

    print(f"PASS: borg_clean.csv loaded successfully. Shape: {df.shape}")
    print(f"Failure rate: {df['failed'].mean():.2%}")

    # 2.3 Engineer New Features
    df["hit_timeout"] = (df["duration_seconds"] >= 295).astype(int)
    df["cpu_utilization_ratio"] = df["average_usage_cpu"] / (df["resource_request_cpu"] + 1e-9)
    df["memory_pressure"] = df["average_usage_memory"] / (df["assigned_memory"] + 1e-9)

    print("\nEngineered features added: hit_timeout, cpu_utilization_ratio, memory_pressure")
    print(df[["hit_timeout", "cpu_utilization_ratio", "memory_pressure"]].describe())

    # 2.4 Drop Columns That Must Be Excluded
    drop_cols = ["event", "user", "collection_name", "collection_logical_name"]
    id_like = [col for col in df.columns if col.endswith("_id") or col == "instance_index"]
    drop_cols += id_like
    drop_cols = [col for col in drop_cols if col in df.columns]
    df.drop(columns=drop_cols, inplace=True)

    print(f"\nDropped {len(drop_cols)} columns: {drop_cols}")
    print(f"Remaining columns: {df.shape[1]}")

    # 2.5 Handle Missing Values
    y = df["failed"].copy()
    X = df.drop(columns=["failed"])
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    imputer = SimpleImputer(strategy="median")
    X[numeric_cols] = imputer.fit_transform(X[numeric_cols])

    print(f"\nImputed missing values in {len(numeric_cols)} numeric columns.")
    print(f"Remaining nulls after imputation: {X.isnull().sum().sum()}")

    # 2.6 Encode Categorical Features
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
        print(f"\nOne-hot encoded {len(cat_cols)} categorical columns: {cat_cols}")
    else:
        print("\nNo categorical columns requiring encoding.")

    # 2.7 Normalise with StandardScaler
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    print("\nApplied StandardScaler to all features.")
    print(X_scaled.describe().loc[["mean", "std"]].round(3))

    # 2.8 Apply SMOTE to Address Class Imbalance
    print(f"\nClass distribution before SMOTE:\n{y.value_counts()}")

    # Subsample to 200,000 rows to avoid SMOTE memory error (as per SOP Error Handling)
    if len(X_scaled) > 200000:
        print("Subsampling to 200,000 rows before SMOTE...")
        np.random.seed(42)
        idx = np.random.choice(X_scaled.index, size=200000, replace=False)
        X_scaled = X_scaled.loc[idx]
        y = y.loc[idx]
        print(f"Class distribution after subsampling:\n{y.value_counts()}")

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    print(f"\nClass distribution after SMOTE:\n{pd.Series(y_resampled).value_counts()}")
    print(f"Final feature matrix shape: {X_resampled.shape}")

    # 2.9 Save the Final Feature Matrix
    features_clean = pd.DataFrame(X_resampled, columns=X_scaled.columns)
    features_clean["failed"] = y_resampled.values

    # create data directory if not exists, though it should exist based on borg_clean.csv check
    os.makedirs("data", exist_ok=True)
    features_clean.to_csv("data/features_clean.csv", index=False)
    print(f"\nSaved features_clean.csv — Shape: {features_clean.shape}")

    # 2.10 Validate Phase 2 Output
    out = pd.read_csv("data/features_clean.csv")

    assert "failed" in out.columns, "FAIL: Target column missing from features_clean.csv"
    assert out.isnull().sum().sum() == 0, "FAIL: Nulls remain in features_clean.csv"
    assert out["failed"].value_counts().min() > 0, "FAIL: Class imbalance not resolved"

    balance = out["failed"].value_counts(normalize=True)
    assert 0.4 <= balance[1] <= 0.6, f"WARN: SMOTE result looks unexpected: {balance.to_dict()}"

    print(f"\nPASS: features_clean.csv validated. Shape: {out.shape}")
    print(f"Class balance after SMOTE:\n{balance.round(3)}")

if __name__ == "__main__":
    main()
