## Feature Engineering & Preprocessing

### 3.1 Engineered Features

Three new features were derived from the cleaned dataset:

- **hit_timeout**: Binary flag (1/0) indicating whether a job's `duration_seconds`
  reached or exceeded the 295-second threshold. Motivated by the hypothesis that
  jobs approaching resource time limits are more likely to fail.
- **cpu_utilization_ratio**: Ratio of `average_usage_cpu` to `resource_request_cpu`,
  capturing how efficiently a job utilised its allocated CPU.
- **memory_pressure**: Ratio of `average_usage_memory` to `assigned_memory`,
  indicating how close a job was to exhausting its memory allocation.

### 3.2 Column Selection

The following columns were dropped prior to modelling:
- `event`: Post-hoc label that directly encodes the outcome — a data leakage source.
- `user`, `collection_name`, `collection_logical_name`: Hashed identifier or high-cardinality columns with no generalizable signal.
- High-cardinality ID columns (e.g. `job_id`, `collection_id`, `instance_index`, `machine_id`).

### 3.3 Imputation

Remaining missing values in numeric columns were filled using **median imputation**
via `sklearn.impute.SimpleImputer`. Median was chosen over mean due to the skewed
distributions observed in Member 1's EDA.

### 3.4 Normalisation

All features were standardised using `sklearn.preprocessing.StandardScaler`
(zero mean, unit variance) to ensure gradient-based and distance-based models
are not biased by feature scale.

### 3.5 Class Imbalance — SMOTE

The original dataset exhibited a 77/23 class split (success/failure).
**SMOTE** (Synthetic Minority Over-sampling Technique) was applied to the
training-equivalent feature matrix, producing a balanced 50/50 distribution
of 154,436 samples per class.

Final feature matrix: **308,872 rows × 147 features**,
saved to `data/features_clean.csv`.
