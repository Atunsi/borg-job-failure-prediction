# Google Borg Job Failure Prediction — EDA Summary

## Project Metadata
- **Member:** 1 (Data Pipeline, Loading, Parsing, Cleaning, EDA)
- **Status:** Complete

## Final Dataset Statistics

| Metric | Value |
|--------|-------|
| Total rows after cleaning | 405,894 |
| Success Rate | 313,216 (77.2%) |
| Failure rate | 22.8 % |
| Number of features retained | 40 |
| Most failure-prone scheduling class | 3 |
| Average duration of failed jobs (seconds) | 245.6 s |
| Average duration of successful jobs (seconds) | 201.9 s |
| Average CPU request | 0.01534 |
| Average memory request | 0.00904 |

## Key Insights

- **Scaling and Structure:** We parsed array and struct columns directly converting them into individual, statistically usable attributes like `cpu_dist_mean`, achieving a 41-column final shape.
- **Duration Impact:** Failed instances actually experience historically longer runtimes spanning an average of 43 seconds longer. 
- **Variable Cleanout:** The `random_sample_usage_memory` column had 100% missing data and was correctly dropped to prune noise in Member 2's Feature Engineering stage.

This summary concludes the EDA stage. The dataset has been appropriately cleaned and written to `outputs/results/borg_clean.csv`. All 10 baseline figures were fully compiled inside `outputs/figures/`.
