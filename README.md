# BORG — Job Failure Prediction in Google Borg Cluster Traces

A comparative machine learning study predicting job failures in 
Google's Borg cluster, enabling proactive resource management and 
reducing wasted CPU cycles from failed jobs.

Trained on 405,894 job instances from the Google Borg Cluster 
Traces 2019. Submitted as a peer-reviewed research paper.

## Results
| Model         | F1 Score | ROC-AUC | Test Set        |
|---------------|----------|---------|-----------------|
| Random Forest | 0.9985   | 1.000   | 81,179 jobs     |
| XGBoost       | 0.9971   | 0.999   | 81,179 jobs     |
| LightGBM      | 0.9968   | 0.999   | 81,179 jobs     |

- Novel contribution: 11-point CPU histogram features per job
- SHAP explainability analysis on top predictive features
- Zero-shot cross-dataset validation on 14M Alibaba 2018 tasks

## Project Structure
- `dashboard/` — Interactive Streamlit dashboard
- `data/` — Raw and processed datasets (ignored by git)
- `notebooks/` — EDA and modelling iterations
- `outputs/` — Figures, serialised models, CSV results
- `src/` — Preprocessing, feature engineering, ML pipelines

## Stack
Python · scikit-learn · XGBoost · LightGBM · SMOTE · SHAP · 
Pandas · NumPy · Streamlit

## How to Run the Dashboard
1. Navigate to the dashboard directory:
   `cd dashboard`
2. Install dependencies and launch:
   `install_and_run.bat`
   Or manually:
   `pip install -r requirements.txt`
   `streamlit run streamlitdash.py`

## Data Access
Dataset not tracked due to size. To download:
1. Install kagglehub: `pip install kagglehub`
2. Run: `python dataset.py`
3. Script downloads "derrickmwiti/google-2019-cluster-sample" 
   automatically
