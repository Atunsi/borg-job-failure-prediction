# Google Borg Job Failure Prediction

## Overview
This repository contains the deliverables for the CS465 project on predicting job failures in the Google Borg cluster trace dataset. The project involves exploratory data analysis (EDA), feature engineering, model training, cross-dataset validation, and an interactive Streamlit dashboard.

## Project Structure
- `dashboard/`: Contains the interactive Streamlit dashboard files and requirements.
  - `streamlitdash.py`: Main dashboard application script.
  - `requirements.txt`: Python dependencies for the dashboard.
  - `install_and_run.bat`: A batch script to easily install dependencies and start the dashboard.
- `data/`: Directory for raw and processed datasets (ignored by git due to size).
- `notebooks/`: Jupyter notebooks detailing EDA and modeling iterations.
- `outputs/`: Output artifacts including figures, serialized models, and CSV results.
- `src/`: Source code for data preprocessing, feature extraction, and machine learning pipelines.

## How to Run the Dashboard

1. Navigate to the `dashboard` directory:
   ```bash
   cd dashboard
   ```
2. Run the provided batch script to install all dependencies and launch the dashboard:
   ```bash
   install_and_run.bat
   ```
   *Alternatively, you can manually install the packages using `pip install -r requirements.txt` and run the dashboard with `streamlit run streamlitdash.py`.*

## Data Requirements & Access
Due to size constraints, the dataset is not tracked in version control. To download the necessary Google 2019 Cluster sample data locally:

1. Ensure you have the `kagglehub` package installed (`pip install kagglehub`).
2. Run the provided dataset script from the project root:
   ```bash
   python dataset.py
   ```
3. The script will automatically download the "derrickmwiti/google-2019-cluster-sample" dataset and print out the path where it has been saved on your local machine.

*Note: You may need to copy the downloaded files into the `data/` directory or update the scripts to reference the printed path before running the pipeline.*

## Tools & Libraries Used
- **Data Manipulation:** pandas, numpy
- **Machine Learning:** scikit-learn, XGBoost, LightGBM
- **Visualization:** matplotlib, seaborn, streamlit
