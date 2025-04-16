# juri
Accidents severity prediction 

# GPU-Accelerated US Accident Severity Prediction

## Overview

This project predicts the severity level (Classes 1-4) of US traffic accidents using the Kaggle US Accidents dataset (2016-2023). It leverages the RAPIDS ecosystem (`cudf`, `cuml`) and GPU-accelerated `xgboost` for efficient data processing and machine learning on a large dataset, demonstrating the benefits of GPU acceleration.

## Objective

*   Predict accident severity based on various features.
*   Implement an end-to-end data science workflow using GPU acceleration via RAPIDS.
*   Train and compare three different classification models (Logistic Regression, Random Forest, XGBoost) on the GPU.

## Dataset

*   **Source:** [Kaggle US Accidents (2016-2023)](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)
*   **Original Size:** ~7.7 million records, 46 features.
*   **File Used:** `US_Accidents_March23.csv`
*   **Note:** Significant data reduction occurred during preprocessing due to dropping rows with nulls in key columns ('Zipcode', 'Wind_Chill(F)', 'Precipitation(in)'), resulting in ~2.6M rows for training/testing.

## Technologies Used

*   **RAPIDS:**
    *   `cudf`: GPU DataFrame manipulation.
    *   `cuml`: GPU-accelerated machine learning (Logistic Regression, Random Forest, StandardScaler, train_test_split).
*   **XGBoost:** GPU-accelerated training (`tree_method='gpu_hist'`).
*   **Pandas/NumPy:** Data handling (minor use, primarily for compatibility where needed).
*   **Scikit-learn:** Metrics (accuracy_score, classification_report - used on CPU after data conversion).
*   **Joblib:** Saving/loading model and scaler.
*   **Matplotlib/Seaborn:** Visualization.
*   **Google Colab:** Development environment (with GPU runtime - NVIDIA A100 used).

## Workflow

1.  **Data Loading:** Loaded CSV into a `cudf` DataFrame.
2.  **Preprocessing (cuDF):**
    *   Dropped irrelevant columns.
    *   Sampled data (50% used in this run) to manage resources if needed (*adjust as necessary*).
    *   Imputed missing numericals (median) and categoricals (mode).
    *   Dropped rows with remaining critical nulls.
    *   Converted boolean columns to integers.
3.  **Feature Engineering (cuDF):**
    *   One-Hot Encoded low-cardinality categoricals (`cudf.get_dummies`).
    *   Frequency Encoded high-cardinality categoricals (using `value_counts` and `map`).
4.  **Data Splitting (cuML):** Split into 70% train / 30% test.
5.  **Target Preparation:** Shifted 'Severity' labels from 1-4 to 0-3.
6.  **Scaling (cuML):** Applied `StandardScaler` to all features.
7.  **Model Training (GPU):**
    *   Trained `cuml.linear_model.LogisticRegression`.
    *   Trained `cuml.ensemble.RandomForestClassifier`.
    *   Trained `xgboost.XGBClassifier` with `tree_method='gpu_hist'`.
8.  **Evaluation:** Calculated accuracy using `cuml.metrics.accuracy_score`. *(Classification reports using sklearn required GPU->CPU data transfer)*.
9.  **Artifact Saving:** Saved the best model (XGBoost) and the scaler using `joblib`.

## Results Summary

| Model               | Accuracy (Test Set) | GPU Train Time (s) | GPU Predict Time (s) |
| :------------------ | :------------------ | :----------------- | :------------------- |
| **XGBoost**         | **~90.8%**          | ~7.9s              | ~0.23s               |
| Random Forest       | ~87.6%              | ~54.4s             | ~0.39s               |
| Logistic Regression | ~87.0%              | ~1.3s              | ~0.02s               |

*(Prediction times are for ~788k test samples)*

**Observations:**
*   XGBoost achieved the highest accuracy.
*   GPU acceleration provided significant speedups, especially for XGBoost and Random Forest training compared to potential CPU times.
*   Logistic Regression training and prediction were extremely fast on the GPU.

## How to Use/Reproduce

1.  **Environment:** Set up a Python environment with RAPIDS installed (requires compatible NVIDIA GPU and drivers). See [RAPIDS installation guide](https://rapids.ai/start.html).
2.  **Data:** Download the `US_Accidents_March23.csv` file from the Kaggle link above and place it in the `/content/` directory (or update path in the notebook).
3.  **Notebook:** Run the `JURI-3.ipynb` notebook cell by cell in a GPU-enabled environment (like Google Colab, Kaggle Kernels, or a local RAPIDS setup).
4.  **Prediction:**
    *   The trained XGBoost model is saved as `xgb_model.joblib`.
    *   The data scaler is saved as `scaler.joblib`.
    *   These can be loaded using `joblib.load()` for making predictions on new data (ensure new data undergoes the *exact same* preprocessing and scaling steps). *(See `make_prediction` function draft in the notebook for an example)*.

## Future Work

*   Explore more sophisticated imputation techniques for missing values.
*   Perform hyperparameter tuning (e.g., using `GridSearchCV`) for potentially better model performance (though this will increase computation time).
*   Evaluate models using additional metrics sensitive to class imbalance (Precision, Recall, F1-score per class, Confusion Matrix).
*   Investigate feature importance to understand key drivers of accident severity.
