# juri
Accidents severity prediction 

Project Report: GPU-Accelerated US Accident Severity Prediction
Objective:
To predict the severity level (1-4 classes) of US traffic accidents using the Kaggle US Accidents dataset (2016-2023) and demonstrate the efficiency of GPU acceleration using the RAPIDS ecosystem (cuDF, cuML, GPU-accelerated XGBoost).
Dataset:
Source: Kaggle US Accidents (2016-2023).
Initial Size: ~7.7 million records, 46 features.
Tooling: Loaded and processed using cudf on a GPU (NVIDIA A100).
Methodology & Workflow:
Data Loading: Efficiently loaded the CSV into a cudf DataFrame (~1.2s).
Preprocessing (GPU-accelerated with cudf):
Dropped irrelevant/high-missing columns (ID, End_Time, End_Lat/Lng, Description, etc.).
Dropped specific time columns ('Start_Time', 'Civil_Twilight', etc.).
Imputed low-percentage missing numericals (Temperature, Humidity, Pressure, Visibility, Wind_Speed) with the median (calculated from the training set).
Imputed low-percentage missing categoricals (Wind_Direction, Weather_Condition) with the mode (calculated from the training set).
Handled critical nulls: Dropped rows with remaining nulls in 'Zipcode', 'Wind_Chill(F)', and 'Precipitation(in)', significantly reducing dataset size (~2.6M rows remaining).
Converted boolean features (Amenity, Bump, etc.) to integers.
Feature Engineering (GPU-accelerated):
One-Hot Encoded low-cardinality categorical features ('Sunrise_Sunset', 'State', 'Source') using cudf.get_dummies.
Frequency Encoded high-cardinality categorical features ('Weather_Condition', 'Wind_Direction', 'City', 'County', 'Zipcode') using cudf's value_counts and map.
Data Splitting: Split data into 70% training / 30% test sets using cuml.model_selection.train_test_split. Final training set size: ~1.8M samples.
Target Variable: Transformed 'Severity' from 1-4 to 0-3 for model compatibility.
Scaling: Applied cuml.preprocessing.StandardScaler to all features after encoding.
Modeling (GPU-accelerated): Trained three classification models:
cuML Logistic Regression (solver='qn')
cuML Random Forest (n_estimators=300, max_depth=16)
XGBoost Classifier (tree_method='gpu_hist', n_estimators=100, max_depth=10)
Results & Performance:
Model Accuracy (on Test Set):
XGBoost: ~90.8%
Random Forest: ~87.6%
Logistic Regression: ~87.0%
GPU Training Times:
XGBoost: ~7.9s
Random Forest: ~54.4s
Logistic Regression: ~1.3s
GPU Prediction Times (on ~788k test samples):
XGBoost: ~0.23s
Random Forest: ~0.39s
Logistic Regression: ~0.02s
Note: Accuracy is high, but given class imbalance, other metrics (precision/recall/F1 per class) would provide deeper insight. Training and prediction times demonstrate significant GPU speed advantages.
Conclusion:
The project successfully implemented a GPU-accelerated workflow using RAPIDS to predict accident severity on a large dataset. Data preprocessing and model training were performed efficiently on the GPU. XGBoost achieved the highest accuracy (90.8%) with fast training and inference times, highlighting the effectiveness of GPU acceleration for this machine learning task.
Artifacts: The trained XGBoost model and the data scaler were saved using joblib for potential deployment or further use.
