

import joblib
import pandas as pd

try:
    model = joblib.load("xgb_model.joblib")
    print("Loaded xgb_model.joblib")
except:
    raise FileNotFoundError("xgb_model.joblib not found")

try:
    X_test = pd.read_csv("X_test.csv")
    print("Loaded X_test.csv")
except:
    raise FileNotFoundError("X_test.csv not found")

y_pred = model.predict(X_test)

pd.DataFrame(y_pred, columns=["predicted"]).to_csv("y_pred.csv", index=False)
print("Predictions saved to y_pred.csv")

print("Sample Predictions:")
print(y_pred[:5])
