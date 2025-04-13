import joblib
import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import ttest_ind


models = {
    "Random Forest": joblib.load("models/best_random_forest.pkl"),
    "XGBoost": joblib.load("models/best_xgboost.pkl"),
}


scaler = joblib.load("models/scaler.pkl")


trained_features = joblib.load("models/selected_features.pkl")

MERGED_DATA_PATH = "data/merged/cleaned_merged_data2.csv"


if not os.path.exists(MERGED_DATA_PATH):
    raise FileNotFoundError(f" Dataset not found at {MERGED_DATA_PATH}. Please check `data_merging.py`.")


df = pd.read_csv(MERGED_DATA_PATH)


missing_features = [col for col in trained_features if col not in df.columns]
if missing_features:
    raise KeyError(f" Missing Features in Dataset: {missing_features}. Please check `data_merging.py`.")


X_test = df[trained_features].tail(100)  
y_test = df["positionOrder"].tail(100)  


X_test.fillna(X_test.mean(), inplace=True)


X_test_scaled = scaler.transform(X_test)

predictions = {}


print("\n Model Performance Metrics:")
for model_name, model in models.items():
    preds = model.predict(X_test_scaled)
    predictions[model_name] = preds  

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print(f"\n {model_name} Performance:")
    print(f"   - MAE: {mae:.4f}")
    print(f"   - RMSE: {rmse:.4f}")
    print(f"   - R² Score: {r2:.4f}")

print("\nConducting T-Test Between Models:")

rf_preds = predictions["Random Forest"]
xgb_preds = predictions["XGBoost"]

t_stat, p_value = ttest_ind(rf_preds, xgb_preds)

print(f"\n🔹 Random Forest vs XGBoost")
print(f"   - T-Statistic: {t_stat:.4f}")
print(f"   - P-Value: {p_value:.6f}")

if p_value < 0.05:
    print("Significant difference detected!")
else:
    print("No significant difference, models perform similarly.")

print("\n Model Evaluation Complete!")
