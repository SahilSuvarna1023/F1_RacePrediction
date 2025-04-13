import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import ttest_ind

models = {
    "Random Forest": joblib.load("models/best_random_forest.pkl"),
    "XGBoost": joblib.load("models/best_xgboost.pkl"),
}

scaler = joblib.load("models/scaler.pkl")


FABRICATED_DATA_PATH = "data/fabricated_test_data.csv"
fabricated_df = pd.read_csv(FABRICATED_DATA_PATH)


selected_features = ["grid", "driver_experience", "avg_team_points", "laps", "year",  "constructor_standings", "constructor_points"]
X_fabricated = fabricated_df[selected_features]


X_fabricated_scaled = scaler.transform(X_fabricated)


predictions = {}

print("\nModel Performance on Fabricated Data:")
for model_name, model in models.items():
    preds = model.predict(X_fabricated_scaled)
    predictions[model_name] = preds  

    print(f"\n{model_name} Predictions (First 5 Samples):")
    print(preds[:5])


print("\n Conducting T-Test Between Models:")
rf_preds = predictions["Random Forest"]
xgb_preds = predictions["XGBoost"]

t_stat, p_value = ttest_ind(rf_preds, xgb_preds)

print(f"\n Random Forest vs XGBoost")
print(f"   - T-Statistic: {t_stat:.4f}")
print(f"   - P-Value: {p_value:.6f}")

if p_value < 0.05:
    print("Significant difference detected!")
else:
    print("No significant difference, models perform similarly.")
