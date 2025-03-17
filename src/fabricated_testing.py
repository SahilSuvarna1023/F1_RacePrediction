import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import ttest_ind

# ✅ Load trained models
models = {
    "Random Forest": joblib.load("models/best_random_forest.pkl"),
    "XGBoost": joblib.load("models/best_xgboost.pkl"),
}

# ✅ Load the scaler
scaler = joblib.load("models/scaler.pkl")

# ✅ Load fabricated test dataset
FABRICATED_DATA_PATH = "data/fabricated_test_data.csv"
fabricated_df = pd.read_csv(FABRICATED_DATA_PATH)

# ✅ Select features
selected_features = ["grid", "driver_experience", "avg_team_points", "laps", "year"]
X_fabricated = fabricated_df[selected_features]

# ✅ Scale fabricated test data
X_fabricated_scaled = scaler.transform(X_fabricated)

# ✅ Store predictions for all models
predictions = {}

print("\n📊 Model Performance on Fabricated Data:")
for model_name, model in models.items():
    preds = model.predict(X_fabricated_scaled)
    predictions[model_name] = preds  # Store predictions

    print(f"\n🏎️ {model_name} Predictions (First 5 Samples):")
    print(preds[:5])

# ✅ Conduct T-Test to compare Random Forest vs XGBoost
print("\n📊 Conducting T-Test Between Models:")
rf_preds = predictions["Random Forest"]
xgb_preds = predictions["XGBoost"]

t_stat, p_value = ttest_ind(rf_preds, xgb_preds)

print(f"\n🔹 Random Forest vs XGBoost")
print(f"   - T-Statistic: {t_stat:.4f}")
print(f"   - P-Value: {p_value:.6f}")

if p_value < 0.05:
    print("   🔥 Significant difference detected!")
else:
    print("   ✅ No significant difference, models perform similarly.")
