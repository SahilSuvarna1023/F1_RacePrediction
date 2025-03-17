import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import ttest_ind

# ✅ Load all trained models
models = {
    "Random Forest": joblib.load("models/best_random_forest.pkl"),
    "XGBoost": joblib.load("models/best_xgboost.pkl"),
#"Linear Regression": joblib.load("models/best_linear_regression.pkl"),
    "Gradient Boosting": joblib.load("models/best_gradient_boosting.pkl"),
    "SVM": joblib.load("models/best_svm.pkl")
}

# ✅ Load the scaler for data transformation
scaler = joblib.load("models/scaler.pkl")

# ✅ Load fabricated test data
FABRICATED_DATA_PATH = "data/new_data.csv"
fabricated_df = pd.read_csv(FABRICATED_DATA_PATH)

# ✅ Rename columns to match the trained model
column_mapping = {
    "num_pit_stops": "grid",
    "driver_performance": "driver_experience",
    "total_races": "avg_team_points",
    "best_lap_time": "laps",
    "lap_time_variability": "year"  # Approximating year with lap variability
}

fabricated_df.rename(columns=column_mapping, inplace=True)

# ✅ Select only the relevant features
selected_features = ["grid", "driver_experience", "avg_team_points", "laps", "year"]

# ✅ Ensure only mapped columns are selected
X_fabricated = fabricated_df[selected_features]

# ✅ Scale fabricated test data
X_fabricated_scaled = scaler.transform(X_fabricated)

# ✅ Store predictions for all models
predictions = {}

# ✅ Run each model on fabricated data
for model_name, model in models.items():
    print(f"\n🏎️ Testing {model_name} on Fabricated Data...")
    
    preds = model.predict(X_fabricated_scaled)
    predictions[model_name] = preds  # Store predictions
    
    print(f"   - Predictions (first 5): {preds[:5]}")

# ✅ Conduct Pairwise T-Test Comparisons
print("\n📊 Conducting T-Tests Between Models:")

model_names = list(predictions.keys())
for i in range(len(model_names)):
    for j in range(i + 1, len(model_names)):
        model1, model2 = model_names[i], model_names[j]
        t_stat, p_value = ttest_ind(predictions[model1], predictions[model2])

        print(f"\n🔹 {model1} vs {model2}")
        print(f"   - T-Statistic: {t_stat:.4f}")
        print(f"   - P-Value: {p_value:.6f}")

        if p_value < 0.05:
            print("   🔥 Significant difference detected!")
        else:
            print("   ✅ No significant difference, models perform similarly.")
