import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ✅ Load the cleaned dataset
MERGED_DATA_PATH = "data/merged/cleaned_merged_data.csv"
if not os.path.exists(MERGED_DATA_PATH):
    raise FileNotFoundError("❌ Cleaned dataset not found! Run data_merging.py first.")

df = pd.read_csv(MERGED_DATA_PATH, low_memory=False)

# ✅ Feature Selection
selected_features = ["grid", "driver_experience", "avg_team_points", "laps", "year"]
target = "positionOrder"

# ✅ Handle Missing Values
df.fillna(df.mean(numeric_only=True), inplace=True)

# ✅ Train-Test Split
X = df[selected_features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Standardize Numerical Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ✅ Hyperparameter Tuning for Random Forest
rf_param_grid = {
    "n_estimators": [50, 100, 200, 300],
    "max_depth": [10, 20, 30, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

rf_model = RandomForestRegressor(random_state=42)
rf_search = RandomizedSearchCV(rf_model, rf_param_grid, n_iter=10, cv=3, n_jobs=-1, random_state=42)
rf_search.fit(X_train_scaled, y_train)

best_rf_model = rf_search.best_estimator_
print("\n✅ Best Random Forest Model:", rf_search.best_params_)

# ✅ Hyperparameter Tuning for XGBoost
xgb_param_grid = {
    "n_estimators": [50, 100, 200, 300],
    "max_depth": [3, 6, 9, 12],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
}

xgb_model = XGBRegressor(random_state=42)
xgb_search = RandomizedSearchCV(xgb_model, xgb_param_grid, n_iter=10, cv=3, n_jobs=-1, random_state=42)
xgb_search.fit(X_train_scaled, y_train)

best_xgb_model = xgb_search.best_estimator_
print("\n✅ Best XGBoost Model:", xgb_search.best_params_)

# ✅ Evaluate the Tuned Models
def evaluate_model(model, X_test_scaled, y_test, name="Model"):
    preds = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    print(f"\n🏎️ {name} Results:")
    print(f"   - MAE: {mae:.4f}")
    print(f"   - RMSE: {rmse:.4f}")
    print(f"   - R² Score: {r2:.4f}")

evaluate_model(best_rf_model, X_test_scaled, y_test, "Random Forest")
evaluate_model(best_xgb_model, X_test_scaled, y_test, "XGBoost")

# ✅ Save Optimized Models
os.makedirs("models", exist_ok=True)
joblib.dump(best_rf_model, "models/best_random_forest.pkl")
joblib.dump(best_xgb_model, "models/best_xgboost.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\n✅ Optimized models saved successfully!")
