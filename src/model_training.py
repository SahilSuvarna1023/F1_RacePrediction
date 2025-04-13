import joblib
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#  Load dataset
MERGED_DATA_PATH = "data/merged/cleaned_merged_data2.csv"

if not os.path.exists(MERGED_DATA_PATH):
    raise FileNotFoundError(" Cleaned dataset not found! Run data_merging.py first.")

df = pd.read_csv(MERGED_DATA_PATH)

#  Select features & target (Updated to include constructor features)
selected_features = ["grid", "driver_experience", "avg_team_points", "laps", 
                     "year", "constructor_standings", "constructor_points"]
target = "positionOrder"

#  Ensure all selected features exist
missing_features = [col for col in selected_features if col not in df.columns]
if missing_features:
    raise KeyError(f" Missing Features in Dataset: {missing_features}")

#  Prepare data
X = df[selected_features]
y = df[target]

#  Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Handle missing values before scaling
X_train.fillna(X_train.mean(), inplace=True)
X_test.fillna(X_test.mean(), inplace=True)

#  Standardize Numerical Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#  Save Scaler
joblib.dump(scaler, "models/scaler.pkl")

#  Train Random Forest Model
rf_model = RandomForestRegressor(n_estimators=150, random_state=42)
rf_model.fit(X_train_scaled, y_train)

#  Train XGBoost Model
xgb_model = XGBRegressor(n_estimators=150, learning_rate=0.1, max_depth=6, random_state=42)
xgb_model.fit(X_train_scaled, y_train)

#  Save Models
joblib.dump(rf_model, "models/best_random_forest.pkl")
joblib.dump(xgb_model, "models/best_xgboost.pkl")
joblib.dump(selected_features, "models/selected_features.pkl")  # Save feature list for future use

#  Evaluate Models
def evaluate_model(model, X_test_scaled, y_test, model_name):
    preds = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print(f"\n🏎️ {model_name} Performance:")
    print(f"   - MAE: {mae:.4f}")
    print(f"   - RMSE: {rmse:.4f}")
    print(f"   - R² Score: {r2:.4f}")

evaluate_model(rf_model, X_test_scaled, y_test, "Random Forest")
evaluate_model(xgb_model, X_test_scaled, y_test, "XGBoost")

print("\n Model Training Complete! Models saved in `models/` folder.")


