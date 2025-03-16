import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ✅ Load the cleaned merged dataset
MERGED_DATA_PATH = "data/merged/cleaned_merged_data.csv"

if not os.path.exists(MERGED_DATA_PATH):
    raise FileNotFoundError("❌ Cleaned merged dataset not found! Run data_merging.py first.")

df = pd.read_csv(MERGED_DATA_PATH, low_memory=False)

# ✅ Step 1: Feature Selection
selected_features = ["grid", "driver_experience", "avg_team_points", "laps", "year"]
target = "positionOrder"  # Predicting race finishing position

# ✅ Step 2: Handle Missing Values
df.fillna(df.mean(numeric_only=True), inplace=True)  # Replace missing values with column mean

# ✅ Step 3: Train-Test Split
X = df[selected_features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Step 4: Standardize Numerical Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ✅ Step 5: Train Random Forest Model
rf_model = RandomForestRegressor(n_estimators=150, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# ✅ Step 6: Evaluate Random Forest
rf_preds = rf_model.predict(X_test_scaled)
rf_mae = mean_absolute_error(y_test, rf_preds)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
rf_r2 = r2_score(y_test, rf_preds)

print("\n🔹 Random Forest Results:")
print(f"   - MAE: {rf_mae:.4f}")
print(f"   - RMSE: {rf_rmse:.4f}")
print(f"   - R² Score: {rf_r2:.4f}")

# ✅ Step 7: Train XGBoost Model
xgb_model = XGBRegressor(n_estimators=150, learning_rate=0.1, max_depth=6, random_state=42)
xgb_model.fit(X_train_scaled, y_train)

# ✅ Step 8: Evaluate XGBoost
xgb_preds = xgb_model.predict(X_test_scaled)
xgb_mae = mean_absolute_error(y_test, xgb_preds)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
xgb_r2 = r2_score(y_test, xgb_preds)

print("\n🏎️ XGBoost Results:")
print(f"   - MAE: {xgb_mae:.4f}")
print(f"   - RMSE: {xgb_rmse:.4f}")
print(f"   - R² Score: {xgb_r2:.4f}")

# ✅ Step 9: Feature Importance Analysis
feature_importance_rf = rf_model.feature_importances_
feature_importance_xgb = xgb_model.feature_importances_

# Convert to DataFrame for visualization
feature_df = pd.DataFrame({
    "Feature": selected_features,
    "Random Forest Importance": feature_importance_rf,
    "XGBoost Importance": feature_importance_xgb
}).sort_values(by="XGBoost Importance", ascending=False)

# Plot Feature Importance
plt.figure(figsize=(10, 5))
sns.barplot(x="XGBoost Importance", y="Feature", hue="Feature", data=feature_df, palette="coolwarm", legend=False)
plt.title("Feature Importance (XGBoost)")
plt.xlabel("Importance")
plt.ylabel("Feature")

plt.savefig("models/feature_importance_xgboost.png", bbox_inches="tight")
print("📊 Feature Importance plot saved as 'models/feature_importance_xgboost.png'")
# ✅ Step 10: Save Trained Models
os.makedirs("models", exist_ok=True)

joblib.dump(rf_model, "models/random_forest_model.pkl")
joblib.dump(xgb_model, "models/xgboost_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")  # Save scaler for future predictions

print("\n✅ Models saved successfully!")
