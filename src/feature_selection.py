import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt

import joblib
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# ✅ Load dataset
MERGED_DATA_PATH = "data/merged/cleaned_merged_data2.csv"
df = pd.read_csv(MERGED_DATA_PATH)

# ✅ Select features & target
selected_features = ["grid", "driver_experience", "avg_team_points", "laps", 
                     "year", "constructor_standings", "constructor_points"]
target = "positionOrder"
X = df[selected_features]
y = df[target]

# ✅ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Standardize Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ✅ Add Constant for AIC/BIC Calculation (Required for statsmodels)
X_train_const = sm.add_constant(X_train_scaled)

# ✅ Fit OLS Model (Ordinary Least Squares) for AIC & BIC
ols_model = sm.OLS(y_train, X_train_const).fit()

# ✅ Calculate AIC & BIC
aic_value = ols_model.aic
bic_value = ols_model.bic

## ✅ Print AIC & BIC
print("\n📊 AIC & BIC Scores for Feature Selection:")
print(f"   - AIC: {aic_value:.2f} (Lower is better)")
print(f"   - BIC: {bic_value:.2f} (Lower is better)")

# ✅ Display Feature Importance from OLS Model
feature_importance = pd.DataFrame({
    "Feature": ["Intercept"] + selected_features, 
    "Coefficient": ols_model.params
}).sort_values(by="Coefficient", ascending=False)

print("\n🏎️ Feature Importance (Regression Coefficients):")
print(feature_importance)

# ===================== 📈 FEATURE IMPORTANCE VISUALIZATIONS ===================== #

# ✅ Random Forest Feature Importance
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_importance = pd.DataFrame({
    "feature": selected_features,
    "importance": rf_model.feature_importances_
}).sort_values(by="importance", ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(x="importance", y="feature", data=rf_importance, palette="YlGnBu")
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("outputs/feature_importance_random_forest.png")
plt.show()

# ✅ XGBoost Feature Importance
xgb_model = XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_importance = pd.DataFrame({
    "feature": selected_features,
    "importance": xgb_model.feature_importances_
}).sort_values(by="importance", ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(x="importance", y="feature", data=xgb_importance, palette="coolwarm")
plt.title("Feature Importance (XGBoost)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("outputs/feature_importance_xgboost.png")
plt.show()

