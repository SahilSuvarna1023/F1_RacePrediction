import joblib
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

# ✅ Print Results
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
