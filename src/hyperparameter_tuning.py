import joblib
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import ttest_ind


MERGED_DATA_PATH = "data/merged/cleaned_merged_data2.csv"

if not os.path.exists(MERGED_DATA_PATH):
    raise FileNotFoundError(" Cleaned dataset not found! Run `data_merging.py` first.")

df = pd.read_csv(MERGED_DATA_PATH)


selected_features = ["grid", "driver_experience", "avg_team_points", "laps", 
                     "year", "constructor_standings", "constructor_points"]
target = "positionOrder"


missing_features = [col for col in selected_features if col not in df.columns]
if missing_features:
    raise KeyError(f" Missing Features in Dataset: {missing_features}")


X = df[selected_features]
y = df[target]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X_train.fillna(X_train.mean(), inplace=True)
X_test.fillna(X_test.mean(), inplace=True)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


joblib.dump(scaler, "models/scaler.pkl")


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
print("\n Best Random Forest Hyperparameters:", rf_search.best_params_)


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
print("\n Best XGBoost Hyperparameters:", xgb_search.best_params_)


def evaluate_model(model, X_test_scaled, y_test, model_name):
    preds = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print(f"\n {model_name} Performance:")
    print(f"   - MAE: {mae:.4f}")
    print(f"   - RMSE: {rmse:.4f}")
    print(f"   - R² Score: {r2:.4f}")

    return preds

rf_preds = evaluate_model(best_rf_model, X_test_scaled, y_test, "Random Forest (Tuned)")
xgb_preds = evaluate_model(best_xgb_model, X_test_scaled, y_test, "XGBoost (Tuned)")


print("\n Conducting T-Test Between Models:")
t_stat, p_value = ttest_ind(rf_preds, xgb_preds)

print(f"\n  Random Forest vs XGBoost")
print(f"   - T-Statistic: {t_stat:.4f}")
print(f"   - P-Value: {p_value:.6f}")

if p_value < 0.05:
    print("    Significant difference detected!")
else:
    print("    No significant difference, models perform similarly.")


joblib.dump(best_rf_model, "models/best_random_forest_tuned.pkl")
joblib.dump(best_xgb_model, "models/best_xgboost_tuned.pkl")

print("\n Hyperparameter Tuning Complete! Best models saved in `models/` folder.")


