import joblib
import pandas as pd
from flask import Flask, request, jsonify


rf_model = joblib.load("models/best_random_forest.pkl")
xgb_model = joblib.load("models/best_xgboost.pkl")
scaler = joblib.load("models/scaler.pkl")


app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "🏎️ F1 Race Prediction API is running!",
        "usage": "Send POST requests to /predict with JSON payload to get race predictions."
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
       
        input_data = request.get_json()

        
        input_df = pd.DataFrame([input_data])

        
        input_scaled = scaler.transform(input_df)

        
        rf_pred = rf_model.predict(input_scaled)[0]
        xgb_pred = xgb_model.predict(input_scaled)[0]

        
        return jsonify({
            "Random_Forest_Prediction": float(round(rf_pred, 2)),
            "XGBoost_Prediction": float(round(xgb_pred, 2)),
            "note": "Lower predicted value = better race finishing position"
        })


    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":

    app.run(host="0.0.0.0", port=5000, debug=True)
