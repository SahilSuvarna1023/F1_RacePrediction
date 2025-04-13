import joblib
import pandas as pd
from flask import Flask, request, jsonify

#  Load trained models and scaler
rf_model = joblib.load("models/best_random_forest.pkl")
xgb_model = joblib.load("models/best_xgboost.pkl")
scaler = joblib.load("models/scaler.pkl")

#  Initialize Flask App
app = Flask(__name__)

#  Define API Routes
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "🏎️ F1 Race Prediction API is running!"})

@app.route("/predict", methods=["POST"])
def predict_race_outcome():
    try:
        #  Get input JSON
        input_json = request.get_json()
        
        #  Convert input to DataFrame
        input_df = pd.DataFrame([input_json])

        #  Scale the input data
        input_scaled = scaler.transform(input_df)

        #  Make predictions
        rf_pred = rf_model.predict(input_scaled)[0]
        xgb_pred = xgb_model.predict(input_scaled)[0]

        #  Return predictions
        return jsonify({
            "Random_Forest_Prediction": round(rf_pred, 2),
            "XGBoost_Prediction": round(xgb_pred, 2)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)})

#  Run the Flask App
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
