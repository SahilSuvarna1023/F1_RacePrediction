# F1_RacePrediction
This F1 Race Prediction project uses Machine Learning models to predict Formula 1 race outcomes based on historical data, including driver experience, grid position, team performance, and track details.

This repository includes data processing, exploratory data analysis (EDA), model training, hyperparameter tuning, and a Flask API for real-time race predictions.

📌 Project Overview
✅ Goal: Predict race finishing positions based on various race parameters.
✅ Data Source: F1 statistics from Ergast API & historical records.
✅ Models Used: Random Forest & XGBoost (tuned for better accuracy).
✅ Deployment: Flask API (ready for cloud hosting).

📂 Project Structure
F1_RacePrediction/
│── data/                # Raw and processed datasets (ignored in Git)
│── models/              # Trained models (stored via Git LFS or cloud)
│── output/              # Prediction results and logs
│── src/
│   ├── data_merging.py         # Cleans and merges datasets
│   ├── exploratory_data.py      # Performs EDA and visualizations
│   ├── model_training.py        # Trains & evaluates ML models
│   ├── hyperparameter_tuning.py # Optimizes models for accuracy
│   ├── flask_api.py             # Flask API for race predictions
│   ├── utils.py                 # Helper functions
│── .gitignore            # Prevents large files from being uploaded
│── requirements.txt      # Python dependencies
│── README.md             # Documentation
│── main.py               # Runs full project pipeline

📊 Dataset Details
This project utilizes Formula 1 race history data, including:

Drivers & Experience: Driver performance history
Constructor Standings: Team rankings & points
Race Data: Track details, laps, weather conditions
Lap Times & Pit Stops: Performance trends during races
🤖 Machine Learning Models
We train and compare two models:

Model	Pros	Cons
Random Forest	Simple, robust, interpretable	Slower for large datasets
XGBoost	High accuracy, handles missing data	Requires hyperparameter tuning
🔹 Best model is chosen based on:

Mean Absolute Error (MAE)
Root Mean Squared Error (RMSE)
R² Score (Model Accuracy)

🛠️ How to Run the Project
1️⃣ Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
2️⃣ Train the Model
bash
Copy
Edit
python src/model_training.py
3️⃣ Run the Flask API
bash
Copy
Edit
python src/flask_api.py
📌 The API will run at: http://127.0.0.1:5000/

🎯 Using the API
🔹 Send a Prediction Request
Send a POST request to /predict with race input data:

Example Input
json
Copy
Edit
{
    "grid": 5,
    "driver_experience": 40,
    "avg_team_points": 120,
    "laps": 60,
    "year": 2024
}
Expected Response
json
Copy
Edit
{
    "Random_Forest_Prediction": 5.2,
    "XGBoost_Prediction": 4.8
}
✅ Lower predicted values mean a higher finishing position!

🌎 Deployment Options
🚀 Want to deploy this API? Here are some options:

Render (Free & Easy Flask Deployment)
AWS EC2 (For professional cloud hosting)
Heroku (If you prefer cloud-based solutions)
🔥 Future Improvements
🔹 Add Deep Learning models (LSTMs for time series data)
🔹 Improve feature engineering (e.g., weather impact on races)
🔹 Deploy a user-friendly web app (using Streamlit)

📜 Contributors
👨‍💻 Your Name - Sahil Suvarna


⭐ Support the Project
If you find this project useful, leave a star ⭐ on GitHub!
