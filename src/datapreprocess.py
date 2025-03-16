import pandas as pd
import os

# Define the path to the data folder
DATA_PATH = "data/"

# List of dataset filenames
file_names = {
    "circuits": "circuits.csv",
    "constructor_results": "constructor_results.csv",
    "constructor_standings": "constructor_standings.csv",
    "constructors": "constructors.csv",
    "driver_standings": "driver_standings.csv",
    "drivers": "drivers.csv",
    "lap_times": "lap_times.csv",
    "pit_stops": "pit_stops.csv",
    "qualifying": "qualifying.csv",
    "races": "races.csv",
    "results": "results.csv",
    "seasons": "seasons.csv",
    "sprint_results": "sprint_results.csv",
    "status": "status.csv",
}

# Load datasets into Pandas DataFrames
dataframes = {}
for key, file in file_names.items():
    file_path = os.path.join(DATA_PATH, file)
    if os.path.exists(file_path):
        dataframes[key] = pd.read_csv(file_path)
        print(f"Loaded {key} dataset with shape {dataframes[key].shape}")
    else:
        print(f"Warning: {file} not found!")

# Function to check missing values
def check_missing_values():
    missing_values = {name: df.isnull().sum().sum() for name, df in dataframes.items()}
    print("\nMissing Values Summary:")
    for name, count in missing_values.items():
        print(f"{name}: {count} missing values")

check_missing_values()

# Handle missing values (basic strategy)
for name, df in dataframes.items():
    df.fillna(method='ffill', inplace=True)  # Forward fill for missing values
    df.fillna(method='bfill', inplace=True)  # Backward fill for missing values

print("\nMissing values handled successfully!")

# Convert data types
def convert_data_types():
    # Convert integer columns
    for df_name, df in dataframes.items():
        for col in df.select_dtypes(include=['float']).columns:
            if df[col].dropna().apply(float.is_integer).all():
                df[col] = df[col].astype('Int64')
    print("\nData types converted for efficiency!")

convert_data_types()

# Save cleaned datasets
CLEANED_DATA_PATH = "data/cleaned/"
os.makedirs(CLEANED_DATA_PATH, exist_ok=True)

for name, df in dataframes.items():
    df.to_csv(f"{CLEANED_DATA_PATH}{name}_cleaned.csv", index=False)
    print(f"Saved cleaned {name} dataset.")

print("\nData Preprocessing Completed Successfully!")
