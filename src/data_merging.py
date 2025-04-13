import pandas as pd
import os


CLEANED_DATA_PATH = "data/cleaned/"
MERGED_DATA_PATH = "data/merged/"
os.makedirs(MERGED_DATA_PATH, exist_ok=True)

drivers = pd.read_csv(os.path.join(CLEANED_DATA_PATH, "drivers_cleaned.csv"), sep=",", na_values=["\\N"], low_memory=False)
races = pd.read_csv(os.path.join(CLEANED_DATA_PATH, "races_cleaned.csv"), sep=",", na_values=["\\N"], low_memory=False)
results = pd.read_csv(os.path.join(CLEANED_DATA_PATH, "results_cleaned.csv"), sep=",", na_values=["\\N"], low_memory=False)
constructors = pd.read_csv(os.path.join(CLEANED_DATA_PATH, "constructors_cleaned.csv"), sep=",", na_values=["\\N"], low_memory=False)
driver_standings = pd.read_csv(os.path.join(CLEANED_DATA_PATH, "driver_standings_cleaned.csv"), sep=",", na_values=["\\N"], low_memory=False)
constructor_standings = pd.read_csv(os.path.join(CLEANED_DATA_PATH, "constructor_standings_cleaned.csv"), sep=",", na_values=["\\N"], low_memory=False)

drivers_selected = drivers[["driverId", "nationality"]]
merged_df = results.merge(drivers_selected, on="driverId", how="left")

merged_df = merged_df.merge(races[["raceId", "year", "circuitId"]], on="raceId", how="left")
merged_df = merged_df.merge(constructors[["constructorId", "constructorRef"]], on="constructorId", how="left")

merged_df['driver_experience'] = merged_df.groupby("driverId")["raceId"].rank(method="first")

constructor_avg_points = constructor_standings.groupby("constructorId")["points"].mean().reset_index()
constructor_avg_points.rename(columns={"points": "avg_team_points"}, inplace=True)
merged_df = merged_df.merge(constructor_avg_points, on="constructorId", how="left")

constructor_standings_selected = constructor_standings[["constructorId", "raceId", "position", "points"]]
constructor_standings_selected.rename(columns={"position": "constructor_standings", "points": "constructor_points"}, inplace=True)
merged_df = merged_df.merge(constructor_standings_selected, on=["constructorId", "raceId"], how="left")

numeric_cols = ["grid", "driver_experience", "avg_team_points", "laps", "year", "positionOrder", "constructor_standings", "constructor_points"]
for col in numeric_cols:
    merged_df[col] = pd.to_numeric(merged_df[col], errors="coerce")

merged_df = merged_df.ffill()  

merged_df["nationality"] = merged_df["nationality"].astype("category").cat.codes
merged_df["constructorRef"] = merged_df["constructorRef"].astype("category").cat.codes

merged_df["final_position"] = merged_df["positionOrder"]

merged_df["final_position"] = pd.to_numeric(merged_df["final_position"], errors="coerce")

cleaned_file_path = os.path.join(MERGED_DATA_PATH, "cleaned_merged_data2.csv")
merged_df.to_csv(cleaned_file_path, index=False)

print(f" Merged dataset saved successfully at {cleaned_file_path}")


