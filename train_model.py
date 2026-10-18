
import json
import pickle

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Load and prepare data
data = pd.read_csv("sales_data.csv", parse_dates=["Date"])
data = data.sort_values("Date").reset_index(drop=True)

# Time-based features
data["DayOfWeek"] = data["Date"].dt.dayofweek
data["Month"] = data["Date"].dt.month
data["DayOfYear"] = data["Date"].dt.dayofyear

features = [
    "Units_Sold",
    "Unit_Price",
    "Discount_Percent",
    "Advertising_Spend",
    "DayOfWeek",
    "Month",
    "DayOfYear",
    "Product",
    "Region",
]

target = "Revenue"

X = data[features].copy()
y = data[target].copy()

# One-hot encode categorical features
X = pd.get_dummies(X, columns=["Product", "Region"], dtype=int)

# Chronological split: earliest 80% for training, latest 20% for testing
split_index = int(len(data) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

models = {
    "Ridge Regression": make_pipeline(
        StandardScaler(),
        Ridge(alpha=1.0)
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),
    "Gradient Boosting": HistGradientBoostingRegressor(
        max_iter=150,
        learning_rate=0.08,
        random_state=42
    ),
}

results = []
trained_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4),
    })
    trained_models[name] = model

# Select model with the lowest test MAE
results_df = pd.DataFrame(results).sort_values("MAE")
best_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_name]

# Save model, feature order and metadata
bundle = {
    "model": best_model,
    "model_name": best_name,
    "features": features,
    "encoded_columns": list(X.columns),
    "metrics": results_df.to_dict(orient="records"),
    "training_rows": len(X_train),
    "testing_rows": len(X_test),
}

with open("sales_model.pkl", "wb") as file:
    pickle.dump(bundle, file)

results_df.to_csv("model_comparison.csv", index=False)

with open("model_metrics.json", "w", encoding="utf-8") as file:
    json.dump(bundle["metrics"], file, indent=2)

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))
print(f"\nBest model by test MAE: {best_name}")
print(f"Training rows: {len(X_train)} | Testing rows: {len(X_test)}")
print("Saved sales_model.pkl, model_comparison.csv, and model_metrics.json")
