
import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_anomalies(data):
    """
    Flag unusual sales records using Isolation Forest.
    Returns a copy of the data with anomaly labels and scores.
    """
    required = ["Revenue", "Profit", "Units_Sold", "Discount_Percent"]

    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    result = data.copy()

    features = result[required].replace(
        [float("inf"), float("-inf")], float("nan")
    )

    valid = features.notna().all(axis=1)
    result["Anomaly"] = "Not evaluated"
    result["Anomaly_Score"] = float("nan")

    if valid.sum() < 10:
        return result

    detector = IsolationForest(
        n_estimators=150,
        contamination=0.05,
        random_state=42
    )

    labels = detector.fit_predict(features.loc[valid])

    result.loc[valid, "Anomaly"] = [
        "Unusual" if label == -1 else "Normal"
        for label in labels
    ]
    result.loc[valid, "Anomaly_Score"] = detector.decision_function(
        features.loc[valid]
    )

    return result


if __name__ == "__main__":
    sales = pd.read_csv("sales_data.csv")
    result = detect_anomalies(sales)

    unusual = result[result["Anomaly"] == "Unusual"]

    print(f"Total records: {len(result)}")
    print(f"Unusual records flagged: {len(unusual)}")
    print("\nSample unusual records:")

    columns = ["Revenue", "Profit", "Units_Sold", "Anomaly_Score"]
    print(unusual[columns].head(10).to_string(index=False))

    result.to_csv("sales_anomaly_results.csv", index=False)
    print("\nSaved results to sales_anomaly_results.csv")
