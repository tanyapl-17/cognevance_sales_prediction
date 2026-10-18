
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def segment_customers(data, clusters=4):
    """
    Demonstration segmentation of sales records using K-Means.
    Note: the dataset has no customer IDs, so these are sales-profile
    segments, not true customer-level segments.
    """
    required = ["Revenue", "Profit", "Units_Sold", "Discount_Percent"]
    missing = [col for col in required if col not in data.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    result = data.copy()
    features = result[required].copy()

    valid = features.notna().all(axis=1)

    if valid.sum() < clusters:
        raise ValueError(
            f"Need at least {clusters} complete records to create segments."
        )

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features.loc[valid])

    model = KMeans(
        n_clusters=clusters,
        random_state=42,
        n_init=10,
    )

    labels = model.fit_predict(scaled)

    result["Segment"] = "Not assigned"
    result.loc[valid, "Segment"] = [
        f"Segment {label + 1}" for label in labels
    ]

    return result


if __name__ == "__main__":
    data = pd.read_csv("sales_data.csv")
    segmented = segment_customers(data)

    summary = (
        segmented[segmented["Segment"] != "Not assigned"]
        .groupby("Segment")
        .agg(
            Records=("Revenue", "size"),
            Average_Revenue=("Revenue", "mean"),
            Average_Profit=("Profit", "mean"),
            Average_Units=("Units_Sold", "mean"),
            Average_Discount=("Discount_Percent", "mean"),
        )
        .round(2)
        .reset_index()
    )

    print("\nSALES PROFILE SEGMENTS")
    print(summary.to_string(index=False))

    segmented.to_csv("sales_segments.csv", index=False)
    summary.to_csv("segment_summary.csv", index=False)

    print("\nSaved sales_segments.csv and segment_summary.csv")
