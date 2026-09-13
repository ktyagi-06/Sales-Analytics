"""
RFM (Recency, Frequency, Monetary) customer segmentation.

Each customer gets a Recency/Frequency/Monetary score, quintile-binned
1-5, then KMeans clusters customers on those three scores. Clusters are
labeled by their centroid profile so the output means something to a
stakeholder, not just "cluster 2".

Usage:
    python src/rfm_segmentation.py --config config.yaml
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from data_loader import load_config, load_sales_data


def compute_rfm_table(df: pd.DataFrame, date_col: str, reference_date=None) -> pd.DataFrame:
    if reference_date is None:
        reference_date = df[date_col].max() + pd.Timedelta(days=1)
    else:
        reference_date = pd.Timestamp(reference_date)

    rfm = df.groupby("CustomerID").agg(
        Recency=(date_col, lambda x: (reference_date - x.max()).days),
        Frequency=("OrderID", "nunique"),
        Monetary=("Sales", "sum"),
    )
    return rfm


def score_rfm(rfm: pd.DataFrame) -> pd.DataFrame:
    """Quintile-bin each dimension 1-5. Recency is inverted (lower days = better = 5)."""
    rfm = rfm.copy()
    rfm["R_score"] = pd.qcut(rfm["Recency"], 5, labels=[5, 4, 3, 2, 1], duplicates="drop").astype(int)
    rfm["F_score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5], duplicates="drop").astype(int)
    rfm["M_score"] = pd.qcut(rfm["Monetary"], 5, labels=[1, 2, 3, 4, 5], duplicates="drop").astype(int)
    rfm["RFM_score"] = rfm["R_score"] + rfm["F_score"] + rfm["M_score"]
    return rfm


def cluster_customers(rfm: pd.DataFrame, n_clusters: int, random_state: int) -> pd.DataFrame:
    rfm = rfm.copy()
    features = rfm[["R_score", "F_score", "M_score"]]
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    rfm["cluster"] = kmeans.fit_predict(X)

    rfm["segment"] = label_clusters(rfm)
    return rfm


def label_clusters(rfm: pd.DataFrame) -> pd.Series:
    """Assign a human-readable label to each cluster based on its mean
    R/F/M scores relative to the overall population."""
    cluster_profiles = rfm.groupby("cluster")[["R_score", "F_score", "M_score"]].mean()
    overall_mean = rfm[["R_score", "F_score", "M_score"]].mean().mean()

    labels = {}
    for cluster_id, row in cluster_profiles.iterrows():
        avg_score = row.mean()
        high_r, high_f, high_m = row["R_score"] >= overall_mean, row["F_score"] >= overall_mean, row["M_score"] >= overall_mean

        if high_r and high_f and high_m:
            labels[cluster_id] = "Champions"
        elif high_f and high_m and not high_r:
            labels[cluster_id] = "Loyal (at risk of lapsing)"
        elif high_r and not high_f and not high_m:
            labels[cluster_id] = "New / Low-Value"
        elif not high_r and not high_f and not high_m:
            labels[cluster_id] = "Lost / Churned"
        elif high_r and (high_f or high_m):
            labels[cluster_id] = "Promising"
        else:
            labels[cluster_id] = "At Risk"

    return rfm["cluster"].map(labels)


def run_rfm_segmentation(config: dict):
    date_col = config["data"]["date_column"]
    output_dir = Path(config["rfm"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_sales_data(config)
    reference_date = config["rfm"].get("reference_date")

    rfm = compute_rfm_table(df, date_col, reference_date)
    rfm = score_rfm(rfm)
    rfm = cluster_customers(rfm, config["rfm"]["n_clusters"], config["rfm"]["random_state"])

    rfm.to_csv(output_dir / "rfm_segments.csv")

    print("=== Segment sizes ===")
    print(rfm["segment"].value_counts())

    print("\n=== Segment profiles (mean values) ===")
    print(rfm.groupby("segment")[["Recency", "Frequency", "Monetary"]].mean().round(2))

    print(f"\nSaved full RFM table to {output_dir}/rfm_segments.csv")
    return rfm


def main():
    parser = argparse.ArgumentParser(description="Run RFM customer segmentation.")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    run_rfm_segmentation(config)


if __name__ == "__main__":
    main()
