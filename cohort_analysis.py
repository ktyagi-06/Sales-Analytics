"""
Monthly cohort retention analysis: group customers by the month of their
first purchase, then track what fraction of each cohort is still active
in each subsequent month.

Usage:
    python src/cohort_analysis.py --config config.yaml
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import load_config, load_sales_data


def build_cohort_table(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df = df.copy()
    df["order_month"] = df[date_col].dt.to_period("M")

    first_purchase = df.groupby("CustomerID")["order_month"].min().rename("cohort_month")
    df = df.merge(first_purchase, on="CustomerID")

    df["period_number"] = (df["order_month"] - df["cohort_month"]).apply(lambda x: x.n)

    cohort_data = (
        df.groupby(["cohort_month", "period_number"])["CustomerID"]
        .nunique()
        .reset_index()
    )
    cohort_pivot = cohort_data.pivot(index="cohort_month", columns="period_number", values="CustomerID")
    return cohort_pivot


def retention_matrix(cohort_pivot: pd.DataFrame) -> pd.DataFrame:
    cohort_sizes = cohort_pivot.iloc[:, 0]
    retention = cohort_pivot.divide(cohort_sizes, axis=0) * 100
    return retention.round(1)


def plot_retention_heatmap(retention: pd.DataFrame, output_path: Path):
    fig, ax = plt.subplots(figsize=(12, max(4, 0.4 * len(retention))))
    sns.heatmap(retention, annot=True, fmt=".0f", cmap="YlGnBu", ax=ax,
                cbar_kws={"label": "% retained"})
    ax.set_title("Monthly Cohort Retention (%)")
    ax.set_xlabel("Months since first purchase")
    ax.set_ylabel("Cohort (first purchase month)")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def run_cohort_analysis(config: dict):
    date_col = config["data"]["date_column"]
    output_dir = Path(config["cohort"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_sales_data(config)
    cohort_pivot = build_cohort_table(df, date_col)
    retention = retention_matrix(cohort_pivot)

    retention.to_csv(output_dir / "cohort_retention.csv")
    plot_retention_heatmap(retention, output_dir / "cohort_retention_heatmap.png")

    avg_by_period = retention.mean(axis=0)
    print("=== Average retention by months-since-first-purchase ===")
    print(avg_by_period.round(1))
    print(f"\nSaved retention matrix to {output_dir}/cohort_retention.csv")

    return retention


def main():
    parser = argparse.ArgumentParser(description="Run monthly cohort retention analysis.")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    run_cohort_analysis(config)


if __name__ == "__main__":
    main()
