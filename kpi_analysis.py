"""
Core KPI analysis: monthly revenue trend, top-N products/categories/regions,
and order-level summary stats.

Usage:
    python src/kpi_analysis.py --config config.yaml
"""
import argparse
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_loader import load_config, load_sales_data


def monthly_revenue_trend(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    monthly = (
        df.set_index(date_col)
        .resample("MS")["Sales"]
        .sum()
        .rename("Revenue")
        .to_frame()
    )
    monthly["MoM_growth_pct"] = monthly["Revenue"].pct_change() * 100
    return monthly


def top_n_breakdown(df: pd.DataFrame, group_col: str, n: int = 10) -> pd.DataFrame:
    result = (
        df.groupby(group_col)
        .agg(Revenue=("Sales", "sum"), Orders=("OrderID", "nunique"), Units=("Quantity", "sum"))
        .sort_values("Revenue", ascending=False)
        .head(n)
    )
    result["Revenue_share_pct"] = result["Revenue"] / df["Sales"].sum() * 100
    return result


def order_level_summary(df: pd.DataFrame, date_col: str) -> dict:
    order_totals = df.groupby("OrderID")["Sales"].sum()
    return {
        "total_revenue": df["Sales"].sum(),
        "n_orders": df["OrderID"].nunique(),
        "n_customers": df["CustomerID"].nunique(),
        "avg_order_value": order_totals.mean(),
        "median_order_value": order_totals.median(),
        "date_range": (df[date_col].min(), df[date_col].max()),
    }


def plot_revenue_trend(monthly: pd.DataFrame, output_path: Path):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(monthly.index, monthly["Revenue"], marker="o", markersize=3)
    ax.set_title("Monthly Revenue Trend")
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue ($)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def run_kpi_analysis(config: dict):
    date_col = config["data"]["date_column"]
    top_n = config["kpi"]["top_n"]
    output_dir = Path(config["kpi"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_sales_data(config)

    summary = order_level_summary(df, date_col)
    print("=== Order-Level Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    monthly = monthly_revenue_trend(df, date_col)
    monthly.to_csv(output_dir / "monthly_revenue.csv")
    plot_revenue_trend(monthly, output_dir / "monthly_revenue_trend.png")
    print(f"\nSaved monthly revenue trend to {output_dir}/monthly_revenue.csv")

    for col in ["Product", "Category", "Region"]:
        if col in df.columns:
            top = top_n_breakdown(df, col, top_n)
            top.to_csv(output_dir / f"top_{col.lower()}.csv")
            print(f"\n=== Top {top_n} by {col} ===")
            print(top)

    return summary, monthly


def main():
    parser = argparse.ArgumentParser(description="Run core KPI analysis on sales data.")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    run_kpi_analysis(config)


if __name__ == "__main__":
    main()
