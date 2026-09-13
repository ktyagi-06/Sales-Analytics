"""
Short-horizon monthly revenue forecasting using Holt-Winters exponential
smoothing (trend + seasonality), benchmarked against a naive seasonal
baseline (this month = same month last year). A model that can't beat
the naive baseline isn't earning its complexity.

Usage:
    python src/forecasting.py --config config.yaml
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

from data_loader import load_config, load_sales_data


def get_monthly_series(df: pd.DataFrame, date_col: str) -> pd.Series:
    series = df.set_index(date_col).resample("MS")["Sales"].sum()
    return series


def naive_seasonal_forecast(series: pd.Series, horizon: int, seasonal_periods: int) -> pd.Series:
    """Forecast each future month as equal to the same month `seasonal_periods` back."""
    forecast_index = pd.date_range(series.index[-1] + pd.DateOffset(months=1), periods=horizon, freq="MS")
    values = []
    for i, date in enumerate(forecast_index):
        lookback_idx = len(series) - seasonal_periods + i
        if 0 <= lookback_idx < len(series):
            values.append(series.iloc[lookback_idx])
        else:
            values.append(series.iloc[-1])  # fallback if history is too short
    return pd.Series(values, index=forecast_index, name="naive_forecast")


def fit_holt_winters(train_series: pd.Series, seasonal_periods: int):
    model = ExponentialSmoothing(
        train_series,
        trend="add",
        seasonal="add" if len(train_series) >= 2 * seasonal_periods else None,
        seasonal_periods=seasonal_periods if len(train_series) >= 2 * seasonal_periods else None,
    )
    return model.fit()


def backtest(series: pd.Series, test_horizon: int, seasonal_periods: int) -> dict:
    """Hold out the last `test_horizon` months, forecast them, and compare
    Holt-Winters against the naive seasonal baseline."""
    if len(series) <= test_horizon + seasonal_periods:
        print("Warning: not enough history for a robust backtest with this seasonal period; "
              "results below should be treated as indicative, not conclusive.")

    train = series.iloc[:-test_horizon]
    test = series.iloc[-test_horizon:]

    hw_fit = fit_holt_winters(train, seasonal_periods)
    hw_forecast = hw_fit.forecast(test_horizon)

    naive_forecast = naive_seasonal_forecast(train, test_horizon, seasonal_periods)

    metrics = {
        "holt_winters_mae": mean_absolute_error(test, hw_forecast),
        "holt_winters_mape": mean_absolute_percentage_error(test, hw_forecast) * 100,
        "naive_mae": mean_absolute_error(test, naive_forecast.values),
        "naive_mape": mean_absolute_percentage_error(test, naive_forecast.values) * 100,
    }
    return metrics, test, hw_forecast, naive_forecast


def plot_forecast(series: pd.Series, future_forecast: pd.Series, output_path: Path):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(series.index, series.values, label="Actual", marker="o", markersize=3)
    ax.plot(future_forecast.index, future_forecast.values, label="Forecast (Holt-Winters)",
             marker="o", markersize=3, linestyle="--", color="crimson")
    ax.set_title("Monthly Revenue: Actual + Forecast")
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue ($)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def run_forecasting(config: dict):
    date_col = config["data"]["date_column"]
    seasonal_periods = config["forecasting"]["seasonal_periods"]
    forecast_horizon = config["forecasting"]["forecast_horizon"]
    test_horizon = config["forecasting"]["test_horizon"]
    output_dir = Path(config["forecasting"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_sales_data(config)
    series = get_monthly_series(df, date_col)

    print(f"Monthly series: {len(series)} months of history "
          f"({series.index[0].date()} to {series.index[-1].date()})")

    metrics, test, hw_backtest_forecast, naive_backtest_forecast = backtest(
        series, test_horizon, seasonal_periods
    )
    print("\n=== Backtest (last {} months held out) ===".format(test_horizon))
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}")

    beat_naive = metrics["holt_winters_mape"] < metrics["naive_mape"]
    print(f"\nHolt-Winters {'beats' if beat_naive else 'does NOT beat'} the naive seasonal baseline on MAPE.")

    # Fit on the FULL series for the actual future forecast.
    final_fit = fit_holt_winters(series, seasonal_periods)
    future_forecast = final_fit.forecast(forecast_horizon)
    future_forecast.index = pd.date_range(series.index[-1] + pd.DateOffset(months=1),
                                            periods=forecast_horizon, freq="MS")

    future_forecast.to_frame("forecast_revenue").to_csv(output_dir / "revenue_forecast.csv")
    plot_forecast(series, future_forecast, output_dir / "revenue_forecast.png")

    print(f"\n=== {forecast_horizon}-month forward forecast ===")
    print(future_forecast.round(2))
    print(f"\nSaved forecast to {output_dir}/revenue_forecast.csv")

    return metrics, future_forecast


def main():
    parser = argparse.ArgumentParser(description="Forecast monthly revenue.")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    run_forecasting(config)


if __name__ == "__main__":
    main()
