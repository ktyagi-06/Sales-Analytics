"""
Interactive sales analytics dashboard.

Run with:
    streamlit run app.py
"""
from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import load_config, load_sales_data
from kpi_analysis import monthly_revenue_trend, top_n_breakdown, order_level_summary
from rfm_segmentation import compute_rfm_table, score_rfm, cluster_customers
from cohort_analysis import build_cohort_table, retention_matrix
from forecasting import get_monthly_series, fit_holt_winters

st.set_page_config(page_title="Sales Analytics Dashboard", layout="wide")
st.title("Sales Analytics Dashboard")

config = load_config("config.yaml")
date_col = config["data"]["date_column"]

data_path = Path(config["data"]["raw_path"])
if not data_path.exists():
    st.warning(
        f"No data found at `{data_path}`. Run "
        f"`python src/generate_synthetic_data.py` first, or place your own "
        f"sales export there."
    )
    st.stop()

df = load_sales_data(config)

tab_overview, tab_segments, tab_cohorts, tab_forecast = st.tabs(
    ["Overview", "Customer Segments", "Cohort Retention", "Forecast"]
)

with tab_overview:
    summary = order_level_summary(df, date_col)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue", f"${summary['total_revenue']:,.0f}")
    c2.metric("Orders", f"{summary['n_orders']:,}")
    c3.metric("Customers", f"{summary['n_customers']:,}")
    c4.metric("Avg Order Value", f"${summary['avg_order_value']:,.2f}")

    monthly = monthly_revenue_trend(df, date_col)
    st.subheader("Monthly Revenue Trend")
    st.line_chart(monthly["Revenue"])

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Top Categories")
        st.dataframe(top_n_breakdown(df, "Category", 10) if "Category" in df.columns else pd.DataFrame())
    with col_b:
        st.subheader("Top Products")
        st.dataframe(top_n_breakdown(df, "Product", 10) if "Product" in df.columns else pd.DataFrame())

with tab_segments:
    st.subheader("RFM Customer Segments")
    rfm = compute_rfm_table(df, date_col, config["rfm"].get("reference_date"))
    rfm = score_rfm(rfm)
    rfm = cluster_customers(rfm, config["rfm"]["n_clusters"], config["rfm"]["random_state"])

    seg_counts = rfm["segment"].value_counts()
    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.dataframe(seg_counts.rename("Customers"))
    with col_b:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie(seg_counts.values, labels=seg_counts.index, autopct="%1.0f%%")
        ax.set_title("Segment Distribution")
        st.pyplot(fig)

    st.subheader("Segment Profiles")
    st.dataframe(rfm.groupby("segment")[["Recency", "Frequency", "Monetary"]].mean().round(2))

    st.subheader("Full RFM Table")
    st.dataframe(rfm.reset_index())

with tab_cohorts:
    st.subheader("Monthly Cohort Retention")
    cohort_pivot = build_cohort_table(df, date_col)
    retention = retention_matrix(cohort_pivot)

    fig, ax = plt.subplots(figsize=(12, max(4, 0.4 * len(retention))))
    sns.heatmap(retention, annot=True, fmt=".0f", cmap="YlGnBu", ax=ax, cbar_kws={"label": "% retained"})
    ax.set_xlabel("Months since first purchase")
    ax.set_ylabel("Cohort")
    st.pyplot(fig)

with tab_forecast:
    st.subheader("Revenue Forecast")
    series = get_monthly_series(df, date_col)
    horizon = st.slider("Forecast horizon (months)", 1, 12, config["forecasting"]["forecast_horizon"])

    fit = fit_holt_winters(series, config["forecasting"]["seasonal_periods"])
    forecast = fit.forecast(horizon)
    forecast.index = pd.date_range(series.index[-1] + pd.DateOffset(months=1), periods=horizon, freq="MS")

    combined = pd.concat([series.rename("Actual"), forecast.rename("Forecast")], axis=1)
    st.line_chart(combined)

    st.caption(
        "Forecasts are Holt-Winters exponential smoothing on historical monthly "
        "revenue only -- they don't account for planned promotions, price "
        "changes, or external demand shocks. Check `src/forecasting.py`'s "
        "backtest against a naive seasonal baseline before trusting this in "
        "a real planning decision."
    )
