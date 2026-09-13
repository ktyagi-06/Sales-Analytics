import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rfm_segmentation import compute_rfm_table, score_rfm, cluster_customers
from cohort_analysis import build_cohort_table, retention_matrix
from kpi_analysis import monthly_revenue_trend, top_n_breakdown, order_level_summary


def make_dummy_sales(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    order_id = 1
    base_date = pd.Timestamp("2024-01-01")

    # 20 customers with varying order counts/recency so RFM has real spread.
    for cust_idx in range(20):
        n_orders = rng.integers(1, 10)
        for _ in range(n_orders):
            day_offset = rng.integers(0, 500)
            rows.append({
                "OrderID": f"ORD-{order_id}",
                "OrderDate": base_date + pd.Timedelta(days=int(day_offset)),
                "CustomerID": f"CUST-{cust_idx}",
                "Product": "Widget",
                "Category": "Misc",
                "Region": rng.choice(["North", "South"]),
                "Quantity": rng.integers(1, 5),
                "Sales": round(rng.uniform(10, 500), 2),
            })
            order_id += 1

    return pd.DataFrame(rows)


def test_compute_rfm_table_shape():
    df = make_dummy_sales()
    rfm = compute_rfm_table(df, "OrderDate")
    assert len(rfm) == df["CustomerID"].nunique()
    assert set(["Recency", "Frequency", "Monetary"]).issubset(rfm.columns)
    assert (rfm["Recency"] >= 0).all()
    assert (rfm["Frequency"] >= 1).all()


def test_score_rfm_bins_are_1_to_5():
    df = make_dummy_sales()
    rfm = compute_rfm_table(df, "OrderDate")
    scored = score_rfm(rfm)
    for col in ["R_score", "F_score", "M_score"]:
        assert scored[col].between(1, 5).all()
    assert (scored["RFM_score"] == scored[["R_score", "F_score", "M_score"]].sum(axis=1)).all()


def test_recency_score_inverted():
    """Lower recency (more recent order) should map to a HIGHER R_score."""
    df = make_dummy_sales()
    rfm = compute_rfm_table(df, "OrderDate")
    scored = score_rfm(rfm)
    most_recent_customer = scored["Recency"].idxmin()
    least_recent_customer = scored["Recency"].idxmax()
    assert scored.loc[most_recent_customer, "R_score"] >= scored.loc[least_recent_customer, "R_score"]


def test_cluster_customers_assigns_all_rows():
    df = make_dummy_sales()
    rfm = score_rfm(compute_rfm_table(df, "OrderDate"))
    clustered = cluster_customers(rfm, n_clusters=3, random_state=42)
    assert "segment" in clustered.columns
    assert clustered["segment"].notna().all()
    assert clustered["cluster"].nunique() <= 3


def test_build_cohort_table_first_period_is_full_cohort():
    df = make_dummy_sales()
    cohort_pivot = build_cohort_table(df, "OrderDate")
    # period 0 for each cohort should equal that cohort's total customer count
    assert (cohort_pivot[0] > 0).all()


def test_retention_matrix_period_zero_is_100_pct():
    df = make_dummy_sales()
    cohort_pivot = build_cohort_table(df, "OrderDate")
    retention = retention_matrix(cohort_pivot)
    assert (retention[0] == 100.0).all()


def test_retention_matrix_never_exceeds_100():
    df = make_dummy_sales()
    cohort_pivot = build_cohort_table(df, "OrderDate")
    retention = retention_matrix(cohort_pivot)
    assert (retention.fillna(0) <= 100.0).all().all()


def test_monthly_revenue_trend_matches_total():
    df = make_dummy_sales()
    monthly = monthly_revenue_trend(df, "OrderDate")
    assert abs(monthly["Revenue"].sum() - df["Sales"].sum()) < 1e-6


def test_top_n_breakdown_revenue_share_sums_le_100():
    df = make_dummy_sales()
    top = top_n_breakdown(df, "Region", n=10)
    assert top["Revenue_share_pct"].sum() <= 100.0 + 1e-6


def test_order_level_summary_customer_count():
    df = make_dummy_sales()
    summary = order_level_summary(df, "OrderDate")
    assert summary["n_customers"] == df["CustomerID"].nunique()
    assert summary["n_orders"] == df["OrderID"].nunique()
