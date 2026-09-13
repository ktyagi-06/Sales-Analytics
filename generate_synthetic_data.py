"""
Generate synthetic order-line-level transactional sales data with:
  - yearly seasonality (a Nov/Dec peak, a summer lull)
  - a growing customer base over time, with realistic churn
  - repeat-purchase behavior so RFM/cohort analysis has real structure
  - a handful of categories/regions/products with different price points

This is NOT real market data -- it's built to have enough structure
(seasonality, churn, repeat purchases) that the analytics pipeline has
something real to find, not a substitute for an actual sales dataset.

Usage:
    python src/generate_synthetic_data.py --n-customers 800 --years 3
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CATEGORIES = {
    "Electronics": ["Wireless Mouse", "Bluetooth Speaker", "USB-C Hub", "Webcam", "Monitor Stand"],
    "Office Supplies": ["Notebook Set", "Desk Organizer", "Stapler", "Pen Pack", "Sticky Notes"],
    "Furniture": ["Office Chair", "Standing Desk", "Bookshelf", "Filing Cabinet", "Desk Lamp"],
    "Home & Kitchen": ["Coffee Maker", "Blender", "Cutting Board Set", "Storage Bins", "Water Bottle"],
}
PRICE_RANGES = {
    "Electronics": (15, 250),
    "Office Supplies": (2, 30),
    "Furniture": (60, 500),
    "Home & Kitchen": (10, 120),
}
REGIONS = ["North", "South", "East", "West"]


def seasonal_weight(month: int) -> float:
    """Relative order volume multiplier by calendar month (Nov/Dec peak, summer lull)."""
    weights = {1: 0.85, 2: 0.85, 3: 0.9, 4: 0.95, 5: 1.0, 6: 0.9,
               7: 0.85, 8: 0.9, 9: 1.0, 10: 1.1, 11: 1.4, 12: 1.6}
    return weights[month]


def generate_synthetic_sales(n_customers: int, years: int, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    # End on the last day of the previous full calendar month so every
    # month in the generated series is complete -- an in-progress current
    # month would otherwise look like a revenue cliff to any monthly
    # aggregation (KPI trend, forecasting) done downstream.
    this_month_start = pd.Timestamp.today().normalize().replace(day=1)
    end_date = this_month_start - pd.Timedelta(days=1)
    start_date = (end_date - pd.DateOffset(years=years)).replace(day=1)
    total_days = (end_date - start_date).days

    # Stagger customer acquisition over time (growing customer base), and
    # assign each customer a "loyalty" score controlling purchase frequency
    # and an eventual churn point so cohorts show realistic decay.
    acquisition_day = np.sort(rng.integers(0, int(total_days * 0.85), size=n_customers))
    loyalty = rng.beta(2, 5, size=n_customers)  # most customers low-to-moderate loyalty, a tail of high loyalty
    region_assignment = rng.choice(REGIONS, size=n_customers)

    rows = []
    order_id = 100000

    for cust_idx in range(n_customers):
        acq_day = acquisition_day[cust_idx]
        cust_id = f"CUST-{cust_idx + 1:05d}"
        region = region_assignment[cust_idx]

        # Expected number of orders over the customer's lifetime, driven by loyalty.
        expected_orders = 1 + loyalty[cust_idx] * 25
        n_orders = max(1, rng.poisson(expected_orders))

        # Customer becomes inactive ("churns") at some point between
        # acquisition and the end of the window; more loyal customers churn later.
        active_days_remaining = total_days - acq_day
        churn_point = acq_day + int(active_days_remaining * min(1.0, loyalty[cust_idx] + rng.uniform(0.1, 0.5)))
        churn_point = min(churn_point, total_days)

        if churn_point <= acq_day:
            order_days = [acq_day]
        else:
            order_days = sorted(rng.integers(acq_day, churn_point + 1, size=n_orders))

        for day_offset in order_days:
            order_date = start_date + pd.Timedelta(days=int(day_offset))
            month = order_date.month
            # Thin out orders in low-season months probabilistically.
            if rng.random() > seasonal_weight(month) / 1.6:
                continue

            n_lines = rng.integers(1, 4)  # 1-3 product lines per order
            for _ in range(n_lines):
                category = rng.choice(list(CATEGORIES.keys()))
                product = rng.choice(CATEGORIES[category])
                low, high = PRICE_RANGES[category]
                unit_price = round(rng.uniform(low, high), 2)
                quantity = rng.integers(1, 5)
                discount = rng.choice([0, 0, 0, 0.05, 0.10, 0.15, 0.20], p=[0.55, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05])
                sales = round(quantity * unit_price * (1 - discount), 2)

                rows.append({
                    "OrderID": f"ORD-{order_id}",
                    "OrderDate": order_date.strftime("%Y-%m-%d"),
                    "CustomerID": cust_id,
                    "Product": product,
                    "Category": category,
                    "Region": region,
                    "Quantity": int(quantity),
                    "UnitPrice": unit_price,
                    "Discount": discount,
                    "Sales": sales,
                })
            order_id += 1

    df = pd.DataFrame(rows)
    df = df.sort_values("OrderDate").reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic transactional sales data.")
    parser.add_argument("--n-customers", type=int, default=800)
    parser.add_argument("--years", type=int, default=3)
    parser.add_argument("--output", type=str, default="data/sales.csv")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    df = generate_synthetic_sales(args.n_customers, args.years, args.random_state)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Generated {len(df):,} order lines across {df['OrderID'].nunique():,} orders "
          f"and {df['CustomerID'].nunique():,} customers -> {output_path}")
    print(f"Date range: {df['OrderDate'].min()} to {df['OrderDate'].max()}")
    print(f"Total revenue: ${df['Sales'].sum():,.2f}")


if __name__ == "__main__":
    main()
