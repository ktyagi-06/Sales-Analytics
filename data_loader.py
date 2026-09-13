"""
Config loading and raw sales data loading utilities.
"""
from pathlib import Path
import yaml
import pandas as pd


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_sales_data(config: dict) -> pd.DataFrame:
    raw_path = Path(config["data"]["raw_path"])
    if not raw_path.exists():
        raise FileNotFoundError(
            f"No data found at {raw_path}. Either place your own sales export "
            f"there (see README for the expected schema), or run "
            f"`python src/generate_synthetic_data.py --output {raw_path}` "
            f"to create a synthetic dataset."
        )
    df = pd.read_csv(raw_path)

    date_col = config["data"]["date_column"]
    required_cols = {date_col, "CustomerID", "OrderID", "Sales"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {raw_path}: {missing}")

    df[date_col] = pd.to_datetime(df[date_col])
    return df
