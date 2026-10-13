# Sales Analytics

An end-to-end sales analytics project: KPI reporting, RFM customer segmentation, cohort retention analysis, and short-horizon sales forecasting, built on transactional order data.

## Highlights

- Config-driven pipeline (`config.yaml`) — no hardcoded paths or parameters in code
- **KPI analysis:** revenue trends, top products/categories/regions, order-level metrics
- **RFM segmentation:** Recency-Frequency-Monetary scoring + KMeans clustering into named customer segments (Champions, Loyal, At Risk, Lost, etc.)
- **Cohort retention analysis:** monthly acquisition cohorts and the retention matrix showing how each cohort's activity decays over time
- **Forecasting:** Holt-Winters exponential smoothing for short-horizon monthly revenue forecasts, with a naive-baseline comparison so the forecast has to earn its keep
- Synthetic data generator included — the project runs immediately without needing an external dataset
- Interactive Streamlit dashboard tying together KPIs, segments, cohorts, and forecast
- Unit tests for the RFM and cohort logic
- GitHub Actions CI

## Dataset

The project works with order-line-level transactional data with this schema:

| Column | Description |
|---|---|
| `OrderID` | Unique order identifier |
| `OrderDate` | Date of the order |
| `CustomerID` | Unique customer identifier |
| `Product` | Product name |
| `Category` | Product category |
| `Region` | Sales region |
| `Quantity` | Units sold |
| `UnitPrice` | Price per unit |
| `Discount` | Discount fraction applied (0–1) |
| `Sales` | Net revenue for the line (`Quantity * UnitPrice * (1 - Discount)`) |

Two ways to get data:

1. **Your own data:** export transactions in the schema above to `data/sales.csv`. This maps directly onto datasets like the classic "Superstore" sales dataset if you already have that.
2. **Synthetic fallback:** `python src/generate_synthetic_data.py` generates ~3 years of realistic transactions with seasonality, customer churn, and repeat-purchase behavior baked in, so every downstream analysis has something real to show.

## Project structure

```
sales-analytics/
├── config.yaml                  # paths, RFM/segment/forecast parameters
├── requirements.txt
├── data/                        # sales.csv goes here (gitignored)
├── reports/                     # generated CSVs + plots (gitignored)
├── notebooks/
│   └── 01_eda.ipynb             # exploratory analysis
├── src/
│   ├── generate_synthetic_data.py
│   ├── data_loader.py
│   ├── kpi_analysis.py          # revenue trends, top-N breakdowns
│   ├── rfm_segmentation.py      # RFM scoring + KMeans segments
│   ├── cohort_analysis.py       # monthly cohort retention matrix
│   └── forecasting.py           # Holt-Winters revenue forecast
├── app.py                       # Streamlit dashboard
├── tests/
│   └── test_analytics.py
└── .github/workflows/ci.yml
```

## Setup

```bash
git clone https://github.com/<your-username>/sales-analytics.git
cd sales-analytics
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
# 1. Get data (pick one)
python src/generate_synthetic_data.py          # synthetic, works offline
# or place your own sales.csv into data/, matching the schema above

# 2. Run the full analysis (KPIs, RFM, cohorts, forecast) and save reports
PYTHONPATH=src python src/kpi_analysis.py --config config.yaml
PYTHONPATH=src python src/rfm_segmentation.py --config config.yaml
PYTHONPATH=src python src/cohort_analysis.py --config config.yaml
PYTHONPATH=src python src/forecasting.py --config config.yaml

# 3. Launch the interactive dashboard
streamlit run app.py
```

## Approach

- **RFM segmentation:** each customer is scored on Recency (days since last order, lower is better), Frequency (order count), and Monetary (total spend), quintile-binned 1–5, then clustered with KMeans on the three scores. Clusters are labeled by their centroid profile (e.g., high F+M and low R → "Champions") rather than left as opaque cluster numbers — segment labels should mean something to a stakeholder, not just to the algorithm.
- **Cohort retention:** customers are grouped by the month of their first purchase; the retention matrix shows, for each cohort, what fraction of its customers were still active in each subsequent month. This surfaces retention/churn patterns that a single "average retention rate" number would hide.
- **Forecasting:** monthly revenue is forecast with Holt-Winters exponential smoothing (captures trend + seasonality without the setup overhead of a full ARIMA grid search). A naive seasonal baseline ("this month = same month last year") is reported alongside it — a forecasting model that doesn't beat that baseline isn't earning its complexity.

## Limitations & next steps

- The synthetic generator produces plausible transactional patterns but is not real market data — segment sizes and forecast accuracy on it are illustrative, not benchmarks to cite.
- Forecasting is univariate (revenue over time only); it doesn't account for planned promotions, price changes, or external demand shocks.
- RFM segment counts (via KMeans) depend on the `n_clusters` choice in `config.yaml` — the default is a reasonable starting point, not a tuned optimum for any particular business.
- ## 🚀 Live Demo

**[Open Sales Analytics Dashboard]((https://sales-analytics-mbyeokseaww4s4yzhvzcet.streamlit.app/)

Explore interactive sales KPIs, customer segmentation, cohort retention analysis, and revenue forecasting.


## License

MIT — see [LICENSE](LICENSE).
