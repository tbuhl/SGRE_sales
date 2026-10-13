# Siemens Gamesa Sales Intelligence Dashboard

Interactive Streamlit dashboard for Siemens Gamesa (SGRE) commercial and economics analytics, built on public data.

## What This App Includes
- **Executive Briefing**: reported orders, book-to-bill, profit before special items, backlog and service revenue from Siemens Energy's quarterly
  releases, a fiscal-year scenario for orders against the company guidance, and eleven auto-generated, number-backed findings.
- **Intelligence Lab** (choose a section at the top):
  - `Momentum & Outlook`: reported results by fiscal quarter, outlook working (orders scenario, revenue and break-even needed for guidance),
    announced orders against reported intake, announcement timing, fiscal-year economics.
  - `Market Structure` (respects the sidebar filters): order-size lumpiness, product series (offshore direct drive, geared, SG 7.0, 4.X-6.X), geography.
  - `Accounts & Service`: account radar and re-engagement watchlist, service contract lengths.
  - `Compare Lab`: head-to-head profile of two countries, continents, customers or platform families.
  - `Data Quality`: checks with severity, field completeness by year, record-level detail, definitions.
- The original tabs: economy, year-by-year, quarterly, across years, platform, country, delivery, correlations, news, patents, information.

## Data Files Used By The App
- `SG_data_new.xlsx` - source workbook (economy and fiscal-year tables, `2017`-`2026` order sheets). The app parses it and caches the result in `data_cache/` (not committed).
- `data/sgre_segment_results.json` - Siemens Gamesa segment results by fiscal quarter (orders, revenue, profit before special items, new-unit / service
  revenue, backlog) taken from the Siemens Energy earnings releases, with the fiscal-year guidance.
- `data/sgre_stock_monthly.json` - share-price history; ends at the 2023 delisting.
- `data/market_prices_monthly.json` - monthly steel (HRC=F) and copper (HG=F).

## Run Locally
```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Method Notes
- **Fiscal year** runs October-September (FY2026 = Oct 2025 - Sep 2026; Q1 = Oct-Dec).
- **Reported vs announced**: reported orders are in EUR and cover every order; announced MW is only what Siemens Gamesa, customers or the trade press disclose,
  dated by announcement (which trails the booking). Reported orders have been about EUR 3m per announced MW in FY24-FY25 - a yardstick, not a reconciliation.
- **Scenarios** (not forecasts): remaining-quarter orders use the lowest, median and highest of the last 8 reported quarters; revenue and profit show what the
  remaining quarter must deliver for the guided comparable growth and break-even.
- **Order list is thin since 2023** (3-5 announced orders a year); reported EUR series are the dependable base for trends.
- **Offshore / onshore** is derived from the platform name; orders without a platform stay unclassified. SG14-236 and SG14-236DD are one platform.
- **Customers** are undisclosed for almost all MW in 2017-2019, and "New" / "First" entries are treated as undisclosed.

## Project Layout (intelligence layer)
```
intel/common.py      constants, fiscal calendar, platform / series classification, segment-results loader
intel/outlook.py     segment table, fiscal-year tables, announced vs reported, fiscal-year scenario, announcement timing
intel/structure.py   concentration, product mix, geography, accounts, service, compare lab
intel/economics.py   fiscal-year KPIs (book-to-bill, service share, margins, backlog cover)
intel/quality.py     completeness and anomaly checks
intel/insights.py    rule-based findings and watchlist
intel/bundle.py      computes everything once (cached by the app)
intel/charts_*.py    Plotly figures; intel/theme.py palette and chart chrome; intel/ui.py page layouts
tools/refresh_market_data.py   refreshes steel / copper monthly JSON from Yahoo Finance
tools/rebuild_cache.py         rebuilds data_cache/sgre_parsed_data.pkl from the workbook
tests/                         pytest suite (python -m pytest tests)
```

## Updating Data
1. Add new orders to the current-year sheet of `SG_data_new.xlsx` (and the fiscal-year table when a year closes).
2. Add the new quarter to `data/sgre_segment_results.json` from the Siemens Energy earnings release (Siemens Gamesa segment table) and update the guidance.
3. `python tools/refresh_market_data.py`, then `python -m pytest tests`.
4. Commit the workbook and JSON files and deploy.

## Tests
```powershell
python -m pip install pytest
python -m pytest tests
```
The tests check each analytic against an independent brute-force calculation, and the segment results against the workbook's fiscal-year totals.

## Deploy (Streamlit Community Cloud)
1. Push the repository to GitHub.
2. Create a new app from the repo with `app.py` as the main file.

## Troubleshooting
- If visuals look stale after deployment, do a hard refresh in the browser.
- If market overlays are missing, check that the JSON files exist in `data/`.
- If `streamlit` is not found, run `python -m streamlit run app.py`.
