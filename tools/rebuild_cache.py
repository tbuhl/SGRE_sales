"""Rebuild data_cache/sgre_parsed_data.pkl from the workbook (default: SG_data_new.xlsx).

Usage: python tools/rebuild_cache.py [workbook.xlsx]
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

import app  # noqa: E402


def main() -> None:
    wb = Path(sys.argv[1]) if len(sys.argv) > 1 else app.find_default_workbook()
    if wb is None:
        raise SystemExit("no workbook found")
    economy = app.parse_economy_sheet(wb)
    economy_fy = app.parse_economy_fy_table(wb)
    quarterly = app.parse_quarterly_onshore_cost_per_mw(wb)
    orders, platforms = app.parse_oi_sheets(wb)
    st = wb.stat()
    payload = {
        "cache_key": f"{wb.resolve()}::{st.st_mtime_ns}::{st.st_size}",
        "workbook_path": str(wb.resolve()),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "economy": economy, "economy_fy": economy_fy, "quarterly_onshore_cost": quarterly,
        "orders": orders, "platforms": platforms,
    }
    out = ROOT / "data_cache" / "sgre_parsed_data.pkl"
    out.parent.mkdir(exist_ok=True)
    pd.to_pickle(payload, out)
    print(f"{wb.name}: {len(orders)} orders, {len(platforms)} platform rows -> {out}")
    print(orders.groupby("order_year").agg(n=("size_mw", "size"), mw=("size_mw", "sum")).to_string())


if __name__ == "__main__":
    main()
