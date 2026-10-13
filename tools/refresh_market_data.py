"""Refresh the monthly steel (HRC=F) and copper (HG=F) JSON from Yahoo Finance's chart endpoint.

Usage: python tools/refresh_market_data.py
Existing months are kept; the most recent months are overwritten with fresh values. The Siemens Gamesa share price is not
refreshed: the company was delisted in 2023 and the stock file ends there.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import requests

DATA = Path(__file__).resolve().parent.parent / "data"
HEADERS = {"User-Agent": "Mozilla/5.0"}


def fetch_close(symbol: str) -> dict[str, float]:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    r = requests.get(url, params={"range": "3y", "interval": "1mo"}, headers=HEADERS, timeout=30)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    offset = res["meta"].get("gmtoffset", 0)  # timestamps are exchange-local midnight
    closes = res["indicators"]["quote"][0]["close"]
    out = {}
    for ts, close in zip(res["timestamp"], closes):
        if close is not None:
            out[datetime.fromtimestamp(ts + offset, tz=timezone.utc).strftime("%Y-%m-01")] = round(close, 4)
    return out


def main() -> None:
    path = DATA / "market_prices_monthly.json"
    rows = {r["date"]: r for r in json.loads(path.read_text(encoding="utf-8"))} if path.exists() else {}
    for sym, key in (("HRC=F", "steel_hrc_usd_per_short_ton"), ("HG=F", "copper_hg_usd_per_lb")):
        for date, close in fetch_close(sym).items():
            rows.setdefault(date, {"date": date})[key] = close
    path.write_text(json.dumps([rows[k] for k in sorted(rows)], indent=2), encoding="utf-8")
    print(f"market: last month {max(rows)}", rows[max(rows)])


if __name__ == "__main__":
    main()
