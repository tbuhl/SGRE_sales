"""Shared fixtures: the parsed Siemens Gamesa dataset (rebuilt from the workbook if the cache is missing)."""
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from intel import common  # noqa: E402
from intel.bundle import build_intel  # noqa: E402


@pytest.fixture(scope="session")
def raw():
    cache = ROOT / "data_cache" / "sgre_parsed_data.pkl"
    if not cache.exists():  # data_cache is not committed; parse the workbook with the app's own parser
        import app  # noqa: PLC0415

        previous = os.getcwd()
        os.chdir(ROOT)
        try:
            wb = app.find_default_workbook()
            economy_fy = app.parse_economy_fy_table(wb)
            orders, platforms = app.parse_oi_sheets(wb)
        finally:
            os.chdir(previous)
        return {"economy_fy": economy_fy, "orders": orders, "platforms": platforms}
    with open(cache, "rb") as fh:
        return pickle.load(fh)


@pytest.fixture(scope="session")
def orders(raw):
    return common.prepare_orders(raw["orders"])


@pytest.fixture(scope="session")
def platforms(raw):
    return common.prepare_platforms(raw["platforms"])


@pytest.fixture(scope="session")
def results():
    previous = os.getcwd()
    os.chdir(ROOT)  # the loader uses a path relative to the working directory
    try:
        return common.load_segment_results()
    finally:
        os.chdir(previous)


@pytest.fixture(scope="session")
def intel(raw, results):
    market = pd.read_json(ROOT / "data" / "market_prices_monthly.json")
    stock = pd.read_json(ROOT / "data" / "sgre_stock_monthly.json")
    return build_intel(raw["economy_fy"], raw["orders"], raw["platforms"], stock, market, results)
