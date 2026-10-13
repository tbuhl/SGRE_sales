"""Shared constants and helpers for the intelligence layer (pure pandas, no Streamlit)."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MEGA_MW = 1000.0  # orders at/above this size are "mega" orders (mostly offshore projects)
GIGA_MW = 2000.0  # orders at/above this size are "jumbo" orders (kept under the shared name GIGA_MW)
QUARTER_END_WINDOW_DAYS = 14  # window used for the quarter-end clustering metric

# Labels that are not a single country (kept visible, but excluded from country-level analysis).
MULTI_COUNTRY_LABELS = {"Unknown"}
# Variants that belong to a single country.
COUNTRY_FIXES = {"United Kingdom (England)": "United Kingdom"}
# Customer entries that are truncated or placeholder text in the workbook ("New", "First"); treated as undisclosed.
CUSTOMER_PLACEHOLDERS = {"new", "first"}

RATING_BUCKET_ORDER = ["<2.5 MW", "2.5-3.9 MW", "4.0-5.9 MW", "6.0-9.9 MW", "10+ MW"]
UNKNOWN = "Unknown"

# Product series, classified from the platform name (indicative).
SERIES_ORDER = [
    "Offshore direct drive (8-15 MW)",
    "Offshore geared (SWT 3.6-7 MW)",
    "SG 7.0 (restarted onshore)",
    "SG 4.X-6.X onshore",
    "SG 3.X onshore",
    "SG 2.X / SWT onshore",
]
OFFSHORE_SERIES = {"Offshore direct drive (8-15 MW)", "Offshore geared (SWT 3.6-7 MW)"}

FISCAL_YEAR_START_MONTH = 10  # Siemens Energy / Siemens Gamesa fiscal year runs October-September


def reported_results_path() -> Path:
    return Path("data") / "sgre_segment_results.json"


def same_day(year: int, ref: pd.Timestamp) -> pd.Timestamp:
    """The ref month/day in another year (Feb-29 falls back to Feb-28)."""
    try:
        return pd.Timestamp(year=year, month=ref.month, day=ref.day)
    except ValueError:
        return pd.Timestamp(year=year, month=ref.month, day=28)


def fiscal_year(ts: Any) -> float:
    """Fiscal year of a date (FY2026 = 1 Oct 2025 - 30 Sep 2026)."""
    if ts is None or pd.isna(ts):
        return float("nan")
    ts = pd.Timestamp(ts)
    return float(ts.year + 1 if ts.month >= FISCAL_YEAR_START_MONTH else ts.year)


def fiscal_quarter(ts: Any) -> float:
    if ts is None or pd.isna(ts):
        return float("nan")
    return float(((pd.Timestamp(ts).month - FISCAL_YEAR_START_MONTH) % 12) // 3 + 1)


def fiscal_quarter_end(fy: int, q: int) -> pd.Timestamp:
    """Calendar end date of a fiscal quarter (Q1 FY2026 ends 31 Dec 2025)."""
    start = pd.Timestamp(year=fy - 1, month=FISCAL_YEAR_START_MONTH, day=1)
    return start + pd.DateOffset(months=3 * q) - pd.Timedelta(days=1)


def rating_bucket(value: Any) -> str:
    if value is None or pd.isna(value):
        return UNKNOWN
    v = float(value)
    if v < 2.5:
        return RATING_BUCKET_ORDER[0]
    if v < 4.0:
        return RATING_BUCKET_ORDER[1]
    if v < 6.0:
        return RATING_BUCKET_ORDER[2]
    if v < 10.0:
        return RATING_BUCKET_ORDER[3]
    return RATING_BUCKET_ORDER[4]


def platform_family(name: Any) -> str:
    """'SG14-236DD' -> 'SG 14-236'; 'SG5.0-145' -> 'SG 5.0-145'; 'SWT-7.0-154' -> 'SWT 7.0-154'; labels without a model -> 'Unknown'."""
    if name is None or pd.isna(name):
        return UNKNOWN
    text = str(name).upper().replace(" ", "")
    if text == "UNKNOWN":
        return UNKNOWN
    m = re.match(r"^(SG|SWT)-?(?:DD-)?([0-9]+(?:\.[0-9X]+)?(?:-[0-9]+)?)", text)
    if not m:
        return UNKNOWN
    prefix, rest = m.groups()
    return f"{prefix} {rest}"


def product_series(platform: Any, rating: Any = None) -> str:
    """Product series from the platform name."""
    text = "" if platform is None or pd.isna(platform) else str(platform).upper().replace(" ", "")
    if not text or text == "UNKNOWN":
        return UNKNOWN
    m = re.match(r"^(SG|SWT)-?(?:DD-)?([0-9]+(?:\.[0-9]+)?)", text)
    if m:
        prefix, r = m.group(1), float(m.group(2))
    else:
        prefix, r = "", (float(rating) if rating is not None and not pd.isna(rating) else float("nan"))
    if "DD" in text or (prefix == "SG" and r >= 8.0):
        return SERIES_ORDER[0]
    if prefix == "SWT" and (r >= 6.0 or text.startswith("SWT-3.6-130")):
        return SERIES_ORDER[1]
    if prefix == "SG" and text.startswith("SG7.0-170"):
        return SERIES_ORDER[2]
    if prefix == "SG" and r >= 4.0:
        return SERIES_ORDER[3]
    if prefix == "SG" and r >= 3.0:
        return SERIES_ORDER[4]
    if prefix in ("SG", "SWT"):
        return SERIES_ORDER[5]
    return UNKNOWN


def segment_of(series: Any) -> str:
    if series is None or series == UNKNOWN:
        return UNKNOWN
    return "Offshore" if series in OFFSHORE_SERIES else "Onshore"


def prepare_orders(orders: pd.DataFrame) -> pd.DataFrame:
    """Add the derived columns the analytics need (does not modify the input)."""
    o = orders.copy()
    o["country_clean"] = o["country"].replace(COUNTRY_FIXES)
    o["multi_country"] = o["country_clean"].isin(MULTI_COUNTRY_LABELS)
    o["customer_raw"] = o["customer"]
    o["customer"] = o["customer"].where(~o["customer"].astype(str).str.lower().isin(CUSTOMER_PLACEHOLDERS), UNKNOWN)
    o["date_known"] = o["order_date"].notna()
    mid_year = pd.to_datetime(o["order_year"].astype(int).astype(str) + "-07-01")
    o["date_eff"] = o["order_date"].where(o["date_known"], mid_year)
    quarter_end = o["order_date"] + pd.offsets.QuarterEnd(0)
    o["days_to_quarter_end"] = (quarter_end - o["order_date"]).dt.days
    o["in_quarter_end_window"] = (o["days_to_quarter_end"] < QUARTER_END_WINDOW_DAYS) & o["date_known"]
    o["order_quarter_label"] = o["order_year"].astype(str) + "-Q" + o["order_quarter"].astype("Int64").astype(str)
    o["fiscal_year"] = o["order_date"].map(fiscal_year)
    o["fiscal_quarter"] = o["order_date"].map(fiscal_quarter)
    o["series"] = o["primary_platform"].map(product_series)
    o["segment"] = o["series"].map(segment_of)
    o["is_mega"] = o["size_mw"] >= MEGA_MW
    o["is_giga"] = o["size_mw"] >= GIGA_MW
    return o


def prepare_platforms(platforms: pd.DataFrame) -> pd.DataFrame:
    p = platforms.copy()
    p["family"] = p["platform"].map(platform_family)
    p["series"] = p["platform"].map(product_series)
    p["segment"] = p["series"].map(segment_of)
    p["rating_bucket"] = p["mw_rating"].map(rating_bucket)
    return p


def as_of_date(orders: pd.DataFrame) -> pd.Timestamp:
    return pd.Timestamp(orders["order_date"].max()).normalize()


def _norm_metric(name: Any) -> str:
    text = str(name).replace("–", "-").replace("—", "-").replace("�", "-")
    return re.sub(r"\s+", " ", text.strip().lower())


def eco_wide(economy: pd.DataFrame) -> pd.DataFrame:
    """Economy as year x metric with normalised (lower-case, ASCII dash) metric names."""
    if economy is None or economy.empty:
        return pd.DataFrame()
    e = economy.copy()
    e["key"] = e["metric"].map(_norm_metric)
    return e.pivot_table(index="year", columns="key", values="value", aggfunc="first").sort_index()


def eco_series(wide: pd.DataFrame, *candidates: str) -> pd.Series:
    """First matching metric as a Series indexed by year (empty Series when none match)."""
    for cand in candidates:
        key = _norm_metric(cand)
        if key in wide.columns:
            return wide[key].dropna()
    return pd.Series(dtype=float)


def load_segment_results(path: Path | None = None) -> dict[str, Any] | None:
    """Siemens Gamesa segment results by fiscal quarter (EUR m) from the Siemens Energy earnings releases."""
    path = path or reported_results_path()
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    q = pd.DataFrame(raw.get("quarterly", []))
    if q.empty:
        return None
    q = q.sort_values(["fy", "quarter"]).reset_index(drop=True)
    q["period_end"] = [fiscal_quarter_end(int(a), int(b)) for a, b in zip(q["fy"], q["quarter"])]
    q["label"] = "Q" + q["quarter"].astype(int).astype(str) + " FY" + (q["fy"].astype(int) % 100).astype(str).str.zfill(2)
    return {
        "quarterly": q,
        "guidance": raw.get("guidance", {}),
        "orders_checked_through": raw.get("orders_checked_through"),
        "orders_check_note": raw.get("orders_check_note", ""),
        "source": raw.get("source", ""),
        "retrieved": raw.get("retrieved", ""),
        "notes": raw.get("notes", ""),
    }


def safe_div(a: float, b: float) -> float:
    if b is None or a is None or pd.isna(a) or pd.isna(b) or b == 0:
        return float("nan")
    return float(a) / float(b)


def pct_change(new: float, old: float) -> float:
    """(new/old - 1) as a fraction; NaN when the base is missing or zero."""
    if old is None or pd.isna(old) or old == 0 or new is None or pd.isna(new):
        return float("nan")
    return float(new) / float(old) - 1.0


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return float("nan")
    return float(np.average(values[mask], weights=weights[mask]))
