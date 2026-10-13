"""Momentum, reconciliation, projection and order-timing analytics (pure pandas).

Two order measures are used throughout:
  * announced MW   - orders from the yearly sheets (press releases and trade press), dated by announcement;
  * segment orders - Siemens Gamesa's reported order intake in EUR m per fiscal quarter (Siemens Energy earnings releases).
Siemens Gamesa announces only a small part of its orders, and the reported series is in EUR, not MW, so the two are
compared through an indicative "EUR per announced MW" ratio rather than reconciled one-to-one.
The fiscal year runs October-September (Q1 = Oct-Dec).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .common import (
    GIGA_MW,
    QUARTER_END_WINDOW_DAYS,
    as_of_date,
    pct_change,
    same_day,
)


MIN_RATIO_MW = 500.0  # below this many announced MW the EUR-per-announced-MW ratio is meaningless


# --------------------------------------------------------------------------------------
# Segment results (reported, EUR m, fiscal quarters)
# --------------------------------------------------------------------------------------
def segment_table(results: dict[str, Any] | None) -> pd.DataFrame:
    """Reported quarterly results with ratios and rolling four-quarter sums."""
    if not results:
        return pd.DataFrame()
    q = results["quarterly"].copy()
    q["book_to_bill"] = q["orders"] / q["revenue"]
    q["margin_before_si"] = q["profit_before_si"] / q["revenue"]
    q["service_share"] = q["revenue_service"] / q["revenue"]
    for col in ("orders", "revenue", "profit_before_si", "revenue_service"):
        q[f"ltm_{col}"] = q[col].rolling(4).sum()
    q["ltm_book_to_bill"] = q["ltm_orders"] / q["ltm_revenue"]
    q["ltm_margin_before_si"] = q["ltm_profit_before_si"] / q["ltm_revenue"]
    q["orders_yoy"] = q["orders"].pct_change(4, fill_method=None)
    q["revenue_yoy"] = q["revenue"].pct_change(4, fill_method=None)
    q["service_yoy"] = q["revenue_service"].pct_change(4, fill_method=None)
    q["cover_years"] = q["backlog_bn"] * 1000.0 / q["ltm_revenue"]
    return q


def fiscal_year_table(seg: pd.DataFrame) -> pd.DataFrame:
    """Fiscal-year sums of the reported quarters (partial years flagged)."""
    if seg.empty:
        return pd.DataFrame()
    g = seg.groupby("fy")
    t = pd.DataFrame(
        {
            "quarters": g.size(),
            "orders": g["orders"].sum(),
            "revenue": g["revenue"].sum(),
            "profit_before_si": g["profit_before_si"].sum(),
            "revenue_service": g["revenue_service"].sum(),
            "revenue_new_units": g["revenue_new_units"].sum(),
        }
    )
    t["complete"] = t["quarters"] == 4
    t["book_to_bill"] = t["orders"] / t["revenue"]
    t["margin_before_si"] = t["profit_before_si"] / t["revenue"]
    t["service_share"] = t["revenue_service"] / t["revenue"]
    t.index.name = "fy"
    return t


def to_date_table(seg: pd.DataFrame, n_quarters: int) -> pd.DataFrame:
    """Orders, revenue and profit for the first ``n_quarters`` of each fiscal year with that many quarters reported."""
    if seg.empty:
        return pd.DataFrame()
    s = seg[seg["quarter"] <= n_quarters]
    g = s.groupby("fy")
    t = pd.DataFrame({"quarters": g.size(), "orders": g["orders"].sum(), "revenue": g["revenue"].sum(), "profit_before_si": g["profit_before_si"].sum()})
    t = t[t["quarters"] == n_quarters].copy()
    for col in ("orders", "revenue"):
        t[f"{col}_yoy"] = t[col].pct_change(fill_method=None)
    t["book_to_bill"] = t["orders"] / t["revenue"]
    t["rank_orders"] = t["orders"].rank(ascending=False, method="min")
    return t


# --------------------------------------------------------------------------------------
# Announced orders versus reported order intake
# --------------------------------------------------------------------------------------
def announced_by_fiscal_quarter(orders: pd.DataFrame, seg: pd.DataFrame) -> pd.DataFrame:
    """Announced MW (total / offshore / onshore) per fiscal quarter next to reported orders in EUR."""
    if seg.empty:
        return pd.DataFrame()
    d = orders[orders["date_known"]]
    key = ["fiscal_year", "fiscal_quarter"]
    tot = d.groupby(key)["size_mw"].sum()
    off = d[d["segment"] == "Offshore"].groupby(key)["size_mw"].sum()
    on = d[d["segment"] == "Onshore"].groupby(key)["size_mw"].sum()
    cnt = d.groupby(key).size()
    out = seg[["fy", "quarter", "label", "orders", "revenue"]].copy()
    idx = list(zip(out["fy"], out["quarter"]))
    out["announced_mw"] = [float(tot.get(k, 0.0)) for k in idx]
    out["announced_offshore_mw"] = [float(off.get(k, 0.0)) for k in idx]
    out["announced_onshore_mw"] = [float(on.get(k, 0.0)) for k in idx]
    out["announced_orders"] = [int(cnt.get(k, 0)) for k in idx]
    out["eur_per_announced_mw"] = (out["orders"] / out["announced_mw"]).where(out["announced_mw"] >= MIN_RATIO_MW)
    return out


def announced_by_fiscal_year(orders: pd.DataFrame, fy_table: pd.DataFrame) -> pd.DataFrame:
    d = orders[orders["date_known"]]
    t = d.groupby("fiscal_year").agg(announced_mw=("size_mw", "sum"), announced_orders=("size_mw", "size"))
    t["offshore_mw"] = d[d["segment"] == "Offshore"].groupby("fiscal_year")["size_mw"].sum().reindex(t.index).fillna(0.0)
    t["onshore_mw"] = d[d["segment"] == "Onshore"].groupby("fiscal_year")["size_mw"].sum().reindex(t.index).fillna(0.0)
    t.index = t.index.astype(int)
    t.index.name = "fy"
    if not fy_table.empty:
        t = t.join(fy_table[["orders", "quarters"]].rename(columns={"orders": "reported_orders_eur"}), how="left")
        t["eur_per_announced_mw"] = (t["reported_orders_eur"] / t["announced_mw"]).where(t["announced_mw"] >= MIN_RATIO_MW)
    return t


# --------------------------------------------------------------------------------------
# Momentum: YTD / LTM on announced orders (calendar year)
# --------------------------------------------------------------------------------------
def ytd_table(orders: pd.DataFrame, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """For each calendar year: MW announced up to the as-of month/day (YTD), with and without jumbo orders, plus the full year."""
    as_of = as_of or as_of_date(orders)
    rows = []
    for year in range(int(orders["order_year"].min()), as_of.year + 1):
        cut = same_day(year, as_of)
        y = orders[orders["order_year"] == year]
        ytd = y[y["order_date"] <= cut]
        rows.append(
            {
                "year": year,
                "ytd_mw": float(ytd["size_mw"].sum()),
                "ytd_orders": int(len(ytd)),
                "ytd_mw_ex_giga": float(ytd.loc[ytd["size_mw"] < GIGA_MW, "size_mw"].sum()),
                "ytd_giga_orders": int((ytd["size_mw"] >= GIGA_MW).sum()),
                "full_mw": float(y["size_mw"].sum()) if year < as_of.year else np.nan,
                "full_orders": int(len(y)) if year < as_of.year else np.nan,
            }
        )
    t = pd.DataFrame(rows).set_index("year")
    t["ytd_share_of_full"] = t["ytd_mw"] / t["full_mw"]
    t["ytd_rank"] = t["ytd_mw"].rank(ascending=False, method="min")
    return t


def ltm_summary(orders: pd.DataFrame, seg: pd.DataFrame | None = None, as_of: pd.Timestamp | None = None) -> dict[str, Any]:
    """Last twelve months versus the twelve before (announced), plus the latest four reported quarters versus the four before."""
    as_of = as_of or as_of_date(orders)
    start = as_of - pd.DateOffset(years=1)
    prior_start = as_of - pd.DateOffset(years=2)
    cur = orders[(orders["order_date"] > start) & (orders["order_date"] <= as_of)]
    prev = orders[(orders["order_date"] > prior_start) & (orders["order_date"] <= start)]

    def stats(frame: pd.DataFrame) -> dict[str, float]:
        return {
            "mw": float(frame["size_mw"].sum()),
            "orders": int(len(frame)),
            "avg_mw": float(frame["size_mw"].mean()) if len(frame) else np.nan,
            "median_mw": float(frame["size_mw"].median()) if len(frame) else np.nan,
            "mw_ex_giga": float(frame.loc[frame["size_mw"] < GIGA_MW, "size_mw"].sum()),
            "giga_orders": int((frame["size_mw"] >= GIGA_MW).sum()),
            "offshore_mw": float(frame.loc[frame["segment"] == "Offshore", "size_mw"].sum()),
            "onshore_mw": float(frame.loc[frame["segment"] == "Onshore", "size_mw"].sum()),
            "countries": int(frame.loc[~frame["multi_country"], "country_clean"].nunique()),
        }

    c, p = stats(cur), stats(prev)
    out: dict[str, Any] = {
        "as_of": as_of,
        "window": (start, as_of),
        "current": c,
        "prior": p,
        "mw_growth": pct_change(c["mw"], p["mw"]),
        "orders_growth": pct_change(c["orders"], p["orders"]),
        "reported": None,
    }
    if seg is not None and len(seg) >= 8:
        last = seg.iloc[-1]
        prior = seg.iloc[-8:-4]
        cur4 = seg.iloc[-4:]
        out["reported"] = {
            "orders": float(cur4["orders"].sum()),
            "prior_orders": float(prior["orders"].sum()),
            "orders_growth": pct_change(cur4["orders"].sum(), prior["orders"].sum()),
            "revenue": float(cur4["revenue"].sum()),
            "prior_revenue": float(prior["revenue"].sum()),
            "revenue_growth": pct_change(cur4["revenue"].sum(), prior["revenue"].sum()),
            "profit_before_si": float(cur4["profit_before_si"].sum()),
            "prior_profit_before_si": float(prior["profit_before_si"].sum()),
            "book_to_bill": float(cur4["orders"].sum() / cur4["revenue"].sum()),
            "first_label": str(cur4.iloc[0]["label"]),
            "last_label": str(last["label"]),
        }
    return out


def cumulative_curves(orders: pd.DataFrame, as_of: pd.Timestamp | None = None, first_year: int = 2017) -> pd.DataFrame:
    """Daily cumulative announced MW per calendar year, on a common (leap-year) calendar axis."""
    as_of = as_of or as_of_date(orders)
    frames = []
    for year in range(first_year, as_of.year + 1):
        end = as_of if year == as_of.year else pd.Timestamp(year=year, month=12, day=31)
        days = pd.date_range(pd.Timestamp(year=year, month=1, day=1), end, freq="D")
        daily = (
            orders[(orders["order_year"] == year) & orders["date_known"]]
            .groupby(orders["order_date"].dt.normalize())["size_mw"]
            .sum()
            .reindex(days, fill_value=0.0)
        )
        frame = pd.DataFrame({"year": year, "date": days, "cum_mw": daily.cumsum().to_numpy()})
        frame["plot_date"] = frame["date"].map(lambda d: d.replace(year=2024))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------------------
# Fiscal-year scenarios
# --------------------------------------------------------------------------------------
def project_fiscal_year(results: dict[str, Any] | None, seg: pd.DataFrame, lookback: int = 8) -> dict[str, Any]:
    """Scenarios for the running fiscal year.

    * Orders are lumpy (one offshore project is worth several quarters), so the remaining quarters are scenarios built from the
      lowest, median and highest quarters of the last ``lookback`` reported quarters - not a forecast.
    * Revenue: what the remaining quarter(s) must deliver for the guided comparable growth (applied to last year's nominal revenue; indicative).
    * Profit: what the remaining quarter(s) must deliver for the guided margin (break-even).
    """
    if seg.empty:
        return {}
    last = seg.iloc[-1]
    fy, n = int(last["fy"]), int(last["quarter"])
    cur = seg[seg["fy"] == fy]
    prev = seg[seg["fy"] == fy - 1]
    remaining = 4 - n
    if remaining <= 0 or len(prev) < 4:
        return {}
    guid = (results or {}).get("guidance", {})
    hist = seg.iloc[-lookback:]["orders"]
    q_low, q_base, q_high = float(hist.min()), float(hist.median()), float(hist.max())
    ytd_orders, ytd_rev, ytd_pbsi = float(cur["orders"].sum()), float(cur["revenue"].sum()), float(cur["profit_before_si"].sum())
    prev_full = {"orders": float(prev["orders"].sum()), "revenue": float(prev["revenue"].sum()), "profit_before_si": float(prev["profit_before_si"].sum())}
    prev_ytd = prev[prev["quarter"] <= n]
    out: dict[str, Any] = {
        "fy": fy,
        "quarters_reported": n,
        "remaining": remaining,
        "orders_ytd": ytd_orders,
        "orders_prev_ytd": float(prev_ytd["orders"].sum()),
        "orders_prev_full": prev_full["orders"],
        "orders_scenario": {
            "conservative": ytd_orders + remaining * q_low,
            "base": ytd_orders + remaining * q_base,
            "strong": ytd_orders + remaining * q_high,
        },
        "quarter_orders": {"low": q_low, "median": q_base, "high": q_high, "lookback": int(len(hist))},
        "revenue_ytd": ytd_rev,
        "revenue_prev_ytd": float(prev_ytd["revenue"].sum()),
        "revenue_prev_full": prev_full["revenue"],
        "pbsi_ytd": ytd_pbsi,
        "pbsi_prev_ytd": float(prev_ytd["profit_before_si"].sum()),
        "last_quarter_revenue": float(last["revenue"]),
        "last_quarter_pbsi": float(last["profit_before_si"]),
        "prev_year_remaining_revenue": float(prev[prev["quarter"] > n]["revenue"].sum()),
        "guidance": guid,
    }
    g = guid.get("revenue_growth_comparable")
    if g:
        lo, hi = prev_full["revenue"] * (1 + g[0]), prev_full["revenue"] * (1 + g[1])
        out["revenue_full_year_guided"] = (lo, hi)
        out["revenue_needed_remaining"] = (lo - ytd_rev, hi - ytd_rev)
    margin = guid.get("profit_margin_before_si")
    if margin is not None:
        full_rev = out.get("revenue_full_year_guided", (prev_full["revenue"], prev_full["revenue"]))
        out["pbsi_needed_remaining"] = tuple(float(margin * r - ytd_pbsi) for r in full_rev)
    return out


# --------------------------------------------------------------------------------------
# Timing / seasonality
# --------------------------------------------------------------------------------------
def quarter_share_table(orders: pd.DataFrame, first_year: int = 2017) -> pd.DataFrame:
    d = orders[orders["date_known"] & (orders["order_year"] >= first_year)]
    mw = d.pivot_table(index="order_year", columns="order_quarter", values="size_mw", aggfunc="sum").fillna(0.0)
    mw = mw.reindex(columns=[1, 2, 3, 4], fill_value=0.0)
    share = mw.div(mw.sum(axis=1).replace(0, np.nan), axis=0)
    share.columns = [f"Q{c}" for c in share.columns]
    return share


def quarter_end_share_by_year(orders: pd.DataFrame) -> pd.DataFrame:
    """Share of each year's announced MW and orders booked in the last N days of a quarter."""
    d = orders[orders["date_known"]]
    g = d.groupby("order_year")
    mw_total = g["size_mw"].sum()
    mw_qe = d[d["in_quarter_end_window"]].groupby("order_year")["size_mw"].sum().reindex(mw_total.index).fillna(0.0)
    n_total = g.size()
    n_qe = d[d["in_quarter_end_window"]].groupby("order_year").size().reindex(n_total.index).fillna(0)
    out = pd.DataFrame({"mw_share": mw_qe / mw_total, "orders_share": n_qe / n_total})
    out.index.name = "year"
    return out


def quarter_end_overall(orders: pd.DataFrame, first_year: int = 2017) -> dict[str, float]:
    o = orders[orders["date_known"] & (orders["order_year"] >= first_year)]
    expected = QUARTER_END_WINDOW_DAYS / 91.25
    q_start = o["order_date"].dt.to_period("Q").dt.start_time
    days_in = (o["order_date"].dt.normalize() - q_start).dt.days
    early = days_in < QUARTER_END_WINDOW_DAYS
    return {
        "mw_share": float(o.loc[o["in_quarter_end_window"], "size_mw"].sum() / o["size_mw"].sum()),
        "orders_share": float(o["in_quarter_end_window"].mean()),
        "early_mw_share": float(o.loc[early, "size_mw"].sum() / o["size_mw"].sum()),
        "expected_if_uniform": float(expected),
        "window_days": QUARTER_END_WINDOW_DAYS,
        "first_year": first_year,
        "years": int(o["order_year"].nunique()),
        "orders": int(len(o)),
    }


def intra_quarter_profile(orders: pd.DataFrame, as_of: pd.Timestamp | None = None, first_year: int = 2017) -> pd.DataFrame:
    """Average cumulative share of a quarter's MW by day of quarter (complete quarters with orders only)."""
    as_of = as_of or as_of_date(orders)
    grid = np.arange(0, 92)
    curves = []
    sub = orders[orders["date_known"] & (orders["order_year"] >= first_year)]
    for (year, quarter), grp in sub.groupby(["order_year", "order_quarter"]):
        q_start = pd.Timestamp(year=int(year), month=3 * int(quarter) - 2, day=1)
        q_end = q_start + pd.offsets.QuarterEnd(0)
        if q_end >= as_of or grp["size_mw"].sum() <= 0:
            continue
        day_idx = (grp["order_date"].dt.normalize() - q_start).dt.days.clip(0, 91)
        daily = np.zeros(92)
        np.add.at(daily, day_idx.to_numpy(), grp["size_mw"].to_numpy())
        curves.append(np.cumsum(daily) / daily.sum())
    if not curves:
        return pd.DataFrame(columns=["day", "mean", "p25", "p75", "uniform", "quarters"])
    arr = np.vstack(curves)
    return pd.DataFrame(
        {
            "day": grid + 1,
            "mean": arr.mean(axis=0),
            "p25": np.percentile(arr, 25, axis=0),
            "p75": np.percentile(arr, 75, axis=0),
            "uniform": (grid + 1) / 92.0,
            "quarters": len(curves),
        }
    )


def month_in_quarter_share(orders: pd.DataFrame, first_year: int = 2017) -> pd.Series:
    o = orders[orders["date_known"] & (orders["order_year"] >= first_year)]
    pos = ((o["order_date"].dt.month - 1) % 3) + 1
    mw = o.groupby(pos)["size_mw"].sum()
    return (mw / mw.sum()).rename("share")
