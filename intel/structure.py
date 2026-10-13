"""Market-structure analytics: concentration, product mix, geography, accounts, service, compare lab."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .common import (
    GIGA_MW,
    MEGA_MW,
    RATING_BUCKET_ORDER,
    SERIES_ORDER,
    UNKNOWN,
    as_of_date,
    pct_change,
    weighted_mean,
)


# --------------------------------------------------------------------------------------
# Concentration / lumpiness
# --------------------------------------------------------------------------------------
def concentration_by_year(orders: pd.DataFrame, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    as_of = as_of or as_of_date(orders)
    rows = []
    for year, g in orders.groupby("order_year"):
        s = g["size_mw"].sort_values(ascending=False)
        total = float(s.sum())
        country = g[~g["multi_country"]].groupby("country_clean")["size_mw"].sum()
        cust = g[g["customer"] != UNKNOWN].groupby("customer")["size_mw"].sum()
        c_share = country / country.sum() if country.sum() > 0 else country
        hhi = float((c_share**2).sum() * 10000) if len(c_share) else np.nan
        rows.append(
            {
                "year": int(year),
                "orders": int(len(g)),
                "mw": total,
                "median_mw": float(s.median()),
                "mean_mw": float(s.mean()),
                "top1_share": float(s.iloc[0] / total) if total else np.nan,
                "top5_share": float(s.head(5).sum() / total) if total else np.nan,
                "top10_share": float(s.head(10).sum() / total) if total else np.nan,
                "mega_orders": int((s >= MEGA_MW).sum()),
                "mega_share": float(s[s >= MEGA_MW].sum() / total) if total else np.nan,
                "giga_orders": int((s >= GIGA_MW).sum()),
                "n_countries": int(len(country)),
                "top_country": country.idxmax() if len(country) else None,
                "top_country_share": float(c_share.max()) if len(c_share) else np.nan,
                "country_hhi": hhi,
                "effective_countries": float(10000 / hhi) if hhi and not np.isnan(hhi) else np.nan,
                "customer_disclosed_share": float(cust.sum() / total) if total else np.nan,
                "customer_top10_share": float(cust.nlargest(10).sum() / cust.sum()) if cust.sum() > 0 else np.nan,
                "partial": bool(year == as_of.year),
            }
        )
    return pd.DataFrame(rows).set_index("year")


def largest_orders(orders: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    cols = ["order_date", "order_year", "country_clean", "customer", "size_mw", "primary_platform", "segment"]
    top = orders.nlargest(n, "size_mw")[cols].copy()
    top["order_date"] = [d.date().isoformat() if pd.notna(d) else f"{int(y)} (no exact date)" for d, y in zip(top["order_date"], top["order_year"])]
    top = top.drop(columns="order_year")
    top.columns = ["Order date", "Country", "Customer", "MW", "Primary platform", "Segment"]
    return top.reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Product mix
# --------------------------------------------------------------------------------------
def mix_by_year(platforms: pd.DataFrame) -> pd.DataFrame:
    """MW-weighted rating/rotor and offshore share per order year (platform-slot basis)."""
    p = platforms[platforms["slot_mw"].fillna(0) > 0]
    rows = []
    for year, g in p.groupby("order_year"):
        known = g[g["segment"] != UNKNOWN]
        mapped = float(known["slot_mw"].sum())
        off = float(known.loc[known["segment"] == "Offshore", "slot_mw"].sum())
        rows.append(
            {
                "year": int(year),
                "rating_mw": weighted_mean(g["mw_rating"], g["slot_mw"]),
                "rotor_m": weighted_mean(g["rotor_m"], g["slot_mw"]),
                "platform_mapped_mw": mapped,
                "offshore_mw": off,
                "offshore_share": off / mapped if mapped > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("year")


def series_mix(platforms: pd.DataFrame) -> pd.DataFrame:
    """Share of platform-mapped MW by product series and order year."""
    p = platforms[(platforms["slot_mw"].fillna(0) > 0) & (platforms["series"] != UNKNOWN)]
    mw = p.pivot_table(index="order_year", columns="series", values="slot_mw", aggfunc="sum").fillna(0.0)
    mw = mw.reindex(columns=[c for c in SERIES_ORDER if c in mw.columns])
    return mw.div(mw.sum(axis=1).replace(0, np.nan), axis=0)


def platform_mw(platforms: pd.DataFrame, year: int | None = None, top: int = 12) -> pd.Series:
    """Platform-slot MW by exact platform (optionally one order year), largest first."""
    p = platforms[(platforms["slot_mw"].fillna(0) > 0) & (platforms["family"] != UNKNOWN)]
    if year is not None:
        p = p[p["order_year"] == year]
    return p.groupby("family")["slot_mw"].sum().sort_values(ascending=False).head(top)


def rating_bucket_mix(platforms: pd.DataFrame) -> pd.DataFrame:
    """Share of platform-mapped MW by rating bucket and order year (columns in size order)."""
    p = platforms[(platforms["slot_mw"].fillna(0) > 0) & (platforms["rating_bucket"] != UNKNOWN)]
    mw = p.pivot_table(index="order_year", columns="rating_bucket", values="slot_mw", aggfunc="sum").fillna(0.0)
    mw = mw.reindex(columns=RATING_BUCKET_ORDER, fill_value=0.0)
    return mw.div(mw.sum(axis=1).replace(0, np.nan), axis=0)


def leading_platforms(platforms: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    """Top platforms per order year with their share of platform-mapped MW."""
    p = platforms[(platforms["slot_mw"].fillna(0) > 0) & (platforms["family"] != UNKNOWN)]
    rows = []
    for year, g in p.groupby("order_year"):
        by = g.groupby("family")["slot_mw"].sum().sort_values(ascending=False)
        total = by.sum()
        for rank, (name, mw) in enumerate(by.head(top_n).items(), start=1):
            rows.append({"year": int(year), "rank": rank, "platform": name, "mw": float(mw), "share": float(mw / total)})
    return pd.DataFrame(rows)


def cagr(series: pd.Series, start: int, end: int) -> float:
    if start not in series.index or end not in series.index:
        return float("nan")
    a, b = series[start], series[end]
    if pd.isna(a) or pd.isna(b) or a <= 0 or end <= start:
        return float("nan")
    return float((b / a) ** (1.0 / (end - start)) - 1.0)


# --------------------------------------------------------------------------------------
# Geography
# --------------------------------------------------------------------------------------
def period_compare(orders: pd.DataFrame, cur: tuple[pd.Timestamp, pd.Timestamp], prior: tuple[pd.Timestamp, pd.Timestamp]) -> pd.DataFrame:
    """Country MW in two (start, end] windows. Multi-country labels are excluded."""
    base = orders[~orders["multi_country"]]

    def window(frame: pd.DataFrame, win: tuple[pd.Timestamp, pd.Timestamp]) -> pd.Series:
        sel = frame[(frame["order_date"] > win[0]) & (frame["order_date"] <= win[1])]
        return sel.groupby("country_clean")["size_mw"].sum()

    a, b = window(base, cur), window(base, prior)
    out = pd.DataFrame({"mw_current": a, "mw_prior": b}).fillna(0.0)
    out["delta_mw"] = out["mw_current"] - out["mw_prior"]
    out["share_current"] = out["mw_current"] / out["mw_current"].sum() if out["mw_current"].sum() else np.nan
    out["share_prior"] = out["mw_prior"] / out["mw_prior"].sum() if out["mw_prior"].sum() else np.nan
    out["share_shift_pp"] = (out["share_current"] - out["share_prior"]) * 100
    out.index.name = "country"
    return out.sort_values("delta_mw")


def continent_mix(orders: pd.DataFrame) -> pd.DataFrame:
    mw = orders.pivot_table(index="order_year", columns="continent", values="size_mw", aggfunc="sum").fillna(0.0)
    return mw.div(mw.sum(axis=1).replace(0, np.nan), axis=0)


def market_status(orders: pd.DataFrame, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """Per country: first/last order, lifetime and recent MW, and a lifecycle status."""
    as_of = as_of or as_of_date(orders)
    base = orders[~orders["multi_country"]]
    ltm_start = as_of - pd.DateOffset(years=1)
    rows = []
    for country, g in base.groupby("country_clean"):
        last = g["date_eff"].max()
        first_year = int(g["order_year"].min())
        months_since = (as_of - last).days / 30.44
        if first_year >= as_of.year - 5:
            status = "Emerging"
        elif months_since <= 12:
            status = "Active"
        elif months_since <= 36:
            status = "Cooling"
        else:
            status = "Dormant"
        rows.append(
            {
                "country": country,
                "status": status,
                "first_order_year": first_year,
                "last_order": last.date(),
                "months_since_last_order": months_since,
                "orders": int(len(g)),
                "lifetime_mw": float(g["size_mw"].sum()),
                "ltm_mw": float(g.loc[g["date_eff"] > ltm_start, "size_mw"].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("lifetime_mw", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Accounts
# --------------------------------------------------------------------------------------
def account_radar(orders: pd.DataFrame, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """Recency / frequency / size per disclosed customer.

    Status is recency-based: Active (ordered in the last 12 months), Cooling (12-24 months), Dormant (> 24 months).
    Orders without an exact date (2020-2023 sheets) are placed in the middle of their order year.
    ``cadence_ratio`` compares days since the last order with the account's own median gap between order dates
    (> 2 means well past its usual rhythm); it is informational because clustered small orders distort the median.
    """
    as_of = as_of or as_of_date(orders)
    k = orders[orders["customer"] != UNKNOWN]
    ltm_start = as_of - pd.DateOffset(years=1)
    prior_start = as_of - pd.DateOffset(years=2)
    rows = []
    for customer, g in k.groupby("customer"):
        dates = np.sort(g["date_eff"].dt.normalize().unique())
        gaps = np.diff(dates).astype("timedelta64[D]").astype(int) if len(dates) > 1 else np.array([], dtype=int)
        median_gap = float(np.median(gaps)) if len(gaps) else np.nan
        last = pd.Timestamp(dates[-1])
        days_since = int((as_of - last).days)
        status = "Active" if days_since <= 365 else ("Cooling" if days_since <= 730 else "Dormant")
        by_country = g.groupby("country_clean")["size_mw"].sum()
        ltm = float(g.loc[g["date_eff"] > ltm_start, "size_mw"].sum())
        prior = float(g.loc[(g["date_eff"] > prior_start) & (g["date_eff"] <= ltm_start), "size_mw"].sum())
        rows.append(
            {
                "customer": customer,
                "status": status,
                "repeat_buyer": bool(len(dates) >= 2),
                "orders": int(len(g)),
                "order_dates": int(len(dates)),
                "lifetime_mw": float(g["size_mw"].sum()),
                "avg_order_mw": float(g["size_mw"].mean()),
                "first_order": pd.Timestamp(dates[0]).date(),
                "last_order": last.date(),
                "days_since_last": days_since,
                "median_gap_days": median_gap,
                "cadence_ratio": days_since / median_gap if median_gap and median_gap > 0 else np.nan,
                "ltm_mw": ltm,
                "prior_ltm_mw": prior,
                "ltm_delta_mw": ltm - prior,
                "top_country": by_country.idxmax(),
                "countries": int(by_country.size),
            }
        )
    return pd.DataFrame(rows).sort_values("lifetime_mw", ascending=False).reset_index(drop=True)


def account_status_summary(radar: pd.DataFrame, min_lifetime_mw: float = 300.0, repeat_only: bool = True) -> pd.DataFrame:
    big = radar[radar["lifetime_mw"] >= min_lifetime_mw]
    if repeat_only:
        big = big[big["repeat_buyer"]]
    summary = big.groupby("status").agg(accounts=("customer", "count"), lifetime_mw=("lifetime_mw", "sum"))
    return summary.reindex([s for s in ["Active", "Cooling", "Dormant"] if s in summary.index])


# --------------------------------------------------------------------------------------
# Service
# --------------------------------------------------------------------------------------
def service_by_year(orders: pd.DataFrame) -> pd.DataFrame:
    """Service contract length on new orders (the workbook holds no service scheme for Siemens Gamesa)."""
    rows = []
    for year, g in orders.groupby("order_year"):
        total = float(g["size_mw"].sum())
        tenure = g.dropna(subset=["service_time_years"])
        rows.append(
            {
                "year": int(year),
                "mw": total,
                "tenure_disclosed_share": float(tenure["size_mw"].sum() / total) if total else np.nan,
                "tenure_wavg_years": weighted_mean(tenure["service_time_years"], tenure["size_mw"]),
                "tenure_median_years": float(tenure["service_time_years"].median()) if len(tenure) else np.nan,
                "contracted_mw_years": float((tenure["size_mw"] * tenure["service_time_years"]).sum()),
            }
        )
    return pd.DataFrame(rows).set_index("year")


def lead_time_by_year(orders: pd.DataFrame) -> pd.DataFrame:
    """Order-to-delivery lead time (days) by order year; negative values are treated as data errors."""
    d = orders[(orders["delivery_days"] >= 0) & orders["delivery_days"].notna()]
    g = d.groupby("order_year")["delivery_days"]
    out = pd.DataFrame({"median_days": g.median(), "p90_days": g.quantile(0.9), "orders": g.size()})
    out.index.name = "year"
    return out


# --------------------------------------------------------------------------------------
# Compare lab
# --------------------------------------------------------------------------------------
COMPARE_DIMENSIONS = ["Country", "Continent", "Customer", "Platform family"]


def entity_orders(orders: pd.DataFrame, platforms: pd.DataFrame, dim: str, value: str) -> pd.DataFrame:
    """Orders belonging to one entity with an `entity_mw` column (platform slot MW for platform families)."""
    if dim == "Country":
        sub = orders[orders["country_clean"] == value].copy()
        sub["entity_mw"] = sub["size_mw"]
    elif dim == "Continent":
        sub = orders[orders["continent"] == value].copy()
        sub["entity_mw"] = sub["size_mw"]
    elif dim == "Customer":
        sub = orders[orders["customer"] == value].copy()
        sub["entity_mw"] = sub["size_mw"]
    elif dim == "Platform family":
        slots = platforms[(platforms["family"] == value) & (platforms["slot_mw"].fillna(0) > 0)]
        mw = slots.groupby("order_id")["slot_mw"].sum().rename("entity_mw")
        sub = orders.join(mw, on="order_id", how="inner")
    else:
        raise ValueError(f"Unknown dimension: {dim}")
    return sub


def entity_options(orders: pd.DataFrame, platforms: pd.DataFrame, dim: str) -> list[str]:
    if dim == "Country":
        s = orders[~orders["multi_country"]].groupby("country_clean")["size_mw"].sum()
    elif dim == "Continent":
        s = orders[orders["continent"] != UNKNOWN].groupby("continent")["size_mw"].sum()
    elif dim == "Customer":
        s = orders[orders["customer"] != UNKNOWN].groupby("customer")["size_mw"].sum()
    else:
        s = platforms[platforms["family"] != UNKNOWN].groupby("family")["slot_mw"].sum()
    return s.sort_values(ascending=False).index.tolist()


def entity_profile(
    orders: pd.DataFrame, platforms: pd.DataFrame, dim: str, value: str, as_of: pd.Timestamp | None = None
) -> dict[str, Any]:
    as_of = as_of or as_of_date(orders)
    sub = entity_orders(orders, platforms, dim, value)
    ltm_start = as_of - pd.DateOffset(years=1)
    prior_start = as_of - pd.DateOffset(years=2)
    ltm = float(sub.loc[sub["date_eff"] > ltm_start, "entity_mw"].sum())
    prior = float(sub.loc[(sub["date_eff"] > prior_start) & (sub["date_eff"] <= ltm_start), "entity_mw"].sum())
    ids = sub["order_id"]
    plat = platforms[platforms["order_id"].isin(ids)]
    if dim == "Platform family":
        plat = plat[plat["family"] == value]
    tenure = sub.dropna(subset=["service_time_years"])
    total_basis = platforms["slot_mw"].sum() if dim == "Platform family" else orders["size_mw"].sum()
    return {
        "entity": value,
        "orders": int(len(sub)),
        "mw": float(sub["entity_mw"].sum()),
        "share_of_total": float(sub["entity_mw"].sum() / total_basis) if total_basis else np.nan,
        "avg_order_mw": float(sub["entity_mw"].mean()) if len(sub) else np.nan,
        "largest_order_mw": float(sub["entity_mw"].max()) if len(sub) else np.nan,
        "first_order": sub["date_eff"].min().date() if len(sub) else None,
        "last_order": sub["date_eff"].max().date() if len(sub) else None,
        "ltm_mw": ltm,
        "prior_ltm_mw": prior,
        "ltm_growth": pct_change(ltm, prior),
        "avg_rating_mw": weighted_mean(plat["mw_rating"], plat["slot_mw"]),
        "avg_rotor_m": weighted_mean(plat["rotor_m"], plat["slot_mw"]),
        "service_tenure_years": weighted_mean(tenure["service_time_years"], tenure["entity_mw"]),
        "offshore_share": float(sub.loc[sub["segment"] == "Offshore", "entity_mw"].sum() / sub["entity_mw"].sum())
        if len(sub) and sub["entity_mw"].sum() > 0
        else np.nan,
        "median_delivery_days": float(sub["delivery_days"].median()) if sub["delivery_days"].notna().any() else np.nan,
        "active_years": int(sub["order_year"].nunique()),
        "countries": int(sub.loc[~sub["multi_country"], "country_clean"].nunique()),
        "customers": int(sub.loc[sub["customer"] != UNKNOWN, "customer"].nunique()),
    }


def entity_yearly(orders: pd.DataFrame, platforms: pd.DataFrame, dim: str, value: str) -> pd.Series:
    sub = entity_orders(orders, platforms, dim, value)
    years = range(int(orders["order_year"].min()), int(orders["order_year"].max()) + 1)
    return sub.groupby("order_year")["entity_mw"].sum().reindex(years, fill_value=0.0).rename(value)


def entity_breakdown(orders: pd.DataFrame, platforms: pd.DataFrame, dim: str, value: str, by: str, top_n: int = 8) -> pd.Series:
    """MW of an entity split by another dimension (country / customer / platform family / rating bucket)."""
    sub = entity_orders(orders, platforms, dim, value)
    if by == "Platform family":
        p = platforms[platforms["order_id"].isin(sub["order_id"]) & (platforms["slot_mw"].fillna(0) > 0)]
        if dim == "Platform family":
            p = p[p["family"] == value]
        s = p.groupby("family")["slot_mw"].sum()
        s = s[s.index != UNKNOWN]
    elif by == "Country":
        s = sub[~sub["multi_country"]].groupby("country_clean")["entity_mw"].sum()
    elif by == "Customer":
        s = sub[sub["customer"] != UNKNOWN].groupby("customer")["entity_mw"].sum()
    else:
        raise ValueError(by)
    s = s.sort_values(ascending=False)
    if len(s) > top_n:
        s = pd.concat([s.head(top_n), pd.Series({"Other": s.iloc[top_n:].sum()})])
    return s
