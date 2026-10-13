"""Tests for the intelligence layer.

Each analytic is checked against an independent brute-force calculation on the dataset, so the tests keep
passing when the data is refreshed. Tests tagged "data sanity" assert properties of the data, not of the code.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from intel import economics, insights, outlook, quality, structure
from intel.common import (
    GIGA_MW,
    MEGA_MW,
    UNKNOWN,
    as_of_date,
    fiscal_quarter,
    fiscal_quarter_end,
    fiscal_year,
    platform_family,
    product_series,
    rating_bucket,
    same_day,
    segment_of,
)


# ---------------------------------------------------------------- helpers
def test_platform_family_series_and_segments():
    assert platform_family("SG14-236DD") == "SG 14-236" == platform_family("SG14-236")
    assert platform_family("SWT-7.0-154") == "SWT 7.0-154" and platform_family("SG2.X") == "SG 2.X"
    assert platform_family("Unknown") == UNKNOWN and platform_family(None) == UNKNOWN
    assert product_series("SG14-236DD") == product_series("SG11-200DD") == "Offshore direct drive (8-15 MW)"
    assert product_series("SWT-6.0-154") == "Offshore geared (SWT 3.6-7 MW)" and product_series("SWT-2.3-108") == "SG 2.X / SWT onshore"
    assert product_series("SG7.0-170") == "SG 7.0 (restarted onshore)" and product_series("SG6.6-170") == "SG 4.X-6.X onshore"
    assert product_series("SG3.4-132") == "SG 3.X onshore" and product_series("Unknown") == UNKNOWN
    assert segment_of("Offshore geared (SWT 3.6-7 MW)") == "Offshore" and segment_of("SG 3.X onshore") == "Onshore" and segment_of(UNKNOWN) == UNKNOWN
    assert rating_bucket(2.0) == "<2.5 MW" and rating_bucket(14) == "10+ MW" and rating_bucket(np.nan) == UNKNOWN


def test_fiscal_calendar():
    assert fiscal_year(pd.Timestamp("2025-09-30")) == 2025 and fiscal_year(pd.Timestamp("2025-10-01")) == 2026
    assert [fiscal_quarter(pd.Timestamp(f"2026-{m:02d}-15")) for m in (1, 3, 4, 6, 7, 9, 10, 12)] == [2, 2, 3, 3, 4, 4, 1, 1]
    assert fiscal_quarter_end(2026, 1) == pd.Timestamp("2025-12-31") and fiscal_quarter_end(2026, 4) == pd.Timestamp("2026-09-30")


def test_same_day_handles_leap_day():
    assert same_day(2023, pd.Timestamp("2024-02-29")) == pd.Timestamp("2023-02-28")


def test_prepare_orders_adds_columns(orders, raw):
    for col in ["country_clean", "multi_country", "date_known", "date_eff", "fiscal_year", "fiscal_quarter", "series", "segment", "is_mega", "customer_raw"]:
        assert col in orders
    assert not (orders["country_clean"] == "United Kingdom (England)").any()
    assert (orders["is_mega"] == (orders["size_mw"] >= MEGA_MW)).all()
    assert not orders["customer"].str.lower().isin(["new", "first"]).any()
    assert orders["customer_raw"].str.lower().isin(["new", "first"]).sum() == (raw["orders"]["customer"].str.lower().isin(["new", "first"])).sum()
    brute = orders["order_date"].map(lambda d: d.year + 1 if d.month >= 10 else d.year)
    assert (orders["fiscal_year"] == brute).all()
    assert orders["date_known"].all()  # data sanity: every Siemens Gamesa order carries an exact date


# ---------------------------------------------------------------- segment results
def test_segment_file_is_consistent(results):
    q = results["quarterly"]
    assert not q.duplicated(["fy", "quarter"]).any() and q["quarter"].between(1, 4).all()
    assert q["period_end"].is_monotonic_increasing
    assert (q["revenue_new_units"] + q["revenue_service"]).sub(q["revenue"]).abs().max() <= 3  # releases round each figure


def test_segment_matches_workbook_fiscal_years(results, raw):
    """data sanity: quarterly segment figures add up to the workbook's fiscal-year orders and revenue."""
    fy = outlook.fiscal_year_table(outlook.segment_table(results))
    wb = raw["economy_fy"].pivot_table(index="year", columns="metric", values="value")
    for year in fy.index[fy["complete"]]:
        assert fy.loc[year, "orders"] == pytest.approx(wb.loc[year, "Total order intake (€m)"], rel=0.005)
        assert fy.loc[year, "revenue"] == pytest.approx(wb.loc[year, "Total revenue (€m)"], rel=0.005)


def test_segment_table_definitions(results):
    s = outlook.segment_table(results)
    assert np.allclose(s["book_to_bill"], s["orders"] / s["revenue"])
    for col in ("orders", "revenue", "profit_before_si"):
        brute = s[col].rolling(4).sum()
        assert np.allclose(s[f"ltm_{col}"].dropna(), brute.dropna())
    assert s["ltm_orders"].isna().sum() == 3
    i = len(s) - 1
    assert s.loc[i, "orders_yoy"] == pytest.approx(s.loc[i, "orders"] / s.loc[i - 4, "orders"] - 1)
    assert s.loc[i, "cover_years"] == pytest.approx(s.loc[i, "backlog_bn"] * 1000 / s.loc[i, "ltm_revenue"])


def test_fiscal_year_and_to_date_tables(results):
    s = outlook.segment_table(results)
    fy = outlook.fiscal_year_table(s)
    last = int(fy.index.max())
    assert not fy.loc[last, "complete"] and fy.loc[last, "quarters"] == len(s[s["fy"] == last])
    assert fy.loc[last, "orders"] == pytest.approx(s[s["fy"] == last]["orders"].sum())
    n = int(s.iloc[-1]["quarter"])
    td = outlook.to_date_table(s, n)
    for year in td.index:
        sel = s[(s["fy"] == year) & (s["quarter"] <= n)]
        assert td.loc[year, "orders"] == pytest.approx(sel["orders"].sum()) and td.loc[year, "revenue"] == pytest.approx(sel["revenue"].sum())
    assert td["rank_orders"].min() == 1


def test_announced_by_fiscal_quarter_matches_bruteforce(orders, results):
    seg = outlook.segment_table(results)
    t = outlook.announced_by_fiscal_quarter(orders, seg)
    row = t[(t["fy"] == 2025) & (t["quarter"] == 4)].iloc[0]  # Jul-Sep 2025
    d = orders[(orders["order_date"] >= "2025-07-01") & (orders["order_date"] <= "2025-09-30")]
    assert row["announced_mw"] == pytest.approx(d["size_mw"].sum()) and row["announced_orders"] == len(d)
    assert row["announced_mw"] == pytest.approx(row["announced_offshore_mw"] + row["announced_onshore_mw"])
    assert t.loc[t["announced_mw"] < outlook.MIN_RATIO_MW, "eur_per_announced_mw"].isna().all()
    y = outlook.announced_by_fiscal_year(orders, outlook.fiscal_year_table(seg))
    assert y["announced_mw"].sum() == pytest.approx(orders["size_mw"].sum())


# ---------------------------------------------------------------- momentum
def test_ytd_matches_bruteforce(orders):
    as_of = as_of_date(orders)
    t = outlook.ytd_table(orders, as_of)
    for year in (as_of.year, as_of.year - 1, as_of.year - 3):
        cut = same_day(year, as_of)
        sel = orders[(orders["order_year"] == year) & (orders["order_date"] <= cut)]
        assert t.loc[year, "ytd_mw"] == pytest.approx(sel["size_mw"].sum())
        assert t.loc[year, "ytd_mw_ex_giga"] == pytest.approx(sel.loc[sel["size_mw"] < GIGA_MW, "size_mw"].sum())


def test_ltm_summary(orders, results):
    as_of = as_of_date(orders)
    seg = outlook.segment_table(results)
    s = outlook.ltm_summary(orders, seg, as_of)
    start = as_of - pd.DateOffset(years=1)
    cur = orders[(orders["order_date"] > start) & (orders["order_date"] <= as_of)]
    assert s["current"]["mw"] == pytest.approx(cur["size_mw"].sum()) and s["current"]["orders"] == len(cur)
    assert s["current"]["offshore_mw"] + s["current"]["onshore_mw"] <= s["current"]["mw"] + 1e-6
    r = s["reported"]
    assert r["orders"] == pytest.approx(seg["orders"].iloc[-4:].sum()) and r["prior_orders"] == pytest.approx(seg["orders"].iloc[-8:-4].sum())
    assert r["book_to_bill"] == pytest.approx(seg["orders"].iloc[-4:].sum() / seg["revenue"].iloc[-4:].sum())


def test_cumulative_curves_end_at_annual_totals(orders):
    as_of = as_of_date(orders)
    c = outlook.cumulative_curves(orders, as_of)
    for year in (2018, 2022, as_of.year):
        final = c[c["year"] == year]["cum_mw"].iloc[-1]
        cut = as_of if year == as_of.year else pd.Timestamp(year, 12, 31)
        assert final == pytest.approx(orders[(orders["order_year"] == year) & (orders["order_date"] <= cut)]["size_mw"].sum())
    assert c.groupby("year")["cum_mw"].apply(lambda s: s.is_monotonic_increasing).all()


# ---------------------------------------------------------------- projection
def test_projection_structure(results):
    seg = outlook.segment_table(results)
    pr = outlook.project_fiscal_year(results, seg, lookback=8)
    sc, qo = pr["orders_scenario"], pr["quarter_orders"]
    hist = seg["orders"].iloc[-8:]
    assert qo["low"] == hist.min() and qo["high"] == hist.max() and qo["median"] == hist.median()
    assert sc["conservative"] <= sc["base"] <= sc["strong"]
    assert sc["base"] == pytest.approx(pr["orders_ytd"] + pr["remaining"] * qo["median"])
    assert pr["orders_ytd"] == pytest.approx(seg[seg["fy"] == pr["fy"]]["orders"].sum())
    g = results["guidance"]["revenue_growth_comparable"]
    assert pr["revenue_full_year_guided"][0] == pytest.approx(pr["revenue_prev_full"] * (1 + g[0]))
    assert pr["revenue_needed_remaining"][0] == pytest.approx(pr["revenue_full_year_guided"][0] - pr["revenue_ytd"])
    assert pr["pbsi_needed_remaining"][0] == pytest.approx(-pr["pbsi_ytd"])  # break-even guidance
    assert pr["prev_year_remaining_revenue"] == pytest.approx(seg[(seg["fy"] == pr["fy"] - 1) & (seg["quarter"] > pr["quarters_reported"])]["revenue"].sum())


# ---------------------------------------------------------------- timing
def test_quarter_end_shares_are_fractions(orders):
    by = outlook.quarter_end_share_by_year(orders)
    assert by["mw_share"].between(0, 1).all() and by["orders_share"].between(0, 1).all()
    overall = outlook.quarter_end_overall(orders)
    recent = orders[orders["order_year"] >= 2017]
    assert overall["mw_share"] == pytest.approx(recent.loc[recent["days_to_quarter_end"] < 14, "size_mw"].sum() / recent["size_mw"].sum())
    start = recent["order_date"].dt.to_period("Q").dt.start_time
    early = recent[(recent["order_date"] - start).dt.days < 14]
    assert overall["early_mw_share"] == pytest.approx(early["size_mw"].sum() / recent["size_mw"].sum())


def test_intra_quarter_profile_is_a_cdf(orders):
    p = outlook.intra_quarter_profile(orders)
    assert p["mean"].is_monotonic_increasing and p["mean"].iloc[-1] == pytest.approx(1.0)
    assert (p["p25"] <= p["p75"] + 1e-12).all()


def test_quarter_and_month_shares_sum_to_one(orders):
    q = outlook.quarter_share_table(orders)
    assert np.allclose(q.sum(axis=1).dropna(), 1.0)
    assert outlook.month_in_quarter_share(orders).sum() == pytest.approx(1.0)


# ---------------------------------------------------------------- structure
def test_concentration_invariants(orders):
    c = structure.concentration_by_year(orders)
    assert ((c["top1_share"] <= c["top5_share"] + 1e-12) & (c["top5_share"] <= c["top10_share"] + 1e-12) & (c["top10_share"] <= 1 + 1e-12)).all()
    year = int(c.index[-2])
    assert c.loc[year, "orders"] == (orders["order_year"] == year).sum()
    assert c.loc[year, "mega_orders"] == ((orders["order_year"] == year) & (orders["size_mw"] >= MEGA_MW)).sum()


def test_period_compare_is_consistent(orders):
    as_of = as_of_date(orders)
    cur = (as_of - pd.DateOffset(years=1), as_of)
    prior = (as_of - pd.DateOffset(years=2), as_of - pd.DateOffset(years=1))
    p = structure.period_compare(orders, cur, prior)
    assert np.allclose(p["delta_mw"], p["mw_current"] - p["mw_prior"])
    assert "Unknown" not in set(p.index)
    brute = orders[(orders["order_date"] > cur[0]) & (orders["order_date"] <= cur[1]) & ~orders["multi_country"]]["size_mw"].sum()
    assert p["mw_current"].sum() == pytest.approx(brute)


def test_account_radar_rules(orders):
    as_of = as_of_date(orders)
    r = structure.account_radar(orders, as_of)
    named = orders[orders["customer"] != UNKNOWN]
    assert r["lifetime_mw"].sum() == pytest.approx(named["size_mw"].sum())
    assert (r.loc[r["days_since_last"] <= 365, "status"] == "Active").all()
    assert (r.loc[r["days_since_last"] > 730, "status"] == "Dormant").all()
    assert (r.loc[~r["repeat_buyer"], "order_dates"] == 1).all()
    assert "New" not in set(r["customer"]) and "First" not in set(r["customer"])


def test_market_status_covers_all_countries(orders):
    m = structure.market_status(orders)
    assert set(m["status"]) <= {"Active", "Cooling", "Dormant", "Emerging"}
    assert m["lifetime_mw"].sum() == pytest.approx(orders[~orders["multi_country"]]["size_mw"].sum())


def test_mix_and_series_shares(platforms):
    mix = structure.mix_by_year(platforms)
    assert mix["offshore_share"].dropna().between(0, 1).all()
    assert np.allclose(structure.rating_bucket_mix(platforms).sum(axis=1), 1.0)
    sm = structure.series_mix(platforms)
    assert np.allclose(sm.sum(axis=1), 1.0)
    year = int(mix.index[-2])
    known = platforms[(platforms["order_year"] == year) & (platforms["slot_mw"] > 0) & (platforms["segment"] != UNKNOWN)]
    assert mix.loc[year, "offshore_mw"] == pytest.approx(known.loc[known["segment"] == "Offshore", "slot_mw"].sum())
    pm = structure.platform_mw(platforms, year, top=50)
    assert "SG 14-236" in pm.index and "SG14-236" not in pm.index and "SG14-236DD" not in pm.index  # DD and non-DD names are one platform


def test_service_tenure_is_bounded(orders):
    s = structure.service_by_year(orders)
    assert s["tenure_disclosed_share"].dropna().between(0, 1).all()
    year = int(s.index[-2])
    y = orders[orders["order_year"] == year]
    assert s.loc[year, "tenure_disclosed_share"] == pytest.approx(y.loc[y["service_time_years"].notna(), "size_mw"].sum() / y["size_mw"].sum())


def test_entity_profile_matches_direct_sums(orders, platforms):
    prof = structure.entity_profile(orders, platforms, "Country", "Spain")
    sub = orders[orders["country_clean"] == "Spain"]
    assert prof["mw"] == pytest.approx(sub["size_mw"].sum()) and prof["orders"] == len(sub)
    assert 0 <= prof["offshore_share"] <= 1
    fam = structure.entity_profile(orders, platforms, "Platform family", "SG 14-236")
    assert fam["mw"] == pytest.approx(platforms.loc[platforms["family"] == "SG 14-236", "slot_mw"].sum())
    assert structure.entity_yearly(orders, platforms, "Country", "Spain").sum() == pytest.approx(prof["mw"])


# ---------------------------------------------------------------- economics
def test_economics_table_definitions(raw, results):
    seg = outlook.segment_table(results)
    fy = outlook.fiscal_year_table(seg)
    t = economics.economics_table(raw["economy_fy"], fy)
    y = 2025
    assert t.loc[y, "book_to_bill"] == pytest.approx(t.loc[y, "order_intake"] / t.loc[y, "revenue"])
    assert t.loc[y, "service_share"] == pytest.approx(t.loc[y, "revenue_service"] / t.loc[y, "revenue"])
    assert t.loc[y, "net_margin"] == pytest.approx(t.loc[y, "net_profit"] / t.loc[y, "revenue"])
    assert t.loc[y, "profit_before_si"] == pytest.approx(fy.loc[y, "profit_before_si"])
    assert t.loc[y, "margin_before_si"] == pytest.approx(fy.loc[y, "profit_before_si"] / t.loc[y, "revenue"])
    assert t["wtg_mw"].dropna().gt(0).all()  # zeros in the workbook mean 'not reported'
    assert 2026 not in t.index or pd.isna(t.loc[2026, "profit_before_si"])  # incomplete fiscal years are not added


def test_zero_as_missing_detects_wtg_volume(raw):
    found = dict(quality.zero_as_missing(raw["economy_fy"]))
    assert "WTG volume [MW]" in found and {2023, 2024, 2025} <= set(found["WTG volume [MW]"])  # data sanity


# ---------------------------------------------------------------- quality + insights
def test_quality_flags_known_issues(intel):
    checks = " | ".join(intel.anomalies["Check"])
    assert "Order list is thin since 2023" in checks
    assert "Zero used for 'not reported'" in checks
    assert "Share-price series ends at delisting" in checks
    seg_check = intel.anomalies[intel.anomalies["Check"] == "Segment results vs workbook fiscal-year totals"]
    assert len(seg_check) == 1 and seg_check.iloc[0]["Severity"] == "OK"
    assert set(intel.anomalies["Severity"]) <= {"High", "Medium", "Low", "OK", "Info"}
    assert intel.anomalies["Affected"].map(type).eq(str).all()  # Arrow-safe column


def test_completeness_is_bounded(intel):
    assert intel.completeness.stack().between(0, 1).all()
    assert intel.completeness["Customer named"].loc[2018] < 0.1 and intel.completeness["Customer named"].loc[2024] == 1.0  # data sanity


def test_anomaly_rows_are_subsets_of_orders(orders):
    for name, frame in quality.anomaly_rows(orders).items():
        if "order_id" in frame:
            assert set(frame["order_id"]) <= set(orders["order_id"]), name


def test_insights_generate_and_are_clean(intel):
    items = insights.generate_insights(intel)
    assert len(items) >= 10
    keys = [i.key for i in items]
    assert len(keys) == len(set(keys))
    for i in items:
        text = f"{i.headline} {i.detail} {i.metric} {i.metric_label}"
        assert i.headline and i.detail and i.metric and i.where
        assert not re.search(r"(\bnan\b|\bNaN\b|\bNone\b|\binf\b|n/a)", text), (i.key, text)
        assert i.tone in {"up", "down", "warn", "neutral"}
    assert insights.watchlist(intel)


def test_each_insight_builder_runs_without_swallowed_errors(intel):
    """generate_insights() hides exceptions; call each builder directly so a regression cannot hide."""
    for build in insights.INSIGHT_BUILDERS:
        assert build(intel) is not None, build.__name__


def test_money_formatters():
    assert insights.eur(-438) == "-EUR 438m" and insights.eur(75) == "EUR 75m" and insights.bn(4575) == "EUR 4.6bn"
