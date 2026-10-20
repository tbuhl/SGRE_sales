"""Links between order flow and company economics (fiscal years, EUR m): book-to-bill, backlog cover, service mix, margins."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import eco_series, eco_wide


def economics_table(economy_fy: pd.DataFrame, fy_results: pd.DataFrame | None = None, market: pd.DataFrame | None = None) -> pd.DataFrame:
    """Annual KPIs by fiscal year from the workbook's FY table, completed with the reported segment results.

    ``fy_results`` (from ``outlook.fiscal_year_table``) supplies profit before special items, which the workbook lacks;
    its complete years also override workbook revenue/orders only where the workbook has no value.
    """
    wide = eco_wide(economy_fy)
    if wide.empty:
        return pd.DataFrame()

    def s(*names: str) -> pd.Series:
        return eco_series(wide, *names)

    t = pd.DataFrame(index=wide.index)
    t["revenue"] = s("Total revenue (€m)")
    t["revenue_service"] = s("Revenue - Service (€m)")
    t["revenue_new_units"] = s("Revenue - Turbines / New Units (€m)")
    t["net_profit"] = s("net profit")
    t["order_intake"] = s("Total order intake (€m)")
    t["orders_onshore"] = s("Total order onshore(€m)")
    t["orders_offshore"] = s("Total order offshore(€m)")
    t["orders_service"] = s("Order intake - Service (€m)")
    wtg = s("WTG volume [MW]").where(lambda x: x > 0)  # the workbook holds 0 instead of 'not reported' after FY2022
    t["wtg_mw"] = wtg
    t["wtg_mw_offshore"] = s("WTG volume [MW] offshore")
    t["wtg_mw_onshore"] = s("WTG volume [MW] onshore")
    t["employees"] = s("Employees")
    t["asp_onshore"] = s("Average selling price (onshore)").combine_first(s("average selling price onshore"))  # two spellings of one series
    t["rev_per_mw"] = (t["revenue_new_units"] / t["wtg_mw"]).where(t["wtg_mw"] > 0)  # new-unit revenue, onshore and offshore, per delivered MW
    t["rev_per_mw_3y"] = t["revenue_new_units"].rolling(3).sum() / t["wtg_mw"].rolling(3).sum()
    for col, name in (("backlog_onshore_bn", "order backlog onshore"), ("backlog_offshore_bn", "order backlog offshore"), ("backlog_service_bn", "order backlog service")):
        t[col] = s(name)
    t["backlog_bn"] = t[["backlog_onshore_bn", "backlog_offshore_bn", "backlog_service_bn"]].sum(axis=1, min_count=3)

    if fy_results is not None and not fy_results.empty:
        done = fy_results[fy_results["complete"]]
        t["profit_before_si"] = done["profit_before_si"].reindex(t.index)
        extra = done.index.difference(t.index)
        if len(extra):
            add = done.loc[extra, ["revenue", "revenue_service", "revenue_new_units", "orders", "profit_before_si"]].rename(columns={"orders": "order_intake"})
            t = pd.concat([t, add]).sort_index()
        # fill missing workbook values from the reported segment quarters
        for col, src in (("revenue", "revenue"), ("order_intake", "orders"), ("revenue_service", "revenue_service"), ("revenue_new_units", "revenue_new_units")):
            t[col] = t[col].fillna(done[src].reindex(t.index))
    else:
        t["profit_before_si"] = np.nan

    t["net_margin"] = t["net_profit"] / t["revenue"]
    t["margin_before_si"] = t["profit_before_si"] / t["revenue"]
    t["service_share"] = t["revenue_service"] / t["revenue"]
    t["service_growth"] = t["revenue_service"].pct_change(fill_method=None)
    t["revenue_growth"] = t["revenue"].pct_change(fill_method=None)
    t["book_to_bill"] = t["order_intake"] / t["revenue"]
    t["offshore_order_share"] = t["orders_offshore"] / (t["orders_onshore"] + t["orders_offshore"])
    t["offshore_mw_share"] = t["wtg_mw_offshore"] / (t["wtg_mw_onshore"] + t["wtg_mw_offshore"])
    t["cover_years"] = t["backlog_bn"] * 1000.0 / t["revenue"]
    t["revenue_per_employee_keur"] = t["revenue"] * 1000.0 / t["employees"]
    if market is not None and not market.empty:
        m = market.copy()
        m["year"] = pd.to_datetime(m["date"]).dt.year
        annual = m.groupby("year")[["steel_hrc_usd_per_short_ton", "copper_hg_usd_per_lb"]].mean()
        t["steel_usd_per_ton"] = annual["steel_hrc_usd_per_short_ton"].reindex(t.index)
    return t
