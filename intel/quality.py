"""Data coverage and integrity checks, so users know how far to trust each analysis."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .common import UNKNOWN, as_of_date
from .outlook import announced_by_fiscal_year, fiscal_year_table, segment_table


FIELD_LABELS = {
    "customer": "Customer named",
    "country": "Specific country",
    "tenure": "Service tenure",
    "delivery": "Delivery year",
    "platform": "Platform identified",
}


def completeness_by_year(orders: pd.DataFrame) -> pd.DataFrame:
    """Share of each year's announced MW for which a field is disclosed (year x field)."""
    rows = []
    for year, g in orders.groupby("order_year"):
        total = g["size_mw"].sum()
        if total <= 0:
            continue

        def share(mask: pd.Series) -> float:
            return float(g.loc[mask, "size_mw"].sum() / total)

        rows.append(
            {
                "year": int(year),
                FIELD_LABELS["customer"]: share(g["customer"] != UNKNOWN),
                FIELD_LABELS["country"]: share(~g["multi_country"]),
                FIELD_LABELS["tenure"]: share(g["service_time_years"].notna()),
                FIELD_LABELS["delivery"]: share(g["delivery_year"].notna()),
                FIELD_LABELS["platform"]: share(g["primary_platform"] != UNKNOWN),
            }
        )
    return pd.DataFrame(rows).set_index("year")


def _platform_gap_groups(orders: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Orders whose platform MW differs from order size: (5-10% band, 10-25% band, >25% band)."""
    rel = (orders["mw_gap"].abs() / orders["size_mw"]).where(orders["size_mw"] > 0)
    big = orders["mw_gap"].abs() > 1.0
    return (
        orders[(rel > 0.05) & (rel <= 0.10) & big],
        orders[(rel > 0.10) & (rel <= 0.25) & big],
        orders[(rel > 0.25) & big],
    )


def zero_as_missing(economy_fy: pd.DataFrame) -> list[tuple[str, list[int]]]:
    """Metrics that hold 0 in some years although neighbouring years have values (0 standing in for 'not reported')."""
    out = []
    if economy_fy is None or economy_fy.empty:
        return out
    for metric, g in economy_fy.groupby("metric"):
        g = g.sort_values("year")
        zeros = g[g["value"] == 0]
        if zeros.empty or (g["value"] != 0).sum() < 2:
            continue
        out.append((str(metric), [int(y) for y in zeros["year"]]))
    return out


def cross_metric_duplicates(economy: pd.DataFrame, min_value: float = 500.0) -> list[tuple[str, int, str, int, float]]:
    """Identical non-trivial values that sit in two different economy metrics (likely a pasted cell)."""
    if economy is None or economy.empty:
        return []
    e = economy[economy["value"].abs() >= min_value]
    e = e[(e["value"] - e["value"].round()).abs() > 1e-9]  # whole numbers coincide too easily
    e = e.assign(v=e["value"].round(4))
    out = []
    for v, g in e.groupby("v"):
        if g["metric"].nunique() > 1 and len(g) == 2:
            a, b = g.iloc[0], g.iloc[1]
            out.append((str(a["metric"]), int(a["year"]), str(b["metric"]), int(b["year"]), float(v)))
    return out


def anomaly_summary(
    orders: pd.DataFrame,
    platforms: pd.DataFrame,
    results: dict[str, Any] | None,
    economy_fy: pd.DataFrame,
    stock: pd.DataFrame | None = None,
    market: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per check: severity, how many records/years are affected and what it means."""
    as_of = as_of_date(orders)
    rows: list[dict[str, Any]] = []

    def add(check: str, severity: str, affected: Any, detail: str) -> None:
        rows.append({"Check": check, "Severity": severity, "Affected": str(affected), "Detail": detail})

    # --- the order list is thin ----------------------------------------------------------
    per_year = orders.groupby("order_year").size()
    thin = per_year[per_year.index >= 2023]
    if len(thin) and (thin <= 6).all():
        add(
            "Order list is thin since 2023",
            "High",
            ", ".join(f"{y}: {n}" for y, n in thin.items()),
            f"Only {int(thin.sum())} orders are listed for {int(thin.index.min())}-{int(thin.index.max())} against {int(per_year[per_year.index < 2023].mean())} a year before. "
            "Siemens Gamesa has announced few orders since the 4.X/5.X onshore sales stop; its reported order intake in EUR is the reliable series, "
            "so announced-order analytics (timing, accounts, country mix) rest on a small sample for recent years.",
        )

    # --- segment results vs workbook ---------------------------------------------------------
    seg = segment_table(results)
    if not seg.empty and economy_fy is not None and not economy_fy.empty:
        fy = fiscal_year_table(seg)
        wb = economy_fy.pivot_table(index="year", columns="metric", values="value")
        diffs = []
        for year in fy.index[fy["complete"]]:
            for seg_col, wb_col in (("orders", "Total order intake (€m)"), ("revenue", "Total revenue (€m)")):
                if year in wb.index and wb_col in wb.columns and pd.notna(wb.loc[year, wb_col]):
                    diffs.append((year, seg_col, fy.loc[year, seg_col], float(wb.loc[year, wb_col])))
        if diffs:
            worst = max(abs(a - b) / b for _, _, a, b in diffs)
            add(
                "Segment results vs workbook fiscal-year totals",
                "OK" if worst <= 0.005 else "Medium",
                f"{len(diffs)} comparisons",
                f"Quarterly Siemens Energy segment figures add up to the workbook's fiscal-year orders and revenue for FY{min(d[0] for d in diffs)}-FY{max(d[0] for d in diffs)} "
                f"with a largest difference of {worst:.2%} (rounding in the releases).",
            )
        derived = seg[seg["basis"].astype(str).str.startswith("derived")]
        if len(derived):
            add("Derived quarter in the segment series", "Info", ", ".join(derived["label"]),
                "Q3 FY24 is the FY2024 total less the three reported quarters (the release used did not list it); treat it as approximate.")
        miss = seg[seg["backlog_bn"].isna()]
        if len(miss):
            add("Backlog missing for some quarters", "Low", ", ".join(miss["label"]), "Order backlog was not found in the earnings releases used for these quarters.")
        latest_fy = int(seg["fy"].max())
        if latest_fy not in economy_fy["year"].unique():
            add("Workbook economy table stops before the latest fiscal year", "Info", f"FY{latest_fy}",
                f"The workbook's fiscal-year table has no FY{latest_fy} data; the Intelligence tabs use data/sgre_segment_results.json for FY{latest_fy} year-to-date.")

    # --- economy table ---------------------------------------------------------------------
    zeros = zero_as_missing(economy_fy)
    if zeros:
        txt = "; ".join(f"{m} ({', '.join(map(str, ys))})" for m, ys in zeros)
        add("Zero used for 'not reported'", "Medium", len(zeros), txt + ". Shown as missing in the derived economics; a literal zero would mean no turbines were shipped.")
    if economy_fy is not None and not economy_fy.empty:
        ec = economy_fy.sort_values(["metric", "year"]).reset_index(drop=True)
        repeats = []
        for metric, g in ec.groupby("metric"):
            g = g.reset_index(drop=True)
            for i in range(1, len(g)):
                if g.loc[i, "year"] == g.loc[i - 1, "year"] + 1 and g.loc[i, "value"] == g.loc[i - 1, "value"] and abs(g.loc[i, "value"]) >= 50:
                    repeats.append(f"{metric} {int(g.loc[i - 1, 'year'])}/{int(g.loc[i, 'year'])} = {g.loc[i, 'value']:,.0f}")
        if repeats:
            add("Identical economics value in consecutive years", "Medium", len(repeats), "; ".join(repeats) + ". Possibly copied rather than reported.")
        dups = cross_metric_duplicates(economy_fy)
        if dups:
            txt = "; ".join(f"{a} {ya} and {b} {yb} are both {v:,.1f}" for a, ya, b, yb, v in dups)
            add("Same value in two different economics rows", "Medium", len(dups), txt + ". A pasted cell is likely.")

    # --- order list --------------------------------------------------------------------------
    small, moderate, large = _platform_gap_groups(orders)
    if len(small) + len(moderate):
        both = pd.concat([small, moderate])
        add("Platform MW differs somewhat from order size", "Info", len(both),
            f"{len(both)} orders ({both['size_mw'].sum():,.0f} MW) have platform-level MW 5-25% away from the order size (net {both['mw_gap'].sum():,.0f} MW), "
            "typically a rounded turbine count or a power mode that differs from the nameplate rating.")
    if len(large):
        add("Platform MW differs a lot from order size", "Medium", len(large),
            f"{len(large)} orders ({large['size_mw'].sum():,.0f} MW) have platform-level MW more than 25% away from the order size "
            f"({(large['platform_mw_sum'] / large['size_mw']).min():.0%}-{(large['platform_mw_sum'] / large['size_mw']).max():.0%} of it): an incomplete platform split or an inconsistent entry. "
            "Platform-based charts use platform-slot MW, order-based charts use order MW.")

    unknown_platform = orders[orders["primary_platform"] == UNKNOWN]
    if len(unknown_platform):
        yrs = unknown_platform.groupby("order_year")["size_mw"].sum()
        add("Platform unknown", "Medium", f"{len(unknown_platform)} orders",
            f"{len(unknown_platform)} orders ({unknown_platform['size_mw'].sum():,.0f} MW) carry no platform, mostly {int(yrs.index.min())}-{int(yrs.idxmax())}. "
            "They are excluded from product-mix charts, and onshore/offshore splits cover only platform-identified orders.")

    multi = orders[orders["multi_country"]]
    if len(multi):
        add("Orders without a country", "Low", len(multi), f"{len(multi)} orders ({multi['size_mw'].sum():,.0f} MW) have no country; excluded from country-level analytics.")

    placeholders = orders[orders["customer_raw"].astype(str).str.lower().isin({"new", "first"})]
    if len(placeholders):
        add("Placeholder customer names", "Low", len(placeholders),
            "Customer is just " + ", ".join(sorted(set(placeholders["customer_raw"].astype(str)))) + f" for {len(placeholders)} orders (truncated entries); treated as undisclosed.")

    twins = orders[orders.duplicated(["order_date", "country", "customer", "size_mw"], keep=False)]
    if len(twins):
        add("Possible duplicate order rows", "Low", len(twins), f"{len(twins)} rows share date, country, customer and size - often twin tranches, but worth a look.")

    neg = orders[orders["delivery_days"] < 0]
    if len(neg):
        add("Delivery before order date", "Low", len(neg), f"{len(neg)} orders have a delivery date earlier than the order date.")

    old = orders[orders["order_year"] <= 2019]
    new = orders[orders["order_year"] >= as_of.year - 2]
    cust_old = float(old.loc[old["customer"] == UNKNOWN, "size_mw"].sum() / old["size_mw"].sum())
    cust_new = float(new.loc[new["customer"] == UNKNOWN, "size_mw"].sum() / new["size_mw"].sum()) if new["size_mw"].sum() else float("nan")
    add("Customer undisclosed", "High" if cust_old > 0.5 else "Medium", f"{cust_old:.0%} of MW in 2017-2019",
        f"Customer is unknown for {cust_old:.0%} of announced MW in 2017-2019 and {cust_new:.0%} since {as_of.year - 2}; account analytics are reliable only for recent years.")

    # --- external series ---------------------------------------------------------------------
    if stock is not None and not stock.empty:
        last = pd.Timestamp(stock["date"].max())
        add("Share-price series ends at delisting", "Info", str(last.date()),
            f"The stock series ends {last.date()}: Siemens Gamesa was delisted after Siemens Energy took full ownership, so there is no later share price.")
    if market is not None and not market.empty:
        last = pd.Timestamp(market["date"].max())
        months = (as_of.year - last.year) * 12 + as_of.month - last.month
        if months >= 3:
            add("Steel/copper series are stale", "Info", f"{months} months", f"Steel/copper data ends {last.date()}; run tools/refresh_market_data.py to extend the overlays.")
    return pd.DataFrame(rows)


def anomaly_rows(orders: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Detail tables behind the anomaly checks."""
    cols = ["order_id", "order_date", "country", "customer", "size_mw", "primary_platform"]
    out: dict[str, pd.DataFrame] = {}
    unk = orders[orders["primary_platform"] == UNKNOWN][cols]
    if len(unk):
        out["Platform unknown"] = unk.sort_values("size_mw", ascending=False).reset_index(drop=True)
    small, moderate, large = _platform_gap_groups(orders)
    for name, frame in (("Platform MW differs a lot from order size", large), ("Platform MW differs somewhat from order size", pd.concat([small, moderate]))):
        if len(frame):
            f = frame[cols + ["platform_mw_sum", "mw_gap"]].copy()
            f["platform / order"] = (f["platform_mw_sum"] / f["size_mw"]).round(2)
            out[name] = f.sort_values("mw_gap", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
    multi = orders[orders["multi_country"]][cols]
    if len(multi):
        out["Orders without a country"] = multi.reset_index(drop=True)
    ph = orders[orders["customer_raw"].astype(str).str.lower().isin({"new", "first"})][cols]
    if len(ph):
        out["Placeholder customer names"] = ph.reset_index(drop=True)
    twins = orders[orders.duplicated(["order_date", "country", "customer", "size_mw"], keep=False)][cols]
    if len(twins):
        out["Possible duplicate order rows"] = twins.sort_values(["order_date", "country"]).reset_index(drop=True)
    return out
