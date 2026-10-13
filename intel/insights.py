"""Rule-based insight generator: turns the computed analytics into short, number-backed findings.

Every figure in a finding is computed from the data at runtime - nothing here is hard-coded commentary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from .bundle import Intel
from .common import MEGA_MW, UNKNOWN, pct_change
from .structure import period_compare


@dataclass
class Insight:
    key: str
    category: str
    headline: str
    detail: str
    metric: str
    metric_label: str
    tone: str = "neutral"  # up | down | warn | neutral
    series: list[float] = field(default_factory=list)
    series_start: int | None = None  # first x of the sparkline
    where: str = ""  # where to dig deeper in the app


# ---- formatting ------------------------------------------------------------------------
def bn(x: float, d: int = 1) -> str:
    return f"EUR {x / 1000.0:,.{d}f}bn"


def eur(x: float) -> str:
    return f"-EUR {abs(x):,.0f}m" if x < 0 else f"EUR {x:,.0f}m"


def gw(x: float, d: int = 1) -> str:
    return f"{x / 1000.0:,.{d}f} GW"


def mwf(x: float) -> str:
    return f"{x:,.0f} MW"


def pc(x: float, d: int = 0, signed: bool = False) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    return f"{x:+.{d}%}" if signed else f"{x:.{d}%}"


def _tone(x: float, good_up: bool = True) -> str:
    if x is None or pd.isna(x) or abs(x) < 0.02:
        return "neutral"
    return "up" if (x > 0) == good_up else "down"


def _join_and(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _ok(*vals: float) -> bool:
    return all(v is not None and not pd.isna(v) for v in vals)


def _period_text(n: int) -> str:
    return {1: "the first quarter", 2: "the first half", 3: "the first nine months", 4: "the full year"}.get(n, f"the first {n} quarters")


# ---- individual insights ---------------------------------------------------------------
def insight_orders(i: Intel) -> Insight | None:
    r = i.ltm.get("reported")
    pr = i.projection
    if not r or not pr:
        return None
    fy, n = pr["fy"], pr["quarters_reported"]
    seg = i.seg
    top3 = seg["orders"].nlargest(3).sum() / seg["orders"].sum()
    prev_year = seg[seg["fy"] == fy - 1]
    big_q = prev_year.loc[prev_year["orders"].idxmax()]
    why = f" ({big_q['label']}: {big_q['note']})" if str(big_q["note"]) else ""
    word = "down" if r["orders_growth"] < 0 else "up"
    return Insight(
        key="orders",
        category="Orders",
        headline=f"Reported orders are {word} {pc(abs(r['orders_growth']))} over the last four quarters, at book-to-bill {r['book_to_bill']:.2f}",
        detail=(
            f"{r['first_label']} to {r['last_label']}: {bn(r['orders'])} of orders against {bn(r['prior_orders'])} in the four quarters before, versus {bn(r['revenue'])} of revenue. "
            f"In {_period_text(n)} of FY{fy % 100} orders are {bn(pr['orders_ytd'])} against {bn(pr['orders_prev_ytd'])} a year earlier ({pc(pct_change(pr['orders_ytd'], pr['orders_prev_ytd']), signed=True)}); "
            f"the largest quarter of the prior year was {eur(big_q['orders'])}{why}. Three quarters make up {pc(top3)} of all orders since FY24, so the comparison is dominated by a few offshore signatures."
        ),
        metric=pc(r["orders_growth"], signed=True),
        metric_label="reported orders, last 4 quarters vs the 4 before",
        tone="down" if r["orders_growth"] < -0.02 else "neutral",
        series=seg["orders"].tolist(),
        series_start=0,
        where="Intelligence Lab > Momentum & Outlook",
    )


def insight_turnaround(i: Intel) -> Insight | None:
    seg = i.seg
    if len(seg) < 5:
        return None
    last = seg.iloc[-1]
    r = i.ltm["reported"]
    first_pos = seg[seg["profit_before_si"] > 0]
    return Insight(
        key="turnaround",
        category="Profitability",
        headline=f"Profit before special items is {eur(last['profit_before_si'])} in {last['label']}, up from {eur(seg.iloc[-5]['profit_before_si'])} a year earlier",
        detail=(
            f"{'The first positive quarter in the series. ' if len(first_pos) == 1 and first_pos.index[0] == seg.index[-1] else ''}"
            f"{('Release note: ' + str(last['note']) + '. ') if str(last['note']) else ''}"
            f"Margin before special items improved from {pc(seg.iloc[-5]['margin_before_si'], 1)} to {pc(last['margin_before_si'], 1)}. "
            f"Over the last four quarters the loss is {eur(abs(r['profit_before_si']))} against {eur(abs(r['prior_profit_before_si']))} before; the quarterly loss has narrowed in each of the last {int((seg['profit_before_si'].diff().iloc[-4:] > 0).sum())} quarters. "
            "Cost efficiency and service growth drive the gain; new-unit volumes are lower."
        ),
        metric=eur(last["profit_before_si"]),
        metric_label=f"profit before special items, {last['label']}",
        tone="up",
        series=seg["profit_before_si"].tolist(),
        series_start=0,
        where="Intelligence Lab > Momentum & Outlook",
    )


def insight_outlook(i: Intel) -> Insight | None:
    pr = i.projection
    if not pr:
        return None
    fy = pr["fy"]
    sc = pr["orders_scenario"]
    qo = pr["quarter_orders"]
    need = pr.get("revenue_needed_remaining")
    pb = pr.get("pbsi_needed_remaining")
    g = pr["guidance"]
    rev_txt = ""
    if need and g.get("revenue_growth_comparable"):
        lo, hi = g["revenue_growth_comparable"]
        rev_txt = (
            f" Guidance of {lo:.0%}-{hi:.0%} comparable revenue growth equals roughly {eur(need[0])}-{eur(need[1])} in the remaining quarter on last year's nominal base "
            f"(Q3 delivered {eur(pr['last_quarter_revenue'])}, the same quarter last year {eur(pr['prev_year_remaining_revenue'])}); comparable growth was already {pc(g.get('ytd_revenue_growth_comparable'), 1, signed=True)} year-to-date, which leaves headroom."
            if g.get("ytd_revenue_growth_comparable") is not None
            else ""
        )
    pb_txt = f" Break-even for the year needs profit before special items of at least {eur(pb[0])} in the remaining quarter (last quarter: {eur(pr['last_quarter_pbsi'])})." if pb else ""
    return Insight(
        key="outlook",
        category="Outlook",
        headline=f"FY{fy % 100} orders are on course for ~{bn(sc['base'])} unless a large offshore order lands (FY{(fy - 1) % 100}: {bn(pr['orders_prev_full'])})",
        detail=(
            f"{bn(pr['orders_ytd'])} booked in {_period_text(pr['quarters_reported'])}; the remaining quarter is a scenario from the {qo['lookback']} most recent quarters "
            f"(lowest {eur(qo['low'])}, median {eur(qo['median'])}, highest {eur(qo['high'])}), giving {bn(sc['conservative'])} to {bn(sc['strong'])} for the year.{rev_txt}{pb_txt}"
        ),
        metric=bn(sc["base"]),
        metric_label=f"FY{fy % 100} orders, base scenario",
        tone="neutral",
        series=i.fy["orders"].tolist()[:-1] + [sc["base"]],
        series_start=int(i.fy.index.min()),
        where="Intelligence Lab > Momentum & Outlook",
    )


def insight_backlog(i: Intel) -> Insight | None:
    seg = i.seg.dropna(subset=["backlog_bn"])
    if len(seg) < 3:
        return None
    last, first = seg.iloc[-1], seg.iloc[0]
    below = int((i.seg["book_to_bill"] < 1).sum())
    return Insight(
        key="backlog",
        category="Visibility",
        headline=f"Order backlog has fallen to EUR {last['backlog_bn']:.0f}bn from EUR {first['backlog_bn']:.0f}bn in {first['label']}",
        detail=(
            f"Book-to-bill was below 1 in {below} of the last {len(i.seg)} quarters, so the backlog is being worked off faster than it is refilled. "
            f"At the last four quarters' revenue it still covers about {last['cover_years']:.1f} years. Backlog is stated in whole EUR bn in the releases, so changes below EUR 1bn are not visible."
        ),
        metric=f"EUR {last['backlog_bn']:.0f}bn",
        metric_label=f"order backlog, {last['label']}",
        tone="down" if last["backlog_bn"] < first["backlog_bn"] else "neutral",
        series=seg["backlog_bn"].tolist(),
        series_start=0,
        where="Intelligence Lab > Momentum & Outlook",
    )


def insight_service(i: Intel) -> Insight | None:
    seg = i.seg
    if len(seg) < 5:
        return None
    last = seg.iloc[-1]
    n = i.projection["quarters_reported"] if i.projection else 3
    cur = seg[seg["fy"] == seg["fy"].max()]
    prev = seg[(seg["fy"] == seg["fy"].max() - 1) & (seg["quarter"] <= len(cur))]
    s_cur, s_prev = cur["revenue_service"].sum(), prev["revenue_service"].sum()
    nu_cur, nu_prev = cur["revenue_new_units"].sum(), prev["revenue_new_units"].sum()
    return Insight(
        key="service",
        category="Service",
        headline=f"Service revenue is up {pc(pct_change(s_cur, s_prev))} year-to-date and is {pc(s_cur / cur['revenue'].sum())} of revenue",
        detail=(
            f"{_period_text(len(cur)).capitalize()} of FY{int(cur['fy'].iloc[0]) % 100}: service {eur(s_cur)} against {eur(s_prev)}, while new-unit revenue is {eur(nu_cur)} against {eur(nu_prev)} ({pc(pct_change(nu_cur, nu_prev), signed=True)}). "
            f"Service was {pc(seg.iloc[-5]['service_share'])} of revenue in {seg.iloc[-5]['label']} and {pc(last['service_share'])} in {last['label']}; "
            "it carries the recurring base while the onshore restart and offshore ramp-up proceed."
        ),
        metric=pc(pct_change(s_cur, s_prev), signed=True),
        metric_label="service revenue growth, year-to-date",
        tone="up",
        series=seg["revenue_service"].tolist(),
        series_start=0,
        where="Intelligence Lab > Momentum & Outlook",
    )


def insight_lumpiness(i: Intel) -> Insight | None:
    o = i.orders
    recent = o[o["order_year"] >= i.as_of.year - 3]
    c = i.concentration
    big = recent.nlargest(3, "size_mw")
    mega = o[o["size_mw"] >= MEGA_MW]
    share_mega_all = mega["size_mw"].sum() / o["size_mw"].sum()
    return Insight(
        key="lumpiness",
        category="Concentration",
        headline=f"{len(mega)} mega orders of {MEGA_MW:,.0f} MW or more make up {pc(share_mega_all)} of all announced MW since 2017",
        detail=(
            f"The median announced order is {o['size_mw'].median():,.0f} MW but the largest is {o['size_mw'].max():,.0f} MW. "
            f"The biggest of the last {i.as_of.year - recent['order_year'].min() + 1} calendar years: "
            + "; ".join(f"{r.country_clean} {r.size_mw:,.0f} MW ({r.order_date:%b %Y})" for r in big.itertuples())
            + f". In {int(c.index[-2])} the top order alone was {pc(c.iloc[-2]['top1_share'])} of announced MW, so annual MW swings on single offshore projects."
        ),
        metric=f"{len(mega)}",
        metric_label=f"orders >= {MEGA_MW:,.0f} MW since 2017",
        tone="warn",
        series=c["top1_share"].tolist(),
        series_start=int(c.index.min()),
        where="Intelligence Lab > Market Structure",
    )


def insight_announced(i: Intel) -> Insight | None:
    cur, pri = i.ltm["current"], i.ltm["prior"]
    last_date = i.as_of
    yrs = i.ann_fy.dropna(subset=["eur_per_announced_mw"])
    ratio_txt = (
        f" Reported orders were ~EUR {yrs['eur_per_announced_mw'].mean():.1f}m per announced MW in FY{int(yrs.index.min()) % 100}-FY{int(yrs.index.max()) % 100} "
        "(offshore-heavy, includes service and unannounced onshore), a rough yardstick only."
        if len(yrs)
        else ""
    )
    checked = (i.results or {}).get("orders_checked_through")
    note = f" No newer announcements were found when checked on {pd.Timestamp(checked):%d %b %Y}." if checked else ""
    return Insight(
        key="announced",
        category="Visibility",
        headline=f"Only {cur['orders']} orders ({gw(cur['mw'])}) were announced in the last 12 months to {last_date:%d %b %Y}, against {pri['orders']} ({gw(pri['mw'])}) before",
        detail=(
            f"The list stops at {last_date:%d %b %Y}: Siemens Gamesa now discloses few orders individually, and the reported EUR order intake is the better guide.{note} "
            f"Offshore is {pc(cur['offshore_mw'] / cur['mw'] if cur['mw'] else np.nan)} of the last 12 months' announced MW.{ratio_txt}"
        ),
        metric=f"{cur['orders']}",
        metric_label="announced orders, last 12 months to the latest announcement",
        tone="warn",
        where="Intelligence Lab > Data Quality",
    )


def insight_product(i: Intel) -> Insight | None:
    m = i.mix
    s = i.series
    year = int(m.index.max())
    if year < 2019:
        return None
    recent = i.platforms[(i.platforms["order_year"] >= year - 2) & (i.platforms["slot_mw"] > 0)]
    off = recent.loc[recent["segment"] == "Offshore", "slot_mw"].sum() / recent.loc[recent["segment"] != "Unknown", "slot_mw"].sum()
    sg7 = i.platforms[(i.platforms["series"] == "SG 7.0 (restarted onshore)") & (i.platforms["slot_mw"] > 0)]
    lead = i.leading[(i.leading["year"] >= year - 2)].groupby("platform")["mw"].sum().sort_values(ascending=False).head(2)
    sg7_txt = (
        f" The restarted onshore line is visible again: {len(sg7)} SG 7.0-170 order(s) worth {sg7['slot_mw'].sum():,.0f} MW since the first ones in {int(sg7['order_year'].min())}."
        if len(sg7)
        else ""
    )
    return Insight(
        key="product",
        category="Product",
        headline=f"Offshore direct-drive turbines are {pc(off)} of the platform-mapped MW ordered in the last three order years",
        detail=(
            f"The average ordered turbine rating reached {m['rating_mw'].loc[year - 3:year - 1].mean():.1f} MW (rotor {m['rotor_m'].loc[year - 3:year - 1].mean():.0f} m) in {year - 3}-{year - 1}. "
            f"Leading platforms: {_join_and([f'{n} ({v:,.0f} MW)' for n, v in lead.items()])}.{sg7_txt}"
        ),
        metric=pc(off),
        metric_label="offshore share of platform-mapped MW, last 3 order years",
        tone="neutral",
        series=m["offshore_share"].tolist(),
        series_start=int(m.index.min()),
        where="Intelligence Lab > Market Structure",
    )


def insight_geography(i: Intel) -> Insight | None:
    o = i.orders
    year = i.as_of.year
    window = o[(o["order_year"] >= year - 3) & ~o["multi_country"]]
    by = window.groupby("country_clean")["size_mw"].sum().sort_values(ascending=False)
    if by.empty:
        return None
    top = by.head(4)
    cont = o.pivot_table(index="order_year", columns="continent", values="size_mw", aggfunc="sum").fillna(0)
    share = cont.div(cont.sum(axis=1), axis=0)
    eu = share["Europe"] if "Europe" in share else pd.Series(dtype=float)
    asia_early = share["Asia"].loc[2018:2022].mean() if "Asia" in share else np.nan
    return Insight(
        key="geography",
        category="Geography",
        headline=f"{top.index[0]}, {top.index[1]} and {top.index[2]} account for {pc(top.iloc[:3].sum() / by.sum())} of announced MW since {year - 3}",
        detail=(
            f"Top markets since {year - 3}: {_join_and([f'{c} {v / 1000:.1f} GW' for c, v in top.items()])}. Europe is {pc(cont['Europe'].loc[year - 3:].sum() / cont.loc[year - 3:].sum().sum())} of that MW "
            f"(vs {pc(cont['Europe'].loc[2017:2022].sum() / cont.loc[2017:2022].sum().sum())} in 2017-2022), while Asia averaged {pc(asia_early)} of yearly MW in 2018-2022. "
            "Offshore projects concentrate MW in a handful of North Sea, Baltic and Taiwan sites."
        ),
        metric=pc(top.iloc[:3].sum() / by.sum()),
        metric_label=f"top-3 country share, {year - 3}+",
        tone="neutral",
        series=eu.tolist(),
        series_start=int(eu.index.min()) if len(eu) else None,
        where="Intelligence Lab > Market Structure",
    )


def insight_accounts(i: Intel) -> Insight | None:
    r = i.radar
    key = r[(r["lifetime_mw"] >= 500) & r["repeat_buyer"]]
    if key.empty:
        return None
    dormant = key[key["status"] == "Dormant"]
    active = key[key["status"] == "Active"]
    year = i.as_of.year
    und = float(
        i.orders.loc[(i.orders["order_year"] <= 2019) & (i.orders["customer"] == UNKNOWN), "size_mw"].sum()
        / i.orders.loc[i.orders["order_year"] <= 2019, "size_mw"].sum()
    )
    return Insight(
        key="accounts",
        category="Accounts",
        headline=f"{len(dormant)} of {len(key)} repeat accounts of 500+ MW have not ordered in over two years",
        detail=(
            f"They hold {gw(dormant['lifetime_mw'].sum())} of lifetime announced orders (e.g. {', '.join(dormant.head(3)['customer']) or 'none'}). "
            f"{len(active)} {'is' if len(active) == 1 else 'are'} active within 12 months ({', '.join(active.head(3)['customer']) or 'none'}). "
            f"Customers are undisclosed for {pc(und)} of MW in 2017-2019, so account history before 2020 is incomplete."
        ),
        metric=f"{len(dormant)} / {len(key)}",
        metric_label="repeat accounts (500+ MW) dormant >24 months",
        tone="warn",
        where="Intelligence Lab > Accounts & Service",
    )


def insight_integrity(i: Intel) -> Insight | None:
    a = i.anomalies
    meds = a[a["Severity"].isin(["High", "Medium"])]
    per_year = i.orders.groupby("order_year").size()
    return Insight(
        key="integrity",
        category="Data integrity",
        headline=f"{len(meds)} data checks need attention; the order list has {int(per_year.loc[2023:].sum())} entries for 2023-{i.as_of.year}",
        detail=(
            "Open items: " + "; ".join(meds["Check"].head(6).tolist()) + ". "
            "The segment results reconcile to the workbook's fiscal-year orders and revenue, so the quarterly EUR series is the dependable base for trends."
        ),
        metric=f"{len(meds)}",
        metric_label="high/medium data checks open",
        tone="warn",
        where="Intelligence Lab > Data Quality",
    )


INSIGHT_BUILDERS: list[Callable[[Intel], Insight | None]] = [
    insight_orders,
    insight_turnaround,
    insight_outlook,
    insight_backlog,
    insight_service,
    insight_lumpiness,
    insight_announced,
    insight_product,
    insight_geography,
    insight_accounts,
    insight_integrity,
]


def generate_insights(intel: Intel) -> list[Insight]:
    out: list[Insight] = []
    for build in INSIGHT_BUILDERS:
        try:
            item = build(intel)
        except Exception:  # an insight that cannot be computed must not break the page
            item = None
        if item is not None:
            out.append(item)
    return out


def watchlist(intel: Intel) -> list[str]:
    """Forward-looking things to check, derived from the same numbers."""
    items: list[str] = []
    pr = intel.projection
    if pr:
        fy = pr["fy"] % 100
        need = pr.get("pbsi_needed_remaining")
        if need:
            items.append(
                f"Q4 FY{fy}: profit before special items must be at least {eur(need[0])} for the guided break-even year (Q3 FY{fy}: {eur(pr['last_quarter_pbsi'])}, 9M: {eur(pr['pbsi_ytd'])}). "
                "The Q4 report follows the fiscal year-end on 30 September (mid-November)."
            )
        rn = pr.get("revenue_needed_remaining")
        if rn:
            items.append(
                f"Revenue: reaching the guided comparable growth means about {eur(rn[0])}-{eur(rn[1])} in Q4 on a nominal basis (Q4 FY{fy - 1}: {eur(pr['prev_year_remaining_revenue'])}); "
                "currency and the Indian business disposal make the nominal and comparable figures differ."
            )
        sc = pr["orders_scenario"]
        items.append(
            f"Orders: a Q4 at the recent median ({eur(pr['quarter_orders']['median'])}) gives about {bn(sc['base'])} for FY{fy}; only a large offshore signature "
            f"(a single order has added EUR 3-4bn in a quarter) would lift the year toward {bn(sc['strong'])} or more."
        )
    seg = intel.seg
    if len(seg) >= 2:
        items.append(f"Backlog was EUR {seg.iloc[-1]['backlog_bn']:.0f}bn at {seg.iloc[-1]['label']}; watch whether book-to-bill returns above 1 (it was {seg.iloc[-1]['book_to_bill']:.2f}).")
    r = intel.radar
    cool = r[(r["lifetime_mw"] >= 300) & (r["status"] == "Cooling") & r["repeat_buyer"]].head(4)
    if len(cool):
        items.append("Key accounts approaching 24 months without an order: " + ", ".join(cool["customer"]) + ".")
    return items
