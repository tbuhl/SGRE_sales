"""Streamlit layout for the Executive Briefing and the Intelligence Lab."""
from __future__ import annotations

import html

import numpy as np
import pandas as pd
import streamlit as st

from . import charts_outlook as co
from . import charts_structure as cs
from . import quality, structure
from .bundle import Intel, build_intel
from .common import GIGA_MW, MEGA_MW, UNKNOWN, load_segment_results, prepare_orders, prepare_platforms, reported_results_path
from .insights import bn, eur, generate_insights, gw, pc, watchlist
from .theme import finding_card_html, inject_css, show, table_view, tokens

LAB_SECTIONS = ["Momentum & Outlook", "Market Structure", "Accounts & Service", "Compare Lab", "Data Quality"]


def _data_signature(economy_fy, orders, platforms, stock, market) -> str:
    """Cheap fingerprint of the inputs, so the cached analytics refresh only when the data changes."""
    rp = reported_results_path()
    rp_sig = f"{rp.stat().st_mtime_ns}" if rp.exists() else "no-results"
    return "|".join(
        [
            str(len(orders)), str(orders["order_date"].max()), f"{orders['size_mw'].sum():.2f}", str(len(platforms)), f"{platforms['slot_mw'].sum():.2f}",
            str(len(economy_fy)), f"{economy_fy['value'].sum():.2f}",
            str(None if stock is None else stock.shape), str(None if market is None else market.shape), rp_sig,
        ]
    )


@st.cache_data(show_spinner="Computing the intelligence layer...")
def _cached_intel(signature: str, _economy_fy, _orders, _platforms, _stock, _market) -> Intel:  # noqa: ARG001 - signature drives the cache
    return build_intel(_economy_fy, _orders, _platforms, _stock, _market, load_segment_results())


def get_intel(economy_fy, orders, platforms, stock=None, market=None) -> Intel:
    return _cached_intel(_data_signature(economy_fy, orders, platforms, stock, market), economy_fy, orders, platforms, stock, market)


def _scope_note(text: str) -> None:
    st.markdown(f'<div class="intel-note">{html.escape(text)}</div>', unsafe_allow_html=True)


def _csv_button(df: pd.DataFrame, label: str, name: str, key: str) -> None:
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), file_name=name, mime="text/csv", key=key)


# ======================================================================================
# Executive Briefing
# ======================================================================================
def render_executive_briefing(intel: Intel) -> None:
    inject_css()
    t = tokens()
    pr = intel.projection
    r = intel.ltm.get("reported")
    seg = intel.seg

    st.markdown(
        f'<div class="intel-hero"><h2>Executive Briefing</h2>'
        f'<p class="intel-sub">Company-level view. Reported results run to {seg.iloc[-1]["label"] if len(seg) else "n/a"} (Siemens Energy earnings releases); '
        f"the order list's latest announcement is {intel.as_of:%d %B %Y}. Every number below is computed from the data - sidebar filters apply to the other tabs, not to this one.</p></div>",
        unsafe_allow_html=True,
    )

    # ---- KPI strip --------------------------------------------------------------------
    cols = st.columns(6)
    if r:
        cols[0].metric("Reported orders, last 4 quarters", bn(r["orders"]), f"{r['orders_growth']:+.0%} vs the 4 before")
        cols[1].metric("Book-to-bill, last 4 quarters", f"{r['book_to_bill']:.2f}", f"revenue {bn(r['revenue'])}", delta_color="off", delta_arrow="off")
    if len(seg) >= 5:
        last, prev = seg.iloc[-1], seg.iloc[-5]
        cols[2].metric(f"Profit before special items, {last['label']}", eur(last["profit_before_si"]), f"vs {eur(prev['profit_before_si'])} in {prev['label']}", delta_color="off", delta_arrow="off")
    if pr:
        sc = pr["orders_scenario"]
        cols[3].metric(f"FY{pr['fy'] % 100} orders (base scenario)", bn(sc["base"]), f"range {sc['conservative'] / 1000:.1f}-{sc['strong'] / 1000:.1f}bn; FY{(pr['fy'] - 1) % 100}: {pr['orders_prev_full'] / 1000:.1f}bn", delta_color="off", delta_arrow="off")
    bl = seg.dropna(subset=["backlog_bn"])
    if len(bl):
        row = bl.iloc[-1]
        cols[4].metric(f"Order backlog, {row['label']}", f"EUR {row['backlog_bn']:.0f}bn", f"{row['cover_years']:.1f} yrs of revenue", delta_color="off", delta_arrow="off")
    if len(seg) >= 5:
        cur = seg[seg["fy"] == seg["fy"].max()]
        prev_y = seg[(seg["fy"] == seg["fy"].max() - 1) & (seg["quarter"] <= len(cur))]
        cols[5].metric("Service revenue, year-to-date", eur(cur["revenue_service"].sum()), f"{cur['revenue_service'].sum() / prev_y['revenue_service'].sum() - 1:+.0%} vs last year")

    # ---- hero charts --------------------------------------------------------------------
    c1, c2 = st.columns(2)
    with c1:
        show(co.fig_segment_quarterly(intel), "brief_seg")
    with c2:
        show(co.fig_fy_orders(intel), "brief_fy")
    st.caption(
        "Orders are lumpy: one offshore project can add EUR 3-4bn in a quarter, so the scenario for the remaining quarter uses the lowest, median and highest of the last "
        f"{pr.get('quarter_orders', {}).get('lookback', 8)} reported quarters. It is a scenario range, not a forecast."
    )
    c3, c4 = st.columns(2)
    with c3:
        show(co.fig_segment_profit(intel), "brief_profit")
    with c4:
        show(co.fig_backlog_trend(intel), "brief_backlog")

    # ---- findings -----------------------------------------------------------------------
    st.subheader("Key findings")
    items = generate_insights(intel)
    if items:
        st.markdown('<div class="intel-grid">' + "".join(finding_card_html(i, t) for i in items) + "</div>", unsafe_allow_html=True)
    else:
        st.info("Not enough data to generate findings.")

    st.subheader("What to watch next")
    wl = watchlist(intel)
    if wl:
        st.markdown('<ul class="intel-watch">' + "".join(f"<li>{html.escape(w)}</li>" for w in wl) + "</ul>", unsafe_allow_html=True)
    with st.expander("How these findings are generated"):
        st.markdown(
            "- Findings are rule-based: each template pulls its numbers from the same tables used on the Intelligence Lab tab, so every claim can be traced to a chart there.\n"
            "- **Reported results** are the Siemens Gamesa segment figures in Siemens Energy's quarterly earnings releases (EUR m, fiscal year October-September).\n"
            "- **Announced MW** is the workbook's order list; it covers only the orders Siemens Gamesa, customers or the trade press announce individually.\n"
            "- **Scenarios** combine reported year-to-date figures with the recent range of quarters and the company's guidance; they are not forecasts.\n"
            "- Fiscal-year economics from the workbook (net income, backlog split, installed MW) end in FY2025 or earlier."
        )


# ======================================================================================
# Intelligence Lab
# ======================================================================================
def render_intelligence_lab(intel: Intel, orders_f: pd.DataFrame, platforms_f: pd.DataFrame) -> None:
    inject_css()
    st.markdown(
        '<div class="intel-hero"><h2>Intelligence Lab</h2><p class="intel-sub">Deeper analytics derived from reported results, the announced order list and the workbook economics.</p></div>',
        unsafe_allow_html=True,
    )
    if hasattr(st, "segmented_control"):
        choice = st.segmented_control("Section", LAB_SECTIONS, default=LAB_SECTIONS[0], key="intel_lab_section", label_visibility="collapsed")
    else:  # older Streamlit
        choice = st.radio("Section", LAB_SECTIONS, horizontal=True, key="intel_lab_section", label_visibility="collapsed")
    choice = choice or LAB_SECTIONS[0]

    if choice == "Momentum & Outlook":
        _section_momentum(intel)
    elif choice == "Market Structure":
        _section_structure(intel, orders_f, platforms_f)
    elif choice == "Accounts & Service":
        _section_accounts_service(intel, orders_f, platforms_f)
    elif choice == "Compare Lab":
        _section_compare(intel)
    else:
        _section_quality(intel)


# --------------------------------------------------------------------------------------
def _section_momentum(intel: Intel) -> None:
    _scope_note("Company-level section: uses the full dataset and ignores the sidebar filters.")
    pr = intel.projection
    seg = intel.seg

    st.subheader("Reported results by fiscal quarter")
    c1, c2 = st.columns(2)
    with c1:
        show(co.fig_segment_quarterly(intel), "lab_seg")
    with c2:
        show(co.fig_segment_profit(intel), "lab_profit")
    c3, c4 = st.columns(2)
    with c3:
        show(co.fig_backlog_trend(intel), "lab_backlog")
    with c4:
        show(co.fig_revenue_mix(intel), "lab_revmix")
    cols = ["label", "orders", "revenue", "book_to_bill", "profit_before_si", "margin_before_si", "revenue_service", "service_share", "backlog_bn", "ltm_orders", "ltm_revenue", "ltm_book_to_bill"]
    view = seg[cols].rename(columns={"label": "Quarter"}).set_index("Quarter")
    table_view(view.round(3), "Data table: reported segment results (EUR m)")
    notes = seg[seg["note"].astype(str).str.len() > 0][["label", "note"]].rename(columns={"label": "Quarter", "note": "What drove it (from the release)"})
    if len(notes):
        with st.expander("What drove each quarter"):
            st.dataframe(notes, width="stretch", hide_index=True)

    st.subheader("Fiscal-year outlook")
    show(co.fig_fy_orders(intel), "lab_fy")
    if pr:
        sc, qo, g = pr["orders_scenario"], pr["quarter_orders"], pr["guidance"]
        with st.expander("Outlook working"):
            rows = [
                ("Orders reported to date", eur(pr["orders_ytd"]), f"same quarters last year: {eur(pr['orders_prev_ytd'])}"),
                ("Remaining-quarter scenario (low / median / high)", f"{eur(qo['low'])} / {eur(qo['median'])} / {eur(qo['high'])}", f"from the last {qo['lookback']} reported quarters"),
                ("Full-year orders (conservative / base / strong)", f"{eur(sc['conservative'])} / {eur(sc['base'])} / {eur(sc['strong'])}", f"last fiscal year: {eur(pr['orders_prev_full'])}"),
            ]
            if pr.get("revenue_full_year_guided"):
                lo, hi = pr["revenue_full_year_guided"]
                rows.append(("Revenue implied by guided growth", f"{eur(lo)} - {eur(hi)}", f"{g['revenue_growth_comparable'][0]:.0%}-{g['revenue_growth_comparable'][1]:.0%} on last year's nominal revenue ({eur(pr['revenue_prev_full'])}); comparable basis excludes currency and portfolio effects"))
                rows.append(("Revenue needed in the remaining quarter", f"{eur(pr['revenue_needed_remaining'][0])} - {eur(pr['revenue_needed_remaining'][1])}", f"year-to-date {eur(pr['revenue_ytd'])}; last quarter {eur(pr['last_quarter_revenue'])}"))
            if pr.get("pbsi_needed_remaining"):
                rows.append(("Profit before special items needed for break-even", eur(pr["pbsi_needed_remaining"][0]), f"year-to-date {eur(pr['pbsi_ytd'])}; last quarter {eur(pr['last_quarter_pbsi'])}"))
            st.dataframe(pd.DataFrame(rows, columns=["Item", "Value", "Basis"]), width="stretch", hide_index=True)
            st.caption(f"Guidance as of: {g.get('as_of', 'n/a')}. Orders have no company guidance; the scenario simply repeats recent quarters.")
        tdt = intel.to_date
        if not tdt.empty:
            table_view(tdt.round(3), f"Data table: first {int(tdt['quarters'].iloc[0])} quarters of each fiscal year")

    st.subheader("Announced orders against reported order intake")
    show(co.fig_announced_vs_orders(intel), "lab_annvs")
    fq = intel.ann_fq.copy()
    fq["eur_per_announced_mw"] = fq["eur_per_announced_mw"].round(2)
    table_view(fq, "Data table: announced MW and reported orders per fiscal quarter")
    st.caption(
        "Reported orders are in EUR and include onshore, offshore and service; announced MW covers only individually disclosed orders and is dated by announcement, "
        "which trails the booking by a quarter or more (the large offshore orders booked in Q3 FY25 were announced in Q4). The ratio is shown only when at least 500 MW were announced; "
        "treat it as a rough yardstick (about EUR 3m per announced MW across FY24-FY25)."
    )

    st.subheader("Announced orders by calendar year")
    show(co.fig_cumulative_curves(intel, height=480, width_hint=1150), "lab_cum")
    qe = intel.qe_overall
    c5, c6 = st.columns(2)
    with c5:
        show(co.fig_intra_quarter(intel), "lab_intraq")
    with c6:
        show(co.fig_quarter_end_share(intel), "lab_qeshare")
    c7, c8 = st.columns(2)
    with c7:
        show(co.fig_quarter_heatmap(intel), "lab_qheat")
    with c8:
        ms = intel.month_share
        mt = pd.DataFrame({"Month of quarter": ["1st month", "2nd month", "3rd month"], "Share of announced MW": [pc(ms.get(1, np.nan)), pc(ms.get(2, np.nan)), pc(ms.get(3, np.nan))]})
        st.markdown("**Which month of the quarter?**")
        st.dataframe(mt, width="stretch", hide_index=True)
        st.markdown(
            f"{pc(qe['mw_share'])} of announced MW falls in the last {qe['window_days']} days of a quarter and {pc(qe['early_mw_share'])} in the first {qe['window_days']} "
            f"(an even spread gives {pc(qe['expected_if_uniform'])} each). Based on {qe['orders']} orders over {qe['years']} years, and large offshore orders dominate, so the pattern is weak evidence."
        )

    st.subheader("Fiscal-year economics")
    if intel.econ.empty:
        st.info("Economy data not available.")
        return
    c9, c10 = st.columns(2)
    with c9:
        show(co.fig_book_to_bill(intel), "lab_b2b")
    with c10:
        show(co.fig_margins(intel), "lab_margin")
    c11, c12 = st.columns(2)
    with c11:
        show(co.fig_service_share_fy(intel), "lab_svcfy")
    with c12:
        show(co.fig_offshore_orders_share(intel), "lab_orderbiz")
    cols_view = ["revenue", "revenue_growth", "net_profit", "net_margin", "profit_before_si", "margin_before_si", "order_intake", "book_to_bill", "service_share", "service_growth", "offshore_order_share", "backlog_bn", "cover_years"]
    table_view(intel.econ[[c for c in cols_view if c in intel.econ]].dropna(how="all").round(3), "Data table: derived economics by fiscal year")
    st.caption("Net income includes special items and the 2023 turbine-quality provisions; margin before special items comes from the segment results (FY24 onward). The workbook's FY table has no FY26 column.")


# --------------------------------------------------------------------------------------
def _section_structure(intel: Intel, orders_f: pd.DataFrame, platforms_f: pd.DataFrame) -> None:
    _scope_note("This section respects the sidebar filters (year range, geography, platform, minimum MW). It covers announced orders only.")
    o = prepare_orders(orders_f)
    p = prepare_platforms(platforms_f)
    if o.empty:
        st.info("No orders after filtering.")
        return
    as_of = intel.as_of

    st.subheader("Order size and lumpiness")
    conc = structure.concentration_by_year(o, as_of)
    mode = st.radio("Size-class view", ["Absolute MW", "Share of MW"], horizontal=True, key="str_size_mode")
    c1, c2 = st.columns(2)
    with c1:
        show(cs.fig_top_share(conc), "str_top")
    with c2:
        show(cs.fig_size_class_mix(o, share=(mode == "Share of MW")), "str_size")
    st.markdown("**Largest orders in scope**")
    st.dataframe(structure.largest_orders(o, 15), width="stretch", hide_index=True)
    table_view(conc[["orders", "mw", "median_mw", "top1_share", "top5_share", "top10_share", "mega_orders", "mega_share", "giga_orders"]].round(3), "Data table: concentration by year")
    st.caption(f"Mega orders are {MEGA_MW:,.0f} MW or more; jumbo orders {GIGA_MW:,.0f} MW or more. With only 3-5 orders a year since 2023, shares such as 'top 5' are 100% by construction.")

    st.subheader("Product mix")
    mix = structure.mix_by_year(p)
    rating_mix = structure.rating_bucket_mix(p)
    series = structure.series_mix(p)
    if mix.empty:
        st.info("No platform-level rows for the current filters.")
    else:
        c3, c4 = st.columns(2)
        with c3:
            show(cs.fig_rating_mix(rating_mix), "str_ratingmix")
        with c4:
            show(cs.fig_series_mix(series), "str_series")
        show(cs.fig_rating_rotor(mix), "str_ratingrotor")
        years = sorted(int(y) for y in p.loc[p["slot_mw"].fillna(0) > 0, "order_year"].unique())
        pick = st.selectbox("Platforms ordered in", years, index=len(years) - 1, key="str_plat_year")
        pm = structure.platform_mw(p, int(pick), 12)
        if len(pm):
            show(cs.fig_platform_mw(pm, f"Platforms by ordered MW, {pick}"), "str_platforms")
        lead = structure.leading_platforms(p, 3)
        if not lead.empty:
            lt = lead.pivot(index="year", columns="rank", values="platform").rename(columns=lambda r: f"#{r} platform")
            sh = lead.pivot(index="year", columns="rank", values="share").rename(columns=lambda r: f"#{r} share")
            tbl = pd.concat([lt, sh.map(lambda v: pc(v))], axis=1)
            tbl = tbl[[c for r in (1, 2, 3) for c in (f"#{r} platform", f"#{r} share") if c in tbl.columns]]
            st.markdown("**Leading platforms by order year** (share of platform-mapped MW)")
            st.dataframe(tbl.sort_index(ascending=False), width="stretch")
        st.caption(
            "Series are classified from the platform name: offshore direct drive = SG 8.0-167 DD, 11-200 DD, 14-222 DD, 14-236 DD and other DD models; offshore geared = SWT 3.6-7 MW; "
            "SG 7.0 = the restarted onshore platform (successor to the 5.X); SG 4.X-6.X = the paused 4.X/5.X/6.X family. Orders with no platform are left out."
        )
        table_view(series.round(3), "Data table: product series share by order year")

    st.subheader("Geography")
    window = st.radio("Compare", ["Last 12 months vs the 12 before", "Last full year vs the year before"], horizontal=True, key="str_geo_window")
    if window.startswith("Last 12"):
        cur = (as_of - pd.DateOffset(years=1), as_of)
        prior = (as_of - pd.DateOffset(years=2), as_of - pd.DateOffset(years=1))
        title = f"Country movers: last 12 months vs prior 12 months (to {as_of:%d %b %Y})"
    else:
        y = as_of.year - 1
        cur = (pd.Timestamp(y - 1, 12, 31), pd.Timestamp(y, 12, 31))
        prior = (pd.Timestamp(y - 2, 12, 31), pd.Timestamp(y - 1, 12, 31))
        title = f"Country movers: {y} vs {y - 1}"
    pcmp = structure.period_compare(o, cur, prior)
    c5, c6 = st.columns([3, 2])
    with c5:
        if pcmp["delta_mw"].abs().sum() == 0:
            st.info("No country-level orders in these windows for the current filters.")
        else:
            show(cs.fig_diverging(pcmp["delta_mw"], 10, title), "str_movers")
    with c6:
        show(cs.fig_continent_mix(o), "str_continent")
    table_view(pcmp.sort_values("delta_mw", ascending=False).round(3), "Data table: country movers")
    c7, c8 = st.columns(2)
    markets = structure.market_status(o, as_of)
    with c7:
        show(cs.fig_breadth(conc), "str_breadth")
    with c8:
        if not markets.empty:
            show(cs.fig_market_status(markets, 16), "str_markets")
    if not markets.empty:
        counts = markets["status"].value_counts()
        st.caption(
            "Market status by months since last order: Active <= 12, Cooling 13-36, Dormant > 36; Emerging = first order within the last 5 years. "
            + ", ".join(f"{k}: {v}" for k, v in counts.items())
            + ". Orders without a country are excluded; England is folded into United Kingdom."
        )
        table_view(markets.round(1), "Data table: market lifecycle")


# --------------------------------------------------------------------------------------
def _section_accounts_service(intel: Intel, orders_f: pd.DataFrame, platforms_f: pd.DataFrame) -> None:
    _scope_note("This section respects the sidebar filters. Customers are undisclosed for almost all 2017-2019 MW, so account history starts in practice around 2020.")
    o = prepare_orders(orders_f)
    if o.empty:
        st.info("No orders after filtering.")
        return
    as_of = intel.as_of

    st.subheader("Account radar")
    radar = structure.account_radar(o, as_of)
    if radar.empty:
        st.info("No disclosed customers in the filtered data.")
    else:
        min_mw = st.slider("Minimum lifetime MW for an account to appear", 50, 2000, 300, step=50, key="acc_min_mw")
        key_accounts = radar[(radar["lifetime_mw"] >= min_mw) & radar["repeat_buyer"]]
        summ = structure.account_status_summary(radar, float(min_mw), repeat_only=True)
        m = st.columns(4)
        for col, status in zip(m[:3], ["Active", "Cooling", "Dormant"]):
            n = int(summ.loc[status, "accounts"]) if status in summ.index else 0
            mw = float(summ.loc[status, "lifetime_mw"]) if status in summ.index else 0.0
            col.metric(f"{status} repeat accounts", f"{n}", f"{mw / 1000:.1f} GW lifetime", delta_color="off", delta_arrow="off")
        und = float(o.loc[o["customer"] == UNKNOWN, "size_mw"].sum() / max(o["size_mw"].sum(), 1))
        m[3].metric("MW with undisclosed customer", pc(und), "limits account analytics", delta_color="off", delta_arrow="off")
        show(cs.fig_account_radar(radar, float(min_mw)), "acc_radar")
        st.caption("Status by time since the last announced order: Active <= 12 months, Cooling 12-24, Dormant > 24. Only disclosed customers are included, and the order list is thin since 2023, so many accounts look dormant.")

        c1, c2 = st.columns([2, 3])
        with c1:
            delta = radar.set_index("customer")["ltm_delta_mw"]
            show(cs.fig_diverging(delta, 8, "Account movers: last 12 months vs prior 12 months", height=520), "acc_movers")
        with c2:
            st.markdown("**Re-engagement watchlist** - repeat accounts that have gone quiet")
            status_pick = st.multiselect("Status", ["Cooling", "Dormant", "Active"], default=["Cooling", "Dormant"], key="acc_status_pick")
            wl = key_accounts[key_accounts["status"].isin(status_pick)].copy().sort_values("lifetime_mw", ascending=False)
            view = wl[["customer", "status", "orders", "lifetime_mw", "last_order", "days_since_last", "median_gap_days", "top_country"]].rename(
                columns={"customer": "Customer", "status": "Status", "orders": "Orders", "lifetime_mw": "Lifetime MW", "last_order": "Last order", "days_since_last": "Days since",
                         "median_gap_days": "Typical gap (days)", "top_country": "Main country"}
            )
            st.dataframe(view.round(0), width="stretch", hide_index=True, height=470)
            _csv_button(view, "Download watchlist (CSV)", "sgre_account_watchlist.csv", "acc_dl")

    st.subheader("Service contracts")
    service = structure.service_by_year(o)
    c3, c4 = st.columns(2)
    with c3:
        show(cs.fig_service_disclosure(service), "svc_disc")
    with c4:
        show(cs.fig_tenure(service), "svc_tenure")
    st.caption("The workbook records the contract length (years) but not a service scheme; lengths are stated for under half of the MW. Reported service revenue and its growth are on the Momentum & Outlook section.")
    table_view(service.round(3), "Data table: service contract length by order year")


# --------------------------------------------------------------------------------------
def _section_compare(intel: Intel) -> None:
    _scope_note("Head-to-head comparison uses the full dataset and ignores the sidebar filters.")
    o, p, as_of = intel.orders, intel.platforms, intel.as_of
    st.subheader("Compare Lab")
    dim = st.radio("Compare", structure.COMPARE_DIMENSIONS, horizontal=True, key="cmp_dim")
    options = structure.entity_options(o, p, dim)
    if len(options) < 2:
        st.info("Not enough entities to compare.")
        return
    ca, cb = st.columns(2)
    a = ca.selectbox("Side A", options, index=0, key=f"cmp_a_{dim}")
    b = cb.selectbox("Side B", options, index=1, key=f"cmp_b_{dim}")
    pa = structure.entity_profile(o, p, dim, a, as_of)
    pb = structure.entity_profile(o, p, dim, b, as_of)

    def fmt(profile: dict) -> dict[str, str]:
        return {
            "Announced MW (2017+)": f"{profile['mw']:,.0f}",
            "Share of total MW": pc(profile["share_of_total"], 1),
            "Orders": f"{profile['orders']:,}",
            "Average order (MW)": f"{profile['avg_order_mw']:,.0f}",
            "Largest order (MW)": f"{profile['largest_order_mw']:,.0f}",
            "Last 12 months (MW)": f"{profile['ltm_mw']:,.0f}",
            "Prior 12 months (MW)": f"{profile['prior_ltm_mw']:,.0f}",
            "12-month change": pc(profile["ltm_growth"], 0, signed=True),
            "Offshore share of MW": pc(profile["offshore_share"]),
            "Avg turbine rating (MW)": "n/a" if pd.isna(profile["avg_rating_mw"]) else f"{profile['avg_rating_mw']:.1f}",
            "Avg rotor (m)": "n/a" if pd.isna(profile["avg_rotor_m"]) else f"{profile['avg_rotor_m']:.0f}",
            "Service contract (years, MW-weighted)": "n/a" if pd.isna(profile["service_tenure_years"]) else f"{profile['service_tenure_years']:.1f}",
            "Median planned delivery (days)": "n/a" if pd.isna(profile["median_delivery_days"]) else f"{profile['median_delivery_days']:,.0f}",
            "First order": str(profile["first_order"]),
            "Last order": str(profile["last_order"]),
            "Active order years": f"{profile['active_years']}",
            "Countries": f"{profile['countries']}",
            "Named customers": f"{profile['customers']}",
        }

    table = pd.DataFrame({a: fmt(pa), b: fmt(pb)})
    table.index.name = "Metric"
    st.dataframe(table.reset_index(), width="stretch", height=640, hide_index=True, column_config={"Metric": st.column_config.TextColumn("Metric", width="large")})

    c1, c2 = st.columns(2)
    with c1:
        show(cs.fig_compare_yearly(structure.entity_yearly(o, p, dim, a), structure.entity_yearly(o, p, dim, b)), "cmp_yearly")
    with c2:
        by_options = [x for x in ["Country", "Customer", "Platform family"] if x != dim]
        by = st.selectbox("Break down by", by_options, index=0, key=f"cmp_by_{dim}")
        show(cs.fig_compare_mix(structure.entity_breakdown(o, p, dim, a, by), structure.entity_breakdown(o, p, dim, b, by), a, b, by), "cmp_mix")
    st.caption("Platform-family comparisons count platform-slot MW; other comparisons count order MW. 'Other' groups the tail beyond the top eight.")


# --------------------------------------------------------------------------------------
SEVERITY_LABEL = {"High": "▲ High", "Medium": "● Medium", "Low": "○ Low", "OK": "✓ OK", "Info": "i Info"}


def _section_quality(intel: Intel) -> None:
    _scope_note("Data-quality section: full dataset, ignoring the sidebar filters.")
    st.subheader("Data quality and reconciliation")
    summary = intel.anomalies.copy()
    m = st.columns(4)
    m[0].metric("High-severity checks", int((summary["Severity"] == "High").sum()), delta_color="off", delta_arrow="off")
    m[1].metric("Medium", int((summary["Severity"] == "Medium").sum()), delta_color="off", delta_arrow="off")
    m[2].metric("Passed / informational", int(summary["Severity"].isin(["OK", "Info"]).sum()), delta_color="off", delta_arrow="off")
    res = intel.results or {}
    m[3].metric("Orders last checked", str(res.get("orders_checked_through", "n/a")), delta_color="off", delta_arrow="off")

    summary["Severity"] = summary["Severity"].map(lambda s: SEVERITY_LABEL.get(s, s))
    st.table(summary.set_index("Check"))
    if res.get("orders_check_note"):
        st.caption(res["orders_check_note"])

    c1, c2 = st.columns(2)
    with c1:
        show(co.fig_announced_vs_orders(intel), "dq_annvs")
    with c2:
        show(cs.fig_completeness_heatmap(intel.completeness), "dq_complete")
    view = intel.ann_fy.copy()
    view["eur_per_announced_mw"] = view["eur_per_announced_mw"].round(2)
    st.markdown("**Announced orders vs reported order intake by fiscal year**")
    st.dataframe(view, width="stretch")
    st.caption(
        "The workbook's fiscal-year order intake (EUR m) is only available for FY2016 onward here, and the reported series is in EUR, not MW. "
        "The ratio is shown only for years with at least 500 announced MW."
    )
    if intel.results:
        with st.expander("Segment results used (EUR m)"):
            st.dataframe(intel.seg[["label", "orders", "revenue", "profit_before_si", "profit", "revenue_new_units", "revenue_service", "backlog_bn", "basis"]].rename(columns={"label": "Quarter"}), width="stretch", hide_index=True)
            st.caption(f"Source: {intel.results['source']} (retrieved {intel.results['retrieved']}). {intel.results['notes']}")

    details = quality.anomaly_rows(intel.orders)
    if details:
        st.markdown("**Record-level detail**")
        for name, frame in details.items():
            with st.expander(f"{name} ({len(frame)})"):
                st.dataframe(frame, width="stretch", hide_index=True)

    with st.expander("Definitions and method notes"):
        st.markdown(
            "- **Fiscal year**: October-September (FY2026 = Oct 2025 - Sep 2026); Q1 = Oct-Dec. Calendar-year charts of announced MW use calendar quarters.\n"
            "- **Reported orders / revenue / profit before special items**: Siemens Gamesa segment in Siemens Energy's quarterly earnings releases.\n"
            "- **Announced MW**: order sizes from the yearly sheets, dated by announcement. Offshore/onshore is derived from the platform name; orders without a platform stay unclassified.\n"
            f"- **Mega / jumbo orders**: {MEGA_MW:,.0f} MW and above / {GIGA_MW:,.0f} MW and above.\n"
            "- **Key account**: a disclosed customer with at least 2 order dates and the chosen minimum lifetime MW.\n"
            "- **Book-to-bill**: orders divided by revenue in the same period; **cover**: order backlog divided by the last four quarters' revenue.\n"
            "- **Scenarios**: not forecasts; see the outlook working on the Momentum & Outlook section."
        )
