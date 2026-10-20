"""Charts for the Executive Briefing and the Momentum & Outlook section."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import outlook
from .bundle import Intel
from .common import GIGA_MW
from .theme import style_figure, tokens, year_axis


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _bar(t: dict, color: str | None = None) -> dict:
    return dict(color=color or t["blue"], line=dict(color=t["surface"], width=2))


def _line(t: dict, color: str, width: float = 2.4) -> dict:
    return dict(color=color, width=width)


# --------------------------------------------------------------------------------------
# Reported segment results
# --------------------------------------------------------------------------------------
def fig_segment_quarterly(intel: Intel, height: int = 440) -> go.Figure:
    """Reported orders and revenue per fiscal quarter, with the four-quarter book-to-bill."""
    t = tokens()
    s = intel.seg
    fig = go.Figure()
    fig.add_trace(go.Bar(x=s["label"], y=s["orders"], name="Orders", marker=_bar(t), hovertemplate="%{x}: EUR %{y:,.0f}m orders<extra></extra>"))
    fig.add_trace(go.Bar(x=s["label"], y=s["revenue"], name="Revenue", marker=_bar(t, t["neutral"]), hovertemplate="%{x}: EUR %{y:,.0f}m revenue<extra></extra>"))
    style_figure(fig, title="Reported orders and revenue per fiscal quarter (EUR m)", height=height, ytitle="EUR m", hovermode="x unified")
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.06)
    fig.update_xaxes(type="category", tickangle=-45)
    return fig


def fig_segment_profit(intel: Intel, height: int = 400) -> go.Figure:
    t = tokens()
    s = intel.seg
    colors = [t["down"] if v < 0 else t["up"] for v in s["profit_before_si"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=s["label"], y=s["profit_before_si"], name="Profit before special items", marker=dict(color=colors, line=dict(color=t["surface"], width=2)),
                         customdata=s["margin_before_si"], hovertemplate="%{x}: EUR %{y:,.0f}m (%{customdata:.1%})<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=t["ink2"], width=1.2))
    style_figure(fig, title="Profit before special items per fiscal quarter (EUR m)", height=height, ytitle="EUR m", legend=False, hovermode="x unified")
    fig.update_layout(bargap=0.3)
    fig.update_xaxes(type="category", tickangle=-45)
    return fig


def fig_backlog_trend(intel: Intel, height: int = 400) -> go.Figure:
    t = tokens()
    s = intel.seg.dropna(subset=["backlog_bn"])
    fig = go.Figure()
    fig.add_trace(go.Bar(x=s["label"], y=s["backlog_bn"], name="Order backlog (EUR bn)", marker=_bar(t), hovertemplate="%{x}: EUR %{y:.0f}bn<extra>Backlog</extra>"))
    fig.add_trace(go.Scatter(x=s["label"], y=s["cover_years"], name="Cover (years of last-4-quarter revenue)", yaxis="y2", mode="lines+markers", line=_line(t, t["orange"]),
                             marker=dict(size=8, line=dict(color=t["surface"], width=2)), hovertemplate="%{x}: %{y:.1f} years<extra>Cover</extra>"))
    style_figure(fig, title="Order backlog (end of quarter)", height=height, ytitle="EUR bn", hovermode="x unified")
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False, title=dict(text="Years", font=dict(color=t["muted"], size=11)), rangemode="tozero"))
    fig.update_xaxes(type="category", tickangle=-45)
    fig.update_yaxes(rangemode="tozero")
    return fig


def fig_revenue_mix(intel: Intel, height: int = 400) -> go.Figure:
    """New-unit vs service revenue per quarter with the service share."""
    t = tokens()
    s = intel.seg
    fig = go.Figure()
    fig.add_trace(go.Bar(x=s["label"], y=s["revenue_new_units"], name="New units", marker=_bar(t), hovertemplate="%{x}: EUR %{y:,.0f}m<extra>New units</extra>"))
    fig.add_trace(go.Bar(x=s["label"], y=s["revenue_service"], name="Service", marker=_bar(t, t["orange"]), hovertemplate="%{x}: EUR %{y:,.0f}m<extra>Service</extra>"))
    fig.add_trace(go.Scatter(x=s["label"], y=s["service_share"], name="Service share (right axis)", yaxis="y2", mode="lines+markers", line=_line(t, t["ink"], 2.0),
                             marker=dict(size=7, line=dict(color=t["surface"], width=2)), hovertemplate="%{x}: %{y:.0%}<extra>Service share</extra>"))
    style_figure(fig, title="Revenue split: new units and service (EUR m)", height=height, ytitle="EUR m", hovermode="x unified")
    fig.update_layout(barmode="stack", bargap=0.25, yaxis2=dict(overlaying="y", side="right", showgrid=False, tickformat=".0%", rangemode="tozero"))
    fig.update_xaxes(type="category", tickangle=-45)
    return fig


def fig_fy_orders(intel: Intel, height: int = 440) -> go.Figure:
    """Reported orders per fiscal year (workbook totals, then the segment series), with the running year and its scenario."""
    t = tokens()
    pr = intel.projection
    fy_now = pr["fy"] if pr else int(intel.fy.index.max())
    wb = intel.econ["order_intake"].dropna()
    wb = wb[wb.index < fy_now]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=wb.index, y=wb, name="Orders (fiscal year)", marker=_bar(t), hovertemplate="FY%{x}: EUR %{y:,.0f}m<extra></extra>"))
    if pr:
        ytd = pr["orders_ytd"]
        base = pr["orders_scenario"]["base"]
        fig.add_trace(go.Bar(x=[fy_now], y=[ytd], name=f"FY{fy_now % 100} reported to date", marker=_bar(t, t["ink2"]), hovertemplate=f"FY{fy_now}: EUR " + "%{y:,.0f}m to date<extra></extra>"))
        fig.add_trace(go.Bar(x=[fy_now], y=[max(base - ytd, 0)], name="Base scenario for the remaining quarter",
                             marker=dict(color=rgba(t["ink2"], 0.30), line=dict(color=t["ink2"], width=1.5), pattern=dict(shape="/", fgcolor=t["ink2"], size=7)),
                             hovertemplate="Base scenario: EUR %{y:,.0f}m<extra></extra>"))
        sc = pr["orders_scenario"]
        fig.add_trace(go.Scatter(x=[fy_now], y=[sc["base"]], mode="markers", name="Scenario range (low / base / high)",
                                 marker=dict(symbol="diamond-open", size=11, color=t["ink"], line=dict(width=2, color=t["ink"])),
                                 error_y=dict(type="data", symmetric=False, array=[sc["strong"] - sc["base"]], arrayminus=[sc["base"] - sc["conservative"]], color=t["ink"], thickness=1.6, width=7),
                                 hovertemplate=f"Base EUR {sc['base']:,.0f}m (range {sc['conservative']:,.0f}-{sc['strong']:,.0f})<extra></extra>"))
    style_figure(fig, title="Reported order intake by fiscal year (EUR m)", height=height, ytitle="EUR m", hovermode="x unified")
    fig.update_layout(barmode="stack", bargap=0.28)
    year_axis(fig)
    return fig


def fig_announced_vs_orders(intel: Intel, first_fy: int = 2018, height: int = 440) -> go.Figure:
    """Announced MW per fiscal year (offshore / onshore) next to reported order intake in EUR."""
    t = tokens()
    a = intel.ann_fy.loc[first_fy:]
    unk = (a["announced_mw"] - a["offshore_mw"] - a["onshore_mw"]).clip(lower=0)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=a.index, y=a["offshore_mw"], name="Announced offshore MW", marker=_bar(t), hovertemplate="FY%{x}: %{y:,.0f} MW<extra>Offshore</extra>"))
    fig.add_trace(go.Bar(x=a.index, y=a["onshore_mw"], name="Announced onshore MW", marker=_bar(t, t["orange"]), hovertemplate="FY%{x}: %{y:,.0f} MW<extra>Onshore</extra>"))
    fig.add_trace(go.Bar(x=a.index, y=unk, name="Announced, platform unknown", marker=_bar(t, t["neutral"]), hovertemplate="FY%{x}: %{y:,.0f} MW<extra>Platform unknown</extra>"))
    rep = intel.econ["order_intake"].reindex(a.index)
    fig.add_trace(go.Scatter(x=a.index, y=rep, name="Reported orders, EUR m (right axis)", yaxis="y2", mode="lines+markers", line=_line(t, t["ink"], 2.2),
                             marker=dict(size=8, symbol="diamond-open", line=dict(width=2, color=t["ink"])), hovertemplate="FY%{x}: EUR %{y:,.0f}m<extra>Reported orders</extra>"))
    style_figure(fig, title="Announced MW by fiscal year against reported order intake", height=height, ytitle="MW", hovermode="x unified")
    fig.update_layout(barmode="stack", bargap=0.28, yaxis2=dict(overlaying="y", side="right", showgrid=False, rangemode="tozero", title=dict(text="EUR m", font=dict(color=t["muted"], size=11))))
    year_axis(fig)
    return fig


# --------------------------------------------------------------------------------------
# Announced orders (calendar years)
# --------------------------------------------------------------------------------------
def fig_cumulative_curves(intel: Intel, first_year: int = 2017, height: int = 470, width_hint: int = 540) -> go.Figure:
    """Cumulative announced MW by calendar day: the current year against every earlier year."""
    t = tokens()
    year = intel.as_of.year
    curves = outlook.cumulative_curves(intel.orders, intel.as_of, first_year)
    past_years = [int(y) for y in sorted(curves["year"].unique()) if y != year]
    finals = curves[curves["year"] != year].groupby("year")["cum_mw"].max()
    label_years = set(finals.nlargest(2).index.tolist()) | {year - 1}
    fig = go.Figure()
    first_gray = True
    for y in past_years:
        c = curves[curves["year"] == y]
        prev = bool(y == year - 1)
        fig.add_trace(go.Scatter(
            x=c["plot_date"], y=c["cum_mw"], mode="lines", name=str(y) if prev else f"Earlier years ({first_year}-{year - 2})", legendgroup="prev" if prev else "earlier",
            showlegend=prev or first_gray, line=dict(color=t["orange"] if prev else t["neutral"], width=2.4 if prev else 1.2), opacity=1.0 if prev else 0.55,
            hovertemplate=f"{y}: " + "%{y:,.0f} MW at %{x|%d %b}<extra></extra>",
        ))
        if not prev:
            first_gray = False
        if y in label_years:
            fig.add_annotation(x=c["plot_date"].iloc[-1], y=c["cum_mw"].iloc[-1], text=str(y), showarrow=False, xanchor="left", xshift=4, font=dict(size=10, color=t["muted"]))
    cur = curves[curves["year"] == year]
    fig.add_trace(go.Scatter(x=cur["plot_date"], y=cur["cum_mw"], mode="lines", name=str(year), line=dict(color=t["blue"], width=3.6),
                             hovertemplate=f"{year}: " + "%{y:,.0f} MW at %{x|%d %b}<extra></extra>"))
    big = intel.orders[(intel.orders["order_year"] == year) & (intel.orders["size_mw"] >= GIGA_MW)]
    if len(big):
        pts = cur.set_index("date").reindex(big["order_date"].dt.normalize())
        fig.add_trace(go.Scatter(x=pts["plot_date"], y=pts["cum_mw"], mode="markers", name=f"Orders of {GIGA_MW:,.0f} MW+ in {year}",
                                 marker=dict(size=11, color=t["blue"], line=dict(color=t["surface"], width=2.5), symbol="diamond"),
                                 text=[f"{c} - {m:,.0f} MW" for c, m in zip(big["customer"], big["size_mw"])], hovertemplate="%{text}<br>%{x|%d %b}<extra></extra>"))
    style_figure(fig, title=f"Cumulative announced MW: {year} (to {intel.as_of:%d %b}) against every year since {first_year}", height=height, ytitle="Cumulative MW", hovermode="closest", width_hint=width_hint)
    fig.update_xaxes(tickformat="%b", range=[pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-25")])
    return fig


def fig_intra_quarter(intel: Intel, height: int = 400) -> go.Figure:
    t = tokens()
    prof = outlook.intra_quarter_profile(intel.orders, intel.as_of)
    fig = go.Figure()
    if prof.empty:
        return style_figure(fig, title="Order timing within a quarter", height=height)
    fig.add_trace(go.Scatter(x=prof["day"], y=prof["p75"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=prof["day"], y=prof["p25"], mode="lines", line=dict(width=0), fill="tonexty", fillcolor=rgba(t["blue"], 0.18),
                             name="Middle half of quarters (25th-75th pct)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=prof["day"], y=prof["uniform"], mode="lines", name="Even spread", line=dict(color=t["neutral"], width=1.6), hovertemplate="Even spread: %{y:.0%}<extra></extra>"))
    fig.add_trace(go.Scatter(x=prof["day"], y=prof["mean"], mode="lines", name="Average quarter", line=dict(color=t["blue"], width=3),
                             hovertemplate="Day %{x}: %{y:.0%} of the quarter's MW announced<extra></extra>"))
    n = int(prof["quarters"].iloc[0])
    style_figure(fig, title=f"Within a quarter ({n} quarters with announced orders since 2017)", height=height, xtitle="Day of quarter", ytitle="Cumulative share of quarter MW", hovermode="x unified", yformat=".0%")
    fig.update_yaxes(range=[0, 1.02])
    return fig


def fig_quarter_end_share(intel: Intel, first_year: int = 2017, height: int = 400) -> go.Figure:
    t = tokens()
    s = intel.qe_by_year.loc[first_year:, "mw_share"]
    exp = intel.qe_overall["expected_if_uniform"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=s.index, y=s, name="Share of year's MW", marker=_bar(t), hovertemplate="%{x}: %{y:.0%}<extra></extra>"))
    fig.add_trace(go.Scatter(x=s.index, y=[exp] * len(s), mode="lines", name="Even-spread expectation", line=dict(color=t["neutral"], width=1.6), hovertemplate="Even spread: %{y:.0%}<extra></extra>"))
    style_figure(fig, title=f"Share of each year's MW announced in the last {intel.qe_overall['window_days']} days of a quarter", height=height, ytitle="Share of MW", yformat=".0%", hovermode="x unified")
    year_axis(fig)
    fig.update_yaxes(range=[0, 1.0])
    return fig


def fig_quarter_heatmap(intel: Intel, first_year: int = 2017, height: int = 400) -> go.Figure:
    t = tokens()
    q = intel.quarter_share.loc[first_year:]
    fig = go.Figure(
        go.Heatmap(
            z=q.to_numpy(), x=list(q.columns), y=[str(i) for i in q.index], colorscale=t["seq"], zmin=0, zmax=0.8,
            text=np.where(q.notna(), (q * 100).round(0).astype("Int64").astype(str) + "%", ""), texttemplate="%{text}",
            textfont=dict(size=11), hovertemplate="%{y} %{x}: %{z:.0%} of the year's announced MW<extra></extra>", colorbar=dict(title="", tickformat=".0%", thickness=10), xgap=2, ygap=2,
        )
    )
    style_figure(fig, title="Calendar-quarter share of each year's announced MW", height=height, hovermode="closest", legend=False)
    fig.update_yaxes(autorange="reversed", showgrid=False, type="category")
    return fig


# --------------------------------------------------------------------------------------
# Fiscal-year economics
# --------------------------------------------------------------------------------------
def fig_book_to_bill(intel: Intel, first_year: int = 2016, height: int = 400) -> go.Figure:
    t = tokens()
    b = intel.econ.loc[first_year:, "book_to_bill"].dropna()
    fig = go.Figure(go.Bar(x=b.index, y=b, name="Orders / revenue", marker=_bar(t), hovertemplate="FY%{x}: %{y:.2f}x<extra></extra>"))
    fig.add_hline(y=1.0, line=dict(color=t["ink2"], width=1.4))
    style_figure(fig, title="Book-to-bill by fiscal year: >1 means the backlog is growing", height=height, ytitle="x", legend=False, hovermode="x unified")
    year_axis(fig)
    return fig


def fig_price_estimates(intel: Intel, first_year: int = 2016, height: int = 400) -> go.Figure:
    """Rough EUR m per MW. Estimates only: Siemens Energy has not disclosed MW or an average selling price since FY2022."""
    t = tokens()
    e = intel.econ.loc[first_year:]
    fig = go.Figure()
    for col, name, color, dash, mode in (
        ("rev_per_mw_3y", "New-unit revenue per delivered MW, 3-year pooled (onshore + offshore)", t["muted"], "dot", "lines"),
        ("rev_per_mw", "New-unit revenue per delivered MW (onshore + offshore)", t["blue"], "solid", "lines+markers"),
        ("asp_onshore", "Average selling price, onshore (as reported)", t["orange"], "solid", "lines+markers"),
    ):
        s = e[col].dropna() if col in e else None
        if s is None or s.empty:
            continue
        fig.add_trace(go.Scatter(x=s.index, y=s, mode=mode, name=name, line=_line(t, color) | {"dash": dash},
                                 marker=dict(size=8, color=color, line=dict(color=t["surface"], width=2)), hovertemplate="FY%{x}: EUR %{y:.2f}m/MW<extra>" + name + "</extra>"))
    style_figure(fig, title="Estimated price per MW (EUR m per MW)", height=height, ytitle="EUR m / MW", hovermode="x unified", legend=True)
    year_axis(fig)
    return fig


def fig_margins(intel: Intel, first_year: int = 2016, height: int = 400) -> go.Figure:
    t = tokens()
    e = intel.econ.loc[first_year:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=e.index, y=e["net_margin"], mode="lines+markers", name="Net income margin", line=_line(t, t["blue"]),
                             marker=dict(size=8, line=dict(color=t["surface"], width=2)), hovertemplate="FY%{x}: %{y:.1%}<extra>Net margin</extra>"))
    p = e["margin_before_si"].dropna()
    if len(p):
        fig.add_trace(go.Scatter(x=p.index, y=p, mode="lines+markers", name="Margin before special items", line=_line(t, t["orange"]),
                                 marker=dict(size=8, line=dict(color=t["surface"], width=2)), hovertemplate="FY%{x}: %{y:.1%}<extra>Before special items</extra>"))
    fig.add_hline(y=0, line=dict(color=t["ink2"], width=1.2))
    style_figure(fig, title="Margins by fiscal year", height=height, ytitle="Share of revenue", yformat=".0%", hovermode="x unified")
    year_axis(fig)
    return fig


def fig_service_share_fy(intel: Intel, first_year: int = 2016, height: int = 400) -> go.Figure:
    t = tokens()
    e = intel.econ.loc[first_year:].dropna(subset=["revenue_service"])
    fig = go.Figure()
    fig.add_trace(go.Bar(x=e.index, y=e["revenue_service"], name="Service revenue (EUR m)", marker=_bar(t), hovertemplate="FY%{x}: EUR %{y:,.0f}m<extra>Service</extra>"))
    fig.add_trace(go.Scatter(x=e.index, y=e["service_share"], name="Share of revenue (right axis)", yaxis="y2", mode="lines+markers", line=_line(t, t["orange"]),
                             marker=dict(size=8, line=dict(color=t["surface"], width=2)), hovertemplate="FY%{x}: %{y:.1%}<extra>Service share</extra>"))
    style_figure(fig, title="Service revenue by fiscal year", height=height, ytitle="EUR m", hovermode="x unified")
    fig.update_layout(bargap=0.28, yaxis2=dict(overlaying="y", side="right", showgrid=False, tickformat=".0%", rangemode="tozero"))
    year_axis(fig)
    return fig


def fig_offshore_orders_share(intel: Intel, height: int = 400) -> go.Figure:
    t = tokens()
    e = intel.econ.dropna(subset=["orders_onshore", "orders_offshore"])
    fig = go.Figure()
    fig.add_trace(go.Bar(x=e.index, y=e["orders_onshore"], name="Onshore", marker=_bar(t, t["orange"]), hovertemplate="FY%{x}: EUR %{y:,.0f}m<extra>Onshore</extra>"))
    fig.add_trace(go.Bar(x=e.index, y=e["orders_offshore"], name="Offshore", marker=_bar(t), hovertemplate="FY%{x}: EUR %{y:,.0f}m<extra>Offshore</extra>"))
    if "orders_service" in e and e["orders_service"].notna().any():
        fig.add_trace(go.Bar(x=e.index, y=e["orders_service"].fillna(0), name="Service", marker=_bar(t, t["neutral"]), hovertemplate="FY%{x}: EUR %{y:,.0f}m<extra>Service</extra>"))
    style_figure(fig, title="Order intake by business (EUR m, years the workbook splits it)", height=height, ytitle="EUR m", hovermode="x unified")
    fig.update_layout(barmode="stack", bargap=0.28)
    year_axis(fig)
    return fig
