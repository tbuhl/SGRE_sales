"""Colour tokens, chart styling and card CSS for the intelligence tabs.

Categorical hues come from the validated 8-slot palette (light and dark steps) and are assigned by entity, never by rank.
Ordered classes use a one-hue blue ramp; polarity uses blue <-> red around a neutral gray.
"""
from __future__ import annotations

import html
from typing import Iterable

import numpy as np
import plotly.graph_objects as go
import streamlit as st

CATEGORICAL_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
CATEGORICAL_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
RAMP_LIGHT = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]  # small -> large (light -> dark)
RAMP_DARK = ["#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4"]  # small -> large (dim -> bright)
NEUTRAL_LIGHT, NEUTRAL_DARK = "#9aa3b2", "#6f7b8d"
SEQ_BLUE_LIGHT = [[0, "#eef5fd"], [0.5, "#6da7ec"], [1, "#104281"]]
SEQ_BLUE_DARK = [[0, "#13213a"], [0.5, "#256abf"], [1, "#9ec5f4"]]


def is_dark() -> bool:
    return bool(st.session_state.get("dark_mode", False))


def tokens() -> dict:
    dark = is_dark()
    cat = CATEGORICAL_DARK if dark else CATEGORICAL_LIGHT
    return {
        "dark": dark,
        "cat": cat,
        "blue": cat[0],
        "orange": cat[1],
        "aqua": cat[2],
        "yellow": cat[3],
        "magenta": cat[4],
        "green": cat[5],
        "violet": cat[6],
        "red": cat[7],
        "ramp": RAMP_DARK if dark else RAMP_LIGHT,
        "neutral": NEUTRAL_DARK if dark else NEUTRAL_LIGHT,
        "ink": "#f5f7fb" if dark else "#0f172a",
        "ink2": "#c3ccd9" if dark else "#334155",
        "muted": "#9aa7b8" if dark else "#64748b",
        "grid": "rgba(148,163,184,0.16)" if dark else "rgba(15,23,42,0.08)",
        "axis": "rgba(148,163,184,0.38)" if dark else "rgba(15,23,42,0.28)",
        "surface": "#0f1727" if dark else "#fbfcfd",
        "seq": SEQ_BLUE_DARK if dark else SEQ_BLUE_LIGHT,
        "template": "plotly_dark" if dark else "plotly_white",
        "up": "#34d399" if dark else "#006300",
        "down": "#f87171" if dark else "#b42318",
        "warn": "#fbbf24" if dark else "#a15c00",
    }


def continent_colors() -> dict[str, str]:
    t = tokens()
    return {
        "Europe": t["blue"],
        "Americas": t["orange"],
        "Asia": t["aqua"],
        "Oceania": t["yellow"],
        "Africa": t["magenta"],
        "Unknown": t["neutral"],
    }


def status_colors() -> dict[str, str]:
    t = tokens()
    return {"Active": t["blue"], "Cooling": t["orange"], "Dormant": t["neutral"], "Emerging": t["aqua"]}


def service_colors() -> dict[str, str]:
    t = tokens()
    return {"Offshore": t["blue"], "Onshore": t["orange"], "Unknown": t["neutral"]}


def _legend_entries(fig: go.Figure) -> list[str]:
    names = []
    for tr in fig.data:
        if getattr(tr, "showlegend", None) is False or not getattr(tr, "name", None):
            continue
        names.append(str(tr.name))
    return names


def style_figure(
    fig: go.Figure,
    *,
    title: str | None = None,
    height: int = 420,
    xtitle: str | None = None,
    ytitle: str | None = None,
    legend: bool = True,
    hovermode: str | None = "x unified",
    yformat: str | None = None,
    width_hint: int = 540,
    legend_position: str | None = None,
) -> go.Figure:
    """Apply the shared chart chrome: transparent surface, hairline grid, ink text.

    The legend sits above the plot when it fits on one row (``width_hint`` is the approximate chart width in px),
    otherwise it moves below the plot so it never collides with the title.
    """
    t = tokens()
    names = _legend_entries(fig) if legend else []
    est_px = sum(len(n) * 6.4 + 44 for n in names)
    legend_bottom = bool(names) and (legend_position == "bottom" or (legend_position is None and est_px > 0.92 * width_hint))
    rows = max(1, int(np.ceil(est_px / max(width_hint * 0.92, 1)))) if names else 0
    top = 62 if title else 40
    bottom = 10
    legend_cfg = dict(orientation="h", xanchor="left", x=0.0, font=dict(size=11, color=t["ink2"]), title_text="")
    if legend_bottom:
        bottom = 50 + rows * 20 + (34 if xtitle else 0)
        plot_h = max(height - top - bottom, 120)
        legend_cfg.update(yanchor="top", y=-(38 + (34 if xtitle else 0)) / plot_h)
    else:
        legend_cfg.update(yanchor="bottom", y=1.0)
    fig.update_layout(
        template=t["template"],
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=t["ink2"], size=12),
        margin=dict(l=10, r=22, t=top, b=bottom),
        showlegend=legend,
        legend=legend_cfg,
        hovermode=hovermode,
        hoverlabel=dict(font_size=12),
    )
    if title:
        fig.update_layout(title=dict(text=title, x=0.01, xanchor="left", font=dict(size=16, color=t["ink"])))
    fig.update_xaxes(showgrid=False, linecolor=t["axis"], zeroline=False, ticks="outside", tickcolor=t["axis"], title_text=xtitle, title_font=dict(color=t["muted"], size=11))
    fig.update_yaxes(
        showgrid=True,
        gridcolor=t["grid"],
        gridwidth=1,
        zeroline=False,
        linecolor="rgba(0,0,0,0)",
        title_text=ytitle,
        title_font=dict(color=t["muted"], size=11),
        tickformat=yformat,
    )
    return fig


def year_axis(fig: go.Figure) -> go.Figure:
    """One tick per year, with half-a-year padding so the first and last labels are never clipped."""
    xs = []
    for tr in fig.data:
        x = getattr(tr, "x", None)
        if x is None:
            continue
        try:
            xs.extend(float(v) for v in x if v is not None and v == v)
        except (TypeError, ValueError):
            continue
    if xs:
        fig.update_xaxes(dtick=1, range=[min(xs) - 0.6, max(xs) + 0.6])
    else:
        fig.update_xaxes(dtick=1)
    return fig


def show(fig: go.Figure, key: str) -> None:
    st.plotly_chart(fig, width="stretch", key=key, config={"displaylogo": False})


def table_view(df, title: str = "Show data table", key: str | None = None) -> None:
    """Every chart gets a table twin (accessible, copyable)."""
    with st.expander(title):
        st.dataframe(df, width="stretch", hide_index=False, key=key)


# ---- cards ------------------------------------------------------------------------------
def sparkline_svg(values: Iterable[float], color: str, width: int = 120, height: int = 34) -> str:
    vals = [float(v) for v in values if v is not None and v == v]
    if len(vals) < 2:
        return ""
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    pad = 3
    pts = []
    for idx, v in enumerate(vals):
        x = pad + (width - 2 * pad) * idx / (len(vals) - 1)
        y = height - pad - (height - 2 * pad) * (v - lo) / span
        pts.append((x, y))
    path = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    lx, ly = pts[-1]
    return (
        f'<svg class="spark" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" aria-label="trend">'
        f'<polyline fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" points="{path}"/>'
        f'<circle cx="{lx:.1f}" cy="{ly:.1f}" r="3.2" fill="{color}" stroke="var(--intel-surface)" stroke-width="2"/></svg>'
    )


TONE_META = {
    "up": ("&#9650;", "Positive"),
    "down": ("&#9660;", "Negative"),
    "warn": ("&#9888;", "Watch"),
    "neutral": ("&#9679;", "Context"),
}


def finding_card_html(item, t: dict) -> str:
    tone = item.tone if item.tone in TONE_META else "neutral"
    icon, label = TONE_META[tone]
    spark_color = {"up": t["up"], "down": t["down"], "warn": t["warn"], "neutral": t["blue"]}[tone]
    spark = sparkline_svg(item.series, spark_color) if item.series else ""
    span = ""
    if item.series and item.series_start is not None:
        span = f'<span class="spark-span">{item.series_start}&ndash;{item.series_start + len(item.series) - 1}</span>'
    return f"""
<div class="intel-card tone-{tone}">
  <div class="intel-card-top"><span class="intel-chip">{html.escape(item.category)}</span>
    <span class="intel-tone tone-{tone}"><span aria-hidden="true">{icon}</span> {label}</span></div>
  <div class="intel-headline">{html.escape(item.headline)}</div>
  <div class="intel-metric-row"><div><div class="intel-metric">{html.escape(item.metric)}</div>
    <div class="intel-mlabel">{html.escape(item.metric_label)}</div></div><div class="intel-spark">{spark}{span}</div></div>
  <div class="intel-detail">{html.escape(item.detail)}</div>
  <div class="intel-where">{html.escape(item.where)}</div>
</div>"""


def inject_css() -> None:
    t = tokens()
    dark = t["dark"]
    card_bg = "rgba(17,24,39,0.62)" if dark else "rgba(255,255,255,0.72)"
    border = "rgba(148,163,184,0.26)" if dark else "rgba(15,23,42,0.10)"
    shadow = "0 18px 40px rgba(0,0,0,0.38)" if dark else "0 18px 36px rgba(2,6,23,0.09)"
    st.markdown(
        f"""
<style>
:root {{ --intel-surface: {t['surface']}; --intel-ink: {t['ink']}; --intel-ink2: {t['ink2']}; --intel-muted: {t['muted']}; }}
.intel-grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(360px,1fr)); gap:14px; margin:0.4rem 0 1.2rem 0; }}
.intel-card {{ background:{card_bg}; border:1px solid {border}; border-left:4px solid {t['blue']}; border-radius:16px; padding:14px 16px 12px 16px;
  box-shadow:{shadow}; backdrop-filter:blur(8px); display:flex; flex-direction:column; gap:6px; }}
.intel-card.tone-up {{ border-left-color:{t['up']}; }}
.intel-card.tone-down {{ border-left-color:{t['down']}; }}
.intel-card.tone-warn {{ border-left-color:{t['warn']}; }}
.intel-card-top {{ display:flex; justify-content:space-between; align-items:center; gap:8px; }}
.intel-chip {{ font-size:0.66rem; letter-spacing:0.09em; text-transform:uppercase; font-weight:800; color:var(--intel-muted); }}
.intel-tone {{ font-size:0.68rem; font-weight:700; color:var(--intel-ink2); border:1px solid {border}; border-radius:999px; padding:1px 8px; }}
.intel-tone.tone-up {{ color:{t['up']}; }} .intel-tone.tone-down {{ color:{t['down']}; }} .intel-tone.tone-warn {{ color:{t['warn']}; }}
.intel-headline {{ font-weight:800; font-size:1.02rem; line-height:1.28; letter-spacing:-0.01em; color:var(--intel-ink); }}
.intel-metric-row {{ display:flex; justify-content:space-between; align-items:flex-end; gap:10px; margin-top:2px; }}
.intel-metric {{ font-size:1.85rem; font-weight:800; letter-spacing:-0.02em; line-height:1.05; color:var(--intel-ink); }}
.intel-mlabel {{ font-size:0.72rem; color:var(--intel-muted); margin-top:2px; max-width:210px; line-height:1.25; }}
.intel-spark {{ display:flex; flex-direction:column; align-items:flex-end; gap:1px; }}
.spark-span {{ font-size:0.62rem; color:var(--intel-muted); }}
.intel-detail {{ font-size:0.84rem; line-height:1.5; color:var(--intel-ink2); }}
.intel-where {{ font-size:0.68rem; color:var(--intel-muted); margin-top:auto; padding-top:4px; }}
.intel-hero {{ margin:0.1rem 0 0.9rem 0; }}
.intel-hero h2 {{ margin:0 0 0.15rem 0; }}
.intel-sub {{ color:var(--intel-muted); font-size:0.86rem; margin:0; }}
.intel-note {{ border:1px solid {border}; border-radius:12px; padding:8px 12px; font-size:0.8rem; color:var(--intel-ink2); background:{card_bg}; margin:0.2rem 0 0.8rem 0; }}
.intel-watch li {{ margin-bottom:0.35rem; line-height:1.45; }}
.sev {{ font-weight:700; }}
</style>
""",
        unsafe_allow_html=True,
    )
