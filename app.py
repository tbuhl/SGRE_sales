from __future__ import annotations

import html
import json
import math
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st


APP_TITLE = "Siemens Gamesa Sales Intelligence Dashboard"
DATA_DIR = Path("data")
DATA_CACHE_DIR = Path("data_cache")
PARSED_DATA_FILE = DATA_CACHE_DIR / "sgre_parsed_data.pkl"
STOCK_MONTHLY_FILE = DATA_DIR / "sgre_stock_monthly.json"
MARKET_MONTHLY_FILE = DATA_DIR / "market_prices_monthly.json"

NEWS_FEEDS = [
    {
        "source": "Windpower Monthly - Siemens Gamesa",
        "url": "https://www.windpowermonthly.com/rss/siemensgamesa",
        "focus": "industry",
    },
    {
        "source": "OffshoreWIND.biz - Siemens Gamesa",
        "url": "https://www.offshorewind.biz/?s=siemens%20gamesa&feed=rss2",
        "focus": "industry",
    },
    {
        "source": "Google News (Reuters + Siemens Gamesa)",
        "url": "https://news.google.com/rss/search?q=site%3Areuters.com%20%22Siemens%20Gamesa%22%20wind&hl=en-US&gl=US&ceid=US:en",
        "focus": "financial",
    },
    {
        "source": "Google News (Bloomberg + Siemens Gamesa)",
        "url": "https://news.google.com/rss/search?q=site%3Abloomberg.com%20%22Siemens%20Gamesa%22%20wind&hl=en-US&gl=US&ceid=US:en",
        "focus": "financial",
    },
    {
        "source": "Google News (Recharge + Siemens Gamesa)",
        "url": "https://news.google.com/rss/search?q=site%3Arechargenews.com%20%22Siemens%20Gamesa%22&hl=en-US&gl=US&ceid=US:en",
        "focus": "industry",
    },
    {
        "source": "Google News (Power Technology + Siemens Gamesa)",
        "url": "https://news.google.com/rss/search?q=site%3Apower-technology.com%20%22Siemens%20Gamesa%22&hl=en-US&gl=US&ceid=US:en",
        "focus": "technical",
    },
]

NORDEX_KEYWORDS = ("siemens gamesa", "siemens-gamesa", "sgre", "sg")
NORDEX_NEWS_BLACKLIST = (
    "your career at siemens gamesa",
    "this is siemens gamesa",
    "onshore wind turbines - siemens gamesa",
    "offshore wind turbines - siemens gamesa",
    "stock price & latest news",
    "jobs",
    "/careers",
    "/career",
    "/jobs",
    "weekly roundup",
)

NON_TARGET_NEWS_TERMS = (
    "vestas",
    "nordex",
    "ge vernova",
    "goldwind",
    "enercon",
)

NORDEX_TITLE_RE = re.compile(r"\b(?:siemens[\s-]?gamesa|sgre)\b", re.IGNORECASE)
PATENTSCOPE_SEARCH_URL = (
    "https://patentscope.wipo.int/search/en/result.jsf"
    "?query=PA%3A%22Siemens%20Gamesa%20Renewable%20Energy%22&sortOption=-DP"
)

DEFAULT_ECON_METRICS = [
    "revenue (mEUR)",
    "EBIT",
    "Order intake (bnEUR)",
    "Order inkake (MW)",
    "Order backlog – wind turbines (bnEUR)",
    "Order backlog – service (bnEUR)",
    "deliveries [MW]",
]

PLATFORM_SLOT_SPECS = [
    {
        "slot": 1,
        "qty": ["#", "No_WTG_1", "NumberofTurbines"],
        "platform": ["platform", "TurbineType"],
        "rotor": ["Rotor", "Rotor_1"],
        "mw": ["MW", "MW_1"],
    },
    {
        "slot": 2,
        "qty": ["#.1", "No_WTG_2"],
        "platform": ["platform.1", "Multi platform"],
        "rotor": ["Rotor.1", "Rotor_2"],
        "mw": ["MW.1", "MW_2"],
    },
    {
        "slot": 3,
        "qty": ["#.2", "No_WTG_3"],
        "platform": ["platform.2", "Multi platform.1"],
        "rotor": ["Rotor.2", "Rotor"],
        "mw": ["MW.2", "MW"],
    },
    {
        "slot": 4,
        "qty": ["#.3", "No_WTG_4"],
        "platform": ["platform.3", "Multi platform.2"],
        "rotor": ["Rotor.3", "Rotor.1"],
        "mw": ["MW.3", "MW.1"],
    },
]

COUNTRY_MAP = {
    "usa": "United States",
    "united states": "United States",
    "united states and canada": "United States and Canada",
    "uk": "United Kingdom",
    "uk - offshore": "United Kingdom - Offshore",
    "holland": "Netherlands",
    "the netherlands": "Netherlands",
    "irland": "Ireland",
    "spanien": "Spain",
    "mongoliet": "Mongolia",
    "den dominikanske republik": "Dominican Republic",
    "dominican republic": "Dominican Republic",
    "?": "Unknown",
    "no info": "Unknown",
    "notmentioned": "Unknown",
    "un disclosed": "Undisclosed",
}

GEO_COUNTRY_OVERRIDES = {
    "United States and Canada": "United States",
    "United Kingdom - Offshore": "United Kingdom",
    "Wales": "United Kingdom",
    "Unknown": None,
    "EU": None,
    "NotMentioned": None,
    "Undisclosed": None,
}

COUNTRY_ISO3_MAP = {
    "Argentina": "ARG",
    "Australia": "AUS",
    "Austria": "AUT",
    "Belgium": "BEL",
    "Bolivia": "BOL",
    "Brazil": "BRA",
    "Bulgaria": "BGR",
    "Canada": "CAN",
    "Cape Verde": "CPV",
    "Chile": "CHL",
    "China": "CHN",
    "Colombia": "COL",
    "Costa Rica": "CRI",
    "Croatia": "HRV",
    "Cyprus": "CYP",
    "Denmark": "DNK",
    "Dominican Republic": "DOM",
    "Egypt": "EGY",
    "El Salvador": "SLV",
    "Estonia": "EST",
    "Finland": "FIN",
    "France": "FRA",
    "Germany": "DEU",
    "Greece": "GRC",
    "Guatemala": "GTM",
    "Honduras": "HND",
    "Hungary": "HUN",
    "India": "IND",
    "Ireland": "IRL",
    "Italy": "ITA",
    "Jamaica": "JAM",
    "Japan": "JPN",
    "Jordan": "JOR",
    "Kazakhstan": "KAZ",
    "Kenya": "KEN",
    "Latvia": "LVA",
    "Lithuania": "LTU",
    "Mexico": "MEX",
    "Mongolia": "MNG",
    "Morocco": "MAR",
    "Netherlands": "NLD",
    "New Zealand": "NZL",
    "Nicaragua": "NIC",
    "Norway": "NOR",
    "Pakistan": "PAK",
    "Panama": "PAN",
    "Peru": "PER",
    "Philippines": "PHL",
    "Poland": "POL",
    "Portugal": "PRT",
    "Puerto Rico": "PRI",
    "Romania": "ROU",
    "Russia": "RUS",
    "Saudi Arabia": "SAU",
    "Senegal": "SEN",
    "Serbia": "SRB",
    "South Africa": "ZAF",
    "South Korea": "KOR",
    "Spain": "ESP",
    "Sri Lanka": "LKA",
    "Sweden": "SWE",
    "Taiwan": "TWN",
    "Thailand": "THA",
    "Turkey": "TUR",
    "Ukraine": "UKR",
    "United Kingdom": "GBR",
    "United States": "USA",
    "Uruguay": "URY",
    "Vietnam": "VNM",
}

COUNTRY_REGION_CONTINENT = {
    "Argentina": ("South America", "Americas"),
    "Australia": ("Oceania", "Oceania"),
    "Austria": ("Western Europe", "Europe"),
    "Belgium": ("Western Europe", "Europe"),
    "Bolivia": ("South America", "Americas"),
    "Brazil": ("South America", "Americas"),
    "Bulgaria": ("Eastern Europe", "Europe"),
    "Canada": ("North America", "Americas"),
    "Cape Verde": ("West Africa", "Africa"),
    "Chile": ("South America", "Americas"),
    "China": ("East Asia", "Asia"),
    "Colombia": ("South America", "Americas"),
    "Costa Rica": ("Central America", "Americas"),
    "Croatia": ("Southern Europe", "Europe"),
    "Cyprus": ("Southern Europe", "Europe"),
    "Denmark": ("Northern Europe", "Europe"),
    "Dominican Republic": ("Caribbean", "Americas"),
    "Egypt": ("North Africa", "Africa"),
    "El Salvador": ("Central America", "Americas"),
    "Estonia": ("Northern Europe", "Europe"),
    "EU": ("Europe", "Europe"),
    "Finland": ("Northern Europe", "Europe"),
    "France": ("Western Europe", "Europe"),
    "Germany": ("Western Europe", "Europe"),
    "Greece": ("Southern Europe", "Europe"),
    "Guatemala": ("Central America", "Americas"),
    "Honduras": ("Central America", "Americas"),
    "Hungary": ("Eastern Europe", "Europe"),
    "India": ("South Asia", "Asia"),
    "Ireland": ("Northern Europe", "Europe"),
    "Italy": ("Southern Europe", "Europe"),
    "Jamaica": ("Caribbean", "Americas"),
    "Japan": ("East Asia", "Asia"),
    "Jordan": ("Middle East", "Asia"),
    "Kazakhstan": ("Central Asia", "Asia"),
    "Kenya": ("East Africa", "Africa"),
    "Latvia": ("Northern Europe", "Europe"),
    "Lithuania": ("Northern Europe", "Europe"),
    "Mexico": ("North America", "Americas"),
    "Mongolia": ("East Asia", "Asia"),
    "Morocco": ("North Africa", "Africa"),
    "Netherlands": ("Western Europe", "Europe"),
    "New Zealand": ("Oceania", "Oceania"),
    "Nicaragua": ("Central America", "Americas"),
    "Norway": ("Northern Europe", "Europe"),
    "Pakistan": ("South Asia", "Asia"),
    "Panama": ("Central America", "Americas"),
    "Peru": ("South America", "Americas"),
    "Philippines": ("Southeast Asia", "Asia"),
    "Poland": ("Eastern Europe", "Europe"),
    "Portugal": ("Southern Europe", "Europe"),
    "Puerto Rico": ("Caribbean", "Americas"),
    "Romania": ("Eastern Europe", "Europe"),
    "Russia": ("Eastern Europe", "Europe"),
    "Saudi Arabia": ("Middle East", "Asia"),
    "Senegal": ("West Africa", "Africa"),
    "Serbia": ("Southern Europe", "Europe"),
    "South Africa": ("Southern Africa", "Africa"),
    "South Korea": ("East Asia", "Asia"),
    "Spain": ("Southern Europe", "Europe"),
    "Sri Lanka": ("South Asia", "Asia"),
    "Sweden": ("Northern Europe", "Europe"),
    "Taiwan": ("East Asia", "Asia"),
    "Thailand": ("Southeast Asia", "Asia"),
    "Turkey": ("Middle East", "Asia"),
    "Ukraine": ("Eastern Europe", "Europe"),
    "United Kingdom": ("Northern Europe", "Europe"),
    "United Kingdom - Offshore": ("Northern Europe", "Europe"),
    "United States": ("North America", "Americas"),
    "United States and Canada": ("North America", "Americas"),
    "Unknown": ("Unknown", "Unknown"),
    "Undisclosed": ("Unknown", "Unknown"),
    "Uruguay": ("South America", "Americas"),
    "Vietnam": ("Southeast Asia", "Asia"),
    "Wales": ("Northern Europe", "Europe"),
}

COUNTRY_REGION_CONTINENT_LOWER = {k.lower(): v for k, v in COUNTRY_REGION_CONTINENT.items()}

SUMMARY_LABELS = {
    "# mw",
    "# of sites",
    "# aom5000 energy based yield",
    "# aom4000 time based yield",
    "% aom5000 of confirmed",
    "% of confirmed energy yield sc",
    "services counters",
    "market share",
    "country",
    "type of contract",
    "potential turbines",
    "total # blades",
    "total # turbines incl undisclosed",
    "w. average rotor dia / #",
    "w. average rotor dia / mw",
    "w. average service contract",
    "weighted average",
    "counters",
    "turbines",
    "sales",
    "1 qtr",
    "2 qtr",
    "3 qtr",
    "4 qtr",
    "0-5",
    "5-10",
    "10-15",
    "15-20",
    "20+",
}

CUSTOMER_UNKNOWN_VALUES = {"?", "nan", "none", "no info", "not mentioned", "unknown"}


def apply_page_style(dark_mode: bool) -> None:
    if dark_mode:
        text_color = "#f5f7fb"
        muted_color = "#a6b3c3"
        bg_main = (
            "radial-gradient(1200px 620px at 96% -12%, rgba(250, 204, 21, 0.16) 0%, rgba(250, 204, 21, 0) 52%),"
            "radial-gradient(900px 520px at -4% 18%, rgba(16, 185, 129, 0.18) 0%, rgba(16, 185, 129, 0) 48%),"
            "linear-gradient(180deg, #090d16 0%, #0d1424 100%)"
        )
        bg_sidebar = "linear-gradient(180deg, rgba(11, 17, 31, 0.95) 0%, rgba(8, 12, 22, 0.98) 100%)"
        header_bg = "rgba(9, 14, 26, 0.82)"
        header_border = "rgba(250, 204, 21, 0.20)"
        card_bg = "rgba(17, 24, 39, 0.62)"
        card_border = "rgba(148, 163, 184, 0.28)"
        card_shadow = "0 24px 56px rgba(0, 0, 0, 0.48), 0 8px 18px rgba(0, 0, 0, 0.25)"
        card_glow = "rgba(250, 204, 21, 0.26)"
        frame_border = "rgba(148, 163, 184, 0.25)"
        metric_label_color = "#d5deea"
        metric_value_color = "#ffffff"
        metric_delta_color = "#f8fafc"
        tab_bg = "rgba(17, 24, 39, 0.66)"
        tab_active_bg = "linear-gradient(120deg, rgba(234, 179, 8, 0.30), rgba(16, 185, 129, 0.28))"
        tab_hover_bg = "rgba(30, 41, 59, 0.80)"
        tab_border = "rgba(148, 163, 184, 0.25)"
        tab_active_border = "rgba(250, 204, 21, 0.40)"
        tab_text_color = "#d7e0ec"
        tab_active_text_color = "#ffffff"
        input_bg = "rgba(15, 23, 42, 0.78)"
        input_border = "rgba(148, 163, 184, 0.30)"
        input_focus = "rgba(250, 204, 21, 0.42)"
        accent_start = "#facc15"
        accent_end = "#10b981"
        link_color = "#a7f3d0"
        link_hover = "#fde047"
        chart_wrap_bg = "rgba(17, 24, 39, 0.56)"
        chart_wrap_border = "rgba(148, 163, 184, 0.25)"
    else:
        text_color = "#0f172a"
        muted_color = "#475569"
        bg_main = (
            "radial-gradient(1200px 620px at 94% -10%, rgba(250, 204, 21, 0.28) 0%, rgba(250, 204, 21, 0) 50%),"
            "radial-gradient(1000px 540px at -8% 20%, rgba(52, 211, 153, 0.24) 0%, rgba(52, 211, 153, 0) 46%),"
            "linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%)"
        )
        bg_sidebar = "linear-gradient(180deg, rgba(255, 255, 255, 0.88) 0%, rgba(248, 250, 252, 0.94) 100%)"
        header_bg = "rgba(255, 255, 255, 0.78)"
        header_border = "rgba(15, 23, 42, 0.10)"
        card_bg = "rgba(255, 255, 255, 0.66)"
        card_border = "rgba(15, 23, 42, 0.10)"
        card_shadow = "0 26px 52px rgba(2, 6, 23, 0.12), 0 8px 18px rgba(2, 6, 23, 0.08)"
        card_glow = "rgba(16, 185, 129, 0.20)"
        frame_border = "rgba(15, 23, 42, 0.11)"
        metric_label_color = "#334155"
        metric_value_color = "#0f172a"
        metric_delta_color = "#0f172a"
        tab_bg = "rgba(255, 255, 255, 0.72)"
        tab_active_bg = "linear-gradient(120deg, rgba(254, 240, 138, 0.74), rgba(167, 243, 208, 0.72))"
        tab_hover_bg = "rgba(255, 255, 255, 0.90)"
        tab_border = "rgba(15, 23, 42, 0.12)"
        tab_active_border = "rgba(16, 185, 129, 0.44)"
        tab_text_color = "#334155"
        tab_active_text_color = "#0f172a"
        input_bg = "rgba(255, 255, 255, 0.80)"
        input_border = "rgba(15, 23, 42, 0.14)"
        input_focus = "rgba(16, 185, 129, 0.36)"
        accent_start = "#eab308"
        accent_end = "#10b981"
        link_color = "#0f766e"
        link_hover = "#a16207"
        chart_wrap_bg = "rgba(255, 255, 255, 0.58)"
        chart_wrap_border = "rgba(15, 23, 42, 0.11)"

    st.markdown(
        f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito+Sans:wght@400;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,300..700,0..1,-50..200');
@import url('https://fonts.googleapis.com/icon?family=Material+Icons');

html, body, .stApp {{
  font-family: 'Nunito Sans', sans-serif;
  color: {text_color};
}}

.stApp {{
  background: {bg_main};
  background-attachment: fixed;
}}

.block-container {{
  padding-top: 1.2rem;
  padding-bottom: 2rem;
}}

h1, h2, h3, h4, h5, h6 {{
  letter-spacing: -0.01em;
  font-weight: 700;
}}

h1 {{
  font-weight: 800;
  background-image: linear-gradient(110deg, {accent_start} 0%, {accent_end} 56%, {text_color} 100%);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}}

p, span, label, li {{
  color: {text_color};
}}

a {{
  color: {link_color};
  text-decoration-color: {link_color};
}}

a:hover {{
  color: {link_hover};
  text-decoration-color: {link_hover};
}}

section[data-testid="stSidebar"] {{
  background: {bg_sidebar};
  border-right: 1px solid {frame_border};
  backdrop-filter: blur(9px);
}}

section[data-testid="stSidebar"] * {{
  color: {text_color};
}}

section[data-testid="stSidebar"] > div:first-child {{
  height: 100vh;
}}

section[data-testid="stSidebar"] div[data-testid="stSidebarUserContent"] {{
  display: flex;
  flex-direction: column;
  height: 100%;
  min-height: 100%;
}}

[data-testid="stHeader"],
[data-testid="stToolbar"] {{
  background: {header_bg} !important;
  border-bottom: 1px solid {header_border} !important;
  backdrop-filter: blur(12px);
}}

[data-testid="stDecoration"] {{
  background: transparent !important;
}}

/* Restore Streamlit/Material icon rendering (prevents ligature text such as keyboard_double_arrow_right) */
[class*="material-icons"],
[class*="material-symbols"],
[data-testid="stHeader"] button span,
[data-testid="stToolbar"] button span,
[data-testid="collapsedControl"] span,
[data-testid="stSidebarCollapseButton"] span,
[data-testid="stSidebarNav"] button span,
button[aria-label*="sidebar"] span,
button[aria-label*="Sidebar"] span,
button[title*="sidebar"] span,
button[title*="Sidebar"] span {{
  font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Icons" !important;
  font-feature-settings: "liga" 1, "calt" 1;
  -webkit-font-feature-settings: "liga" 1, "calt" 1;
  font-weight: normal;
  font-style: normal;
  letter-spacing: normal;
  text-transform: none;
}}

section[data-testid="stSidebar"] div[data-baseweb="select"] > div,
section[data-testid="stSidebar"] div[data-baseweb="input"] > div,
section[data-testid="stSidebar"] div[data-testid="stNumberInput"] input,
section[data-testid="stSidebar"] div[data-testid="stTextInput"] input {{
  background: {input_bg};
  border: 1px solid {input_border};
  color: {text_color};
  border-radius: 10px;
}}

section[data-testid="stSidebar"] div[data-baseweb="select"] > div:focus-within,
section[data-testid="stSidebar"] div[data-baseweb="input"] > div:focus-within,
section[data-testid="stSidebar"] div[data-testid="stNumberInput"] input:focus,
section[data-testid="stSidebar"] div[data-testid="stTextInput"] input:focus {{
  border-color: {input_focus} !important;
  box-shadow: 0 0 0 2px {input_focus} !important;
}}

div[data-testid="stAlert"],
div[data-testid="stDataFrame"] {{
  background: radial-gradient(65% 80% at 50% 0%, {card_glow} 0%, rgba(255,255,255,0) 100%), {card_bg};
  border: 1px solid {card_border};
  border-radius: 14px;
  box-shadow: {card_shadow};
  backdrop-filter: blur(8px);
}}

div[data-testid="stMetric"] {{
  background: radial-gradient(65% 80% at 50% 0%, {card_glow} 0%, rgba(255,255,255,0) 100%), {card_bg};
  border: 1px solid {card_border};
  border-radius: 14px;
  padding: 0.45rem 0.82rem;
  box-shadow: {card_shadow};
  backdrop-filter: blur(8px);
}}

div[data-testid="stMetric"] div[data-testid="stMetricLabel"] {{
  color: {metric_label_color} !important;
  font-weight: 700 !important;
}}

div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{
  color: {metric_value_color} !important;
  letter-spacing: -0.01em;
}}

div[data-testid="stMetricDelta"] {{
  color: {metric_delta_color} !important;
  font-size: 0.72rem !important;
  line-height: 1.05 !important;
}}

.fy-kpi-card {{
  background: radial-gradient(65% 80% at 50% 0%, {card_glow} 0%, rgba(255,255,255,0) 100%), {card_bg};
  border: 1px solid {card_border};
  border-radius: 14px;
  padding: 0.55rem 0.82rem;
  box-shadow: {card_shadow};
  backdrop-filter: blur(8px);
}}

.fy-kpi-title {{
  color: {metric_label_color};
  font-size: 0.78rem;
  font-weight: 700;
  margin-bottom: 0.15rem;
}}

.fy-kpi-value {{
  color: {metric_value_color};
  font-size: 1.55rem;
  font-weight: 800;
  letter-spacing: -0.01em;
  line-height: 1.15;
}}

.fy-kpi-delta {{
  color: {metric_delta_color};
  font-size: 0.72rem;
  margin-top: 0.08rem;
}}

div[data-testid="stPlotlyChart"] {{
  background: radial-gradient(65% 80% at 50% 0%, {card_glow} 0%, rgba(255,255,255,0) 100%), {chart_wrap_bg};
  border: 1px solid {chart_wrap_border};
  border-radius: 16px;
  padding: 0.35rem 0.45rem 0.2rem 0.45rem;
  box-shadow: {card_shadow};
  backdrop-filter: blur(8px);
}}

.stTabs [data-baseweb="tab-list"] {{
  gap: 0.4rem;
  padding-bottom: 0.1rem;
}}

.stTabs [data-baseweb="tab"] {{
  background: {tab_bg};
  border: 1px solid {tab_border};
  color: {tab_text_color} !important;
  border-radius: 11px 11px 0 0;
  padding-top: 0.42rem;
  padding-bottom: 0.48rem;
  transition: all 0.18s ease;
}}

.stTabs [data-baseweb="tab"]:hover {{
  background: {tab_hover_bg};
  border-color: {tab_active_border};
  transform: translateY(-1px);
}}

.stTabs [data-baseweb="tab"][aria-selected="true"] {{
  background: {tab_active_bg} !important;
  color: {tab_active_text_color} !important;
  border-color: {tab_active_border} !important;
  box-shadow: 0 10px 24px rgba(2, 6, 23, 0.18);
}}

div[data-testid="stVerticalBlock"] div[data-testid="stDataFrame"] {{
  border: 1px solid {frame_border};
  border-radius: 14px;
}}

[data-testid="stCaptionContainer"] p,
[data-testid="stCaptionContainer"] span {{
  color: {muted_color} !important;
}}

.sidebar-footer {{
  padding: 0.35rem 0.45rem 0.05rem 0.45rem;
  border-top: 1px solid {frame_border};
  font-size: 0.62rem;
  line-height: 1.2;
  opacity: 0.8;
}}

.sidebar-footer-wrap {{
  margin-top: auto;
  margin-left: 0.15rem;
  margin-right: 0.15rem;
  margin-bottom: 0.2rem;
  background: radial-gradient(65% 80% at 50% 0%, {card_glow} 0%, rgba(255,255,255,0) 100%), {card_bg};
  border: 1px solid {card_border};
  border-radius: 10px;
  box-shadow: {card_shadow};
  backdrop-filter: blur(8px);
}}
</style>
        """,
        unsafe_allow_html=True,
    )


def plotly_template() -> str:
    return "plotly_dark" if st.session_state.get("dark_mode", False) else "plotly_white"


def normalize_col_name(name: Any) -> str:
    return re.sub(r"\s+", " ", str(name).strip().lower())


def normalize_metric_name(value: Any) -> str | None:
    text = clean_text(value)
    if text is None:
        return None
    # Normalize different dash encodings to plain ASCII dash
    text = text.replace("???", "-").replace("???", "-").replace("????????", "-")
    text = text.replace("renevue", "revenue").replace("capcacity", "capacity").replace("employyes", "employees")
    text = text.replace("pr emp", "per employee")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def clean_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).replace("\xa0", " ").replace("\u202f", " ").strip()
    if not text:
        return None
    if text.lower() in {"nan", "none", "nat"}:
        return None
    return text


def to_float(value: Any) -> float:
    if isinstance(value, (int, np.integer)):
        return float(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if not math.isnan(float(value)) else np.nan

    text = clean_text(value)
    if text is None:
        return np.nan

    compact = text.replace(" ", "").replace("%", "").replace(",", ".")
    if compact in {"?", "-", "--"}:
        return np.nan
    try:
        return float(compact)
    except ValueError:
        return np.nan


def parse_mixed_date(value: Any) -> pd.Timestamp | pd.NaT:
    if isinstance(value, pd.Timestamp):
        return value.normalize()
    if isinstance(value, datetime):
        return pd.Timestamp(value).normalize()

    text = clean_text(value)
    if text is None:
        return pd.NaT

    compact = re.sub(r"\s+", "", text)

    if re.fullmatch(r"[+-]?\d+(\.\d+)?", compact):
        num = float(compact)
        if 20_000 <= num <= 70_000:
            ts = pd.Timestamp("1899-12-30") + pd.to_timedelta(num, unit="D")
            if 1990 <= ts.year <= 2035:
                return ts.normalize()
        return pd.NaT

    fix_year = re.fullmatch(r"(\d{1,2})[./-](\d{1,2})[./-](\d{5})", compact)
    if fix_year:
        day, month, year = fix_year.groups()
        text = f"{day}.{month}.{year[-4:]}"
    else:
        text = text.strip()

    iso_like = re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?", text)
    ts = pd.to_datetime(text, errors="coerce", dayfirst=not bool(iso_like))
    if pd.isna(ts):
        return pd.NaT
    if ts.year < 1990 or ts.year > 2035:
        return pd.NaT
    return pd.Timestamp(ts).normalize()


def normalize_country(value: Any) -> str:
    text = clean_text(value)
    if text is None:
        return "Unknown"
    norm = re.sub(r"\s+", " ", text).strip().lower()
    if norm in COUNTRY_MAP:
        return COUNTRY_MAP[norm]
    if norm in SUMMARY_LABELS:
        return "Unknown"
    if norm.startswith("#"):
        return "Unknown"
    return text


def map_country_for_geo(country: Any) -> str | None:
    text = clean_text(country)
    if text is None:
        return None
    text = GEO_COUNTRY_OVERRIDES.get(text, text)
    if text is None:
        return None
    return COUNTRY_ISO3_MAP.get(text)


def country_region_continent(country: Any) -> tuple[str, str]:
    text = clean_text(country)
    if text is None:
        return "Unknown", "Unknown"
    mapped = COUNTRY_REGION_CONTINENT.get(text)
    if mapped is not None:
        return mapped
    return COUNTRY_REGION_CONTINENT_LOWER.get(text.lower(), ("Unknown", "Unknown"))


def apply_region_continent_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "country" not in df.columns:
        return df

    out = df.copy()
    rc = out["country"].apply(country_region_continent)
    out["region"] = rc.apply(lambda t: t[0])
    out["continent"] = rc.apply(lambda t: t[1])
    return out


def render_sidebar_footer() -> None:
    st.sidebar.markdown(
        """
<div class="sidebar-footer-wrap">
  <div class="sidebar-footer">
  Unofficial dashboard - not affiliated with or endorsed by Siemens Gamesa. Data compiled manually from public Siemens Gamesa reports/announcements; verify against original sources. Provided "as is" without warranty.
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

def normalize_customer(value: Any) -> str:
    text = clean_text(value)
    if text is None:
        return "Unknown"
    norm = text.strip().lower()
    if norm in CUSTOMER_UNKNOWN_VALUES:
        return "Unknown"
    return text


def normalize_platform(value: Any) -> str:
    text = clean_text(value)
    if text is None:
        return "Unknown"
    text = text.upper().replace(",", ".")
    text = re.sub(r"\s+", "", text)
    if text in {"?", "NAN", "NONE"}:
        return "Unknown"
    return text


def parse_platform_specs(platform: str) -> tuple[float, float]:
    if platform == "Unknown":
        return np.nan, np.nan

    p = platform.upper().replace(" ", "")
    # SGRE patterns like "SG14-236" or "SWT-3.0-113" encode MW first and rotor second.
    sgre = re.search(r"(?:SG|SWT)-?([0-9]+(?:\.[0-9]+)?)-(\d{2,3})", p)
    if sgre:
        power = float(sgre.group(1))
        rotor = float(sgre.group(2))
        return rotor, power

    full = re.search(r"(?:V|N)(\d{2,3})[-/ ]([0-9]+(?:\.[0-9]+)?)(?:X|MW|KW)", p)
    if full:
        rotor = float(full.group(1))
        power = float(full.group(2))
        return rotor, power

    full_alt = re.search(r"(?:V|N)(\d{2,3})[-/ ]([0-9]+(?:\.[0-9]+)?)(MW|KW)", p)
    if full_alt:
        rotor = float(full_alt.group(1))
        power = float(full_alt.group(2))
        if full_alt.group(3) == "KW":
            power = power / 1000.0
        return rotor, power

    mw_only = re.search(r"([0-9]+(?:\.[0-9]+)?)(MW|KW)", p)
    if mw_only:
        power = float(mw_only.group(1))
        if mw_only.group(2) == "KW":
            power = power / 1000.0
        return np.nan, power

    return np.nan, np.nan

def first_text(row: pd.Series, candidates: list[str]) -> str | None:
    cols_norm = {normalize_col_name(col): col for col in row.index}
    for col in candidates:
        source_col = col if col in row.index else cols_norm.get(normalize_col_name(col))
        if source_col is not None:
            val = clean_text(row[source_col])
            if val is not None:
                return val
    return None


def first_float(row: pd.Series, candidates: list[str]) -> float:
    cols_norm = {normalize_col_name(col): col for col in row.index}
    for col in candidates:
        source_col = col if col in row.index else cols_norm.get(normalize_col_name(col))
        if source_col is not None:
            val = to_float(row[source_col])
            if not math.isnan(val):
                return val
    return np.nan


def quarter_end_date(year: int, quarter: int) -> pd.Timestamp:
    month = quarter * 3
    return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(1)


def parse_quarter_label(value: Any) -> int | None:
    text = clean_text(value)
    if text is None:
        return None
    token = re.sub(r"[^a-z0-9]", "", text.lower())
    match = re.fullmatch(r"q([1-4])", token)
    if match:
        return int(match.group(1))
    return None


def extract_order_block(df: pd.DataFrame) -> pd.DataFrame:
    cols_norm = {normalize_col_name(c): c for c in df.columns}
    country_check_col = cols_norm.get("country check") or cols_norm.get("check country")
    platform_check_col = cols_norm.get("platform check") or cols_norm.get("check platform")

    core_candidates = [
        "date",
        "country",
        "size [mw]",
        "mwsize",
        "service time [years]",
        "service time  [years]",
        "service_len",
        "customer",
        "#",
        "numberofturbines",
        "no_wtg_1",
        "platform",
        "platform.1",
        "platform.2",
        "platform.3",
        "turbinetype",
        "multi platform",
        "multi platform.1",
        "multi platform.2",
        "installation q",
        "installation y",
        "operation y",
        "oparetion q",
    ]
    core_cols = [cols_norm[c] for c in core_candidates if c in cols_norm]
    if not core_cols:
        return df.iloc[0:0].copy()

    if country_check_col or platform_check_col:
        keep = pd.Series(False, index=df.index)
        if country_check_col:
            keep |= df[country_check_col].notna()
        if platform_check_col:
            keep |= df[platform_check_col].notna()
        block = df.loc[keep].copy()
    else:
        nonempty = df[core_cols].notna().any(axis=1).to_numpy()
        if not nonempty.any():
            return df.iloc[0:0].copy()
        start = int(np.argmax(nonempty))
        run = 0
        cutoff = len(df)
        for idx in range(start, len(df)):
            if nonempty[idx]:
                run = 0
            else:
                run += 1
                if run >= 5:
                    cutoff = idx - run + 1
                    break
        block = df.iloc[start:cutoff].copy()

    block = block[block[core_cols].notna().any(axis=1)]
    block.reset_index(drop=True, inplace=True)
    return block

def parse_oi_sheet_year(sheet_name: Any) -> int | None:
    text = clean_text(sheet_name)
    if text is None:
        return None
    if re.fullmatch(r"\d{4}", text):
        year = int(text)
    else:
        m = re.fullmatch(r"OI[\s_-]*(\d{4})", text, flags=re.IGNORECASE)
        if m is None:
            return None
        year = int(m.group(1))
    if 1900 <= year <= 2100:
        return year
    return None


def find_economy_sheet_name(workbook: Path) -> str | None:
    xls = pd.ExcelFile(workbook)
    for cand in ["economy", "Economy", "Siemens Gamesa Economy", "SGRE Economy"]:
        if cand in xls.sheet_names:
            return cand
    for s in xls.sheet_names:
        if "econom" in s.lower():
            return s
    return None


def parse_year_cell(value: Any) -> int | None:
    if isinstance(value, (int, np.integer)):
        y = int(value)
        return y if 1990 <= y <= 2100 else None
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        y = int(value)
        return y if 1990 <= y <= 2100 else None
    text = clean_text(value)
    if text is None:
        return None
    if re.fullmatch(r"\d{4}", text):
        y = int(text)
        return y if 1990 <= y <= 2100 else None
    return None


def parse_fy_label_year(value: Any) -> int | None:
    text = clean_text(value)
    if text is None:
        return None
    match = re.fullmatch(r"FY\s*'?(\d{2,4})", text, flags=re.IGNORECASE)
    if match is None:
        return None
    y = int(match.group(1))
    if y < 100:
        y += 2000
    return y if 1990 <= y <= 2100 else None


def parse_economy_fy_table(workbook: Path) -> pd.DataFrame:
    sheet_name = find_economy_sheet_name(workbook)
    if sheet_name is None:
        return pd.DataFrame(columns=["metric", "year", "value"])

    raw = pd.read_excel(workbook, sheet_name=sheet_name, header=None, dtype=object)
    if raw.empty:
        return pd.DataFrame(columns=["metric", "year", "value"])

    scan_cols = min(12, raw.shape[1])  # User-provided FY block is in columns A-L.
    fy_header_row: int | None = None
    fy_year_map: dict[int, int] = {}
    for i in range(min(raw.shape[0], 140)):
        row_map: dict[int, int] = {}
        for j in range(scan_cols):
            y = parse_fy_label_year(raw.iat[i, j])
            if y is not None:
                row_map[j] = y
        if len(row_map) >= 4:
            fy_header_row = i
            fy_year_map = row_map

    if fy_header_row is None or not fy_year_map:
        return pd.DataFrame(columns=["metric", "year", "value"])

    rows: list[dict[str, Any]] = []
    blank_run = 0
    for r in range(fy_header_row + 1, min(raw.shape[0], fy_header_row + 90)):
        metric = normalize_metric_name(raw.iat[r, 0])
        if metric is None:
            blank_run += 1
            if blank_run >= 6:
                break
            continue
        blank_run = 0

        metric_low = metric.lower().strip()
        if metric_low in {"country", "in %"}:
            continue

        for c, y in fy_year_map.items():
            val = to_float(raw.iat[r, c])
            if math.isnan(val):
                continue
            rows.append({"metric": metric, "year": int(y), "value": float(val)})

    if not rows:
        return pd.DataFrame(columns=["metric", "year", "value"])

    fy = pd.DataFrame(rows)
    fy["year"] = pd.to_numeric(fy["year"], errors="coerce").astype("Int64")
    fy["value"] = pd.to_numeric(fy["value"], errors="coerce")
    fy = fy.dropna(subset=["metric", "year", "value"]).copy()
    fy["year"] = fy["year"].astype(int)
    fy = fy.groupby(["metric", "year"], as_index=False)["value"].mean()
    fy = fy.sort_values(["metric", "year"])
    return fy


def parse_quarterly_onshore_cost_per_mw(workbook: Path) -> pd.DataFrame:
    sheet_name = find_economy_sheet_name(workbook)
    if sheet_name is None:
        return pd.DataFrame(columns=["metric", "date", "year", "quarter", "value"])

    raw = pd.read_excel(workbook, sheet_name=sheet_name, header=None, dtype=object)
    if raw.empty:
        return pd.DataFrame(columns=["metric", "date", "year", "quarter", "value"])

    metric_row: int | None = None
    for i in range(min(raw.shape[0], 80)):
        label = (clean_text(raw.iat[i, 0]) or "").lower()
        if "average selling price" in label and "onshore" in label:
            metric_row = i
            break

    if metric_row is None:
        return pd.DataFrame(columns=["metric", "date", "year", "quarter", "value"])

    q_row: int | None = None
    q_cols: list[int] = []
    search_start = max(0, metric_row - 30)
    for i in range(search_start, metric_row):
        cols = []
        for c in range(min(20, raw.shape[1])):
            if parse_quarter_label(raw.iat[i, c]) is not None:
                cols.append(c)
        if len(cols) > len(q_cols):
            q_row = i
            q_cols = cols

    if q_row is None or len(q_cols) < 4:
        return pd.DataFrame(columns=["metric", "date", "year", "quarter", "value"])

    year_row: int | None = None
    best_count = 0
    for i in range(max(0, q_row - 8), q_row):
        count = 0
        for c in q_cols:
            if parse_year_cell(raw.iat[i, c]) is not None:
                count += 1
        if count > best_count:
            best_count = count
            year_row = i

    if year_row is None:
        return pd.DataFrame(columns=["metric", "date", "year", "quarter", "value"])

    year_markers: dict[int, int] = {}
    for c in range(min(20, raw.shape[1])):
        y = parse_year_cell(raw.iat[year_row, c])
        if y is not None:
            year_markers[c] = y
    if not year_markers:
        return pd.DataFrame(columns=["metric", "date", "year", "quarter", "value"])

    year_by_col: dict[int, int] = {}
    for c, y in sorted(year_markers.items()):
        q_here = parse_quarter_label(raw.iat[q_row, c])
        if q_here == 3:
            for offset, expected_q in [(-1, 4), (0, 3), (1, 2), (2, 1)]:
                cc = c + offset
                if cc < 0 or cc >= raw.shape[1]:
                    continue
                q_val = parse_quarter_label(raw.iat[q_row, cc])
                if q_val is None:
                    continue
                if q_val == expected_q:
                    year_by_col[cc] = y

    current_year: int | None = None
    for c in range(min(20, raw.shape[1])):
        if c in year_markers:
            current_year = year_markers[c]
        if current_year is not None and c not in year_by_col:
            year_by_col[c] = current_year

    metric_name = normalize_metric_name(raw.iat[metric_row, 0]) or "Average selling price onshore"
    rows: list[dict[str, Any]] = []
    for c in sorted(set(q_cols)):
        quarter = parse_quarter_label(raw.iat[q_row, c])
        year = year_by_col.get(c)
        value = to_float(raw.iat[metric_row, c])
        if quarter is None or year is None or math.isnan(value):
            continue
        rows.append(
            {
                "metric": metric_name,
                "date": quarter_end_date(int(year), int(quarter)),
                "year": int(year),
                "quarter": int(quarter),
                "value": float(value),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["metric", "date", "year", "quarter", "value"])

    out = pd.DataFrame(rows)
    out = out.drop_duplicates(subset=["year", "quarter"], keep="last")
    out = out.sort_values("date")
    return out


def parse_economy_sheet(workbook: Path) -> pd.DataFrame:
    sheet_name = find_economy_sheet_name(workbook)
    if sheet_name is None:
        return pd.DataFrame(columns=["metric", "year", "value"])

    raw = pd.read_excel(workbook, sheet_name=sheet_name, dtype=object)
    if raw.empty:
        return pd.DataFrame(columns=["metric", "year", "value"])

    rows: list[dict[str, Any]] = []
    nrows, ncols = raw.shape
    for i in range(nrows):
        row_vals = raw.iloc[i].tolist()
        year_cols: list[tuple[int, int]] = []
        for j, v in enumerate(row_vals):
            y = parse_year_cell(v)
            if y is not None:
                year_cols.append((j, y))
        if len(year_cols) < 5:
            continue

        first_year_col = min(j for j, _ in year_cols)
        year_map = {j: y for j, y in year_cols}
        blank_run = 0

        for k in range(i + 1, min(i + 60, nrows)):
            vals = raw.iloc[k].tolist()
            left = [clean_text(vals[c]) for c in range(first_year_col)]
            left_nonempty = [t for t in left if t is not None]

            metric = left_nonempty[-1] if left_nonempty else None
            if metric is None:
                blank_run += 1
                if blank_run >= 5:
                    break
                continue
            blank_run = 0

            metric_norm = normalize_metric_name(metric)
            if metric_norm is None:
                continue
            metric_low = metric_norm.lower().strip()
            if metric_low in {"country", "q1", "q2", "q3", "q4", "in %"}:
                continue
            if re.fullmatch(r"\d{4}", metric_low):
                continue

            group_label = left_nonempty[-2] if len(left_nonempty) >= 2 else None
            if group_label:
                group_norm = normalize_metric_name(group_label)
            else:
                group_norm = None

            metric_name = metric_norm
            if group_norm and group_norm.lower() not in {metric_low, "country"}:
                if metric_low not in {"total"}:
                    metric_name = f"{group_norm} - {metric_norm}"

            values: dict[int, float] = {}
            for col_idx, year in year_map.items():
                value = to_float(vals[col_idx])
                if not math.isnan(value):
                    values[year] = value
            if not values:
                continue

            rec = {"metric": metric_name}
            rec.update(values)
            rows.append(rec)

    if not rows:
        return pd.DataFrame(columns=["metric", "year", "value"])

    econ = pd.DataFrame(rows)
    value_cols = [c for c in econ.columns if isinstance(c, int)]
    if not value_cols:
        return pd.DataFrame(columns=["metric", "year", "value"])

    long = econ.melt(id_vars="metric", value_vars=value_cols, var_name="year", value_name="value")
    long = long.dropna(subset=["value"]).copy()
    long["year"] = pd.to_numeric(long["year"], errors="coerce").astype("Int64")
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["year", "value"])
    long["year"] = long["year"].astype(int)
    fy_block = parse_economy_fy_table(workbook)
    if not fy_block.empty:
        long = pd.concat([long, fy_block], ignore_index=True)

    long = long.groupby(["metric", "year"], as_index=False)["value"].mean()
    return long

def parse_oi_sheets(workbook: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    xl = pd.ExcelFile(workbook)
    oi_sheets = [(s, parse_oi_sheet_year(s)) for s in xl.sheet_names]
    oi_sheets = [(s, y) for s, y in oi_sheets if y is not None]
    oi_sheets = sorted(oi_sheets, key=lambda pair: pair[1])

    order_rows: list[dict[str, Any]] = []
    platform_rows: list[dict[str, Any]] = []

    for sheet, year in oi_sheets:
        raw = pd.read_excel(workbook, sheet_name=sheet, dtype=object)
        raw.columns = [str(c).strip() for c in raw.columns]
        block = extract_order_block(raw)

        for row_idx, row in block.iterrows():
            order_id = f"{year}_{row_idx:04d}"
            order_date = parse_mixed_date(first_text(row, ["date"]))
            size_mw = first_float(row, ["size [MW]", "MWsize", "size [mw]"])
            service_time_years = first_float(row, ["Service time  [years]", "Service time [years]", "service_len"])

            delivery_q = first_float(row, ["delivery q", "del q", "installation q", "operation q", "oparetion q"])
            delivery_year = first_float(row, ["delivery y", "del year", "installation y", "operation y"])
            delivery_days = first_float(row, ["Delivery days", "delivery days"])
            if math.isnan(delivery_q) or delivery_q not in {1.0, 2.0, 3.0, 4.0}:
                delivery_q = np.nan
            if math.isnan(delivery_year):
                delivery_year = np.nan
            elif delivery_year < 1990 or delivery_year > 2035:
                delivery_year = np.nan

            if math.isnan(delivery_days) and pd.notna(order_date) and not math.isnan(delivery_q) and not math.isnan(delivery_year):
                q_end = quarter_end_date(int(delivery_year), int(delivery_q))
                delivery_days = float((q_end - order_date).days)

            country = normalize_country(first_text(row, ["country"]))
            region, continent = country_region_continent(country)
            customer = normalize_customer(first_text(row, ["customer"]))

            order_data = {
                "order_id": order_id,
                "sheet_name": sheet,
                "sheet_year": year,
                "order_date": order_date,
                "order_year": int(order_date.year) if pd.notna(order_date) else year,
                "order_quarter": int(order_date.quarter) if pd.notna(order_date) else np.nan,
                "country": country,
                "region": region,
                "continent": continent,
                "service_time_years": service_time_years,
                "customer": customer,
                "size_mw": size_mw,
                "delivery_q": int(delivery_q) if not math.isnan(delivery_q) else np.nan,
                "delivery_year": int(delivery_year) if not math.isnan(delivery_year) else np.nan,
                "delivery_days": delivery_days,
            }

            row_platform_indices: list[int] = []

            for spec in PLATFORM_SLOT_SPECS:
                qty = first_float(row, spec["qty"])
                platform_raw = first_text(row, spec["platform"])
                rotor_raw = first_float(row, spec["rotor"])
                mw_raw = first_float(row, spec["mw"])

                if (
                    platform_raw is None
                    and math.isnan(qty)
                    and math.isnan(rotor_raw)
                    and math.isnan(mw_raw)
                ):
                    continue

                platform = normalize_platform(platform_raw)
                rotor_guess, mw_guess = parse_platform_specs(platform)

                rotor = rotor_raw if not math.isnan(rotor_raw) else rotor_guess
                mw_rating = mw_raw if not math.isnan(mw_raw) else mw_guess
                slot_mw = qty * mw_rating if (not math.isnan(qty) and not math.isnan(mw_rating)) else np.nan

                platform_rows.append(
                    {
                        "order_id": order_id,
                        "sheet_year": year,
                        "order_year": int(order_data["order_year"]),
                        "order_date": order_data["order_date"],
                        "country": country,
                        "region": region,
                        "continent": continent,
                        "service_time_years": service_time_years,
                        "customer": customer,
                        "delivery_q": order_data["delivery_q"],
                        "delivery_year": order_data["delivery_year"],
                        "delivery_days": order_data["delivery_days"],
                        "order_size_mw": size_mw,
                        "slot": spec["slot"],
                        "platform": platform,
                        "turbines_qty": qty,
                        "rotor_m": rotor,
                        "mw_rating": mw_rating,
                        "slot_mw": slot_mw,
                    }
                )
                row_platform_indices.append(len(platform_rows) - 1)

            if row_platform_indices:
                if len(row_platform_indices) == 1 and not math.isnan(size_mw):
                    idx = row_platform_indices[0]
                    if math.isnan(platform_rows[idx]["slot_mw"]):
                        platform_rows[idx]["slot_mw"] = size_mw
                        q = platform_rows[idx]["turbines_qty"]
                        p = platform_rows[idx]["mw_rating"]
                        if math.isnan(q) and not math.isnan(p) and p != 0:
                            platform_rows[idx]["turbines_qty"] = size_mw / p
                first_idx = row_platform_indices[0]
                order_data["primary_platform"] = platform_rows[first_idx]["platform"]
            else:
                fallback_platform = normalize_platform(first_text(row, ["TurbineType", "platform"]))
                order_data["primary_platform"] = fallback_platform if fallback_platform != "Unknown" else "Unknown"
                if not math.isnan(size_mw):
                    rotor_guess, mw_guess = parse_platform_specs(order_data["primary_platform"])
                    platform_rows.append(
                        {
                            "order_id": order_id,
                            "sheet_year": year,
                            "order_year": int(order_data["order_year"]),
                            "order_date": order_data["order_date"],
                            "country": country,
                            "region": region,
                            "continent": continent,
                            "service_time_years": service_time_years,
                            "customer": customer,
                            "delivery_q": order_data["delivery_q"],
                            "delivery_year": order_data["delivery_year"],
                            "delivery_days": order_data["delivery_days"],
                            "order_size_mw": size_mw,
                            "slot": 0,
                            "platform": order_data["primary_platform"],
                            "turbines_qty": first_float(row, ["NumberofTurbines"]),
                            "rotor_m": rotor_guess,
                            "mw_rating": mw_guess,
                            "slot_mw": size_mw,
                        }
                    )

            order_rows.append(order_data)

    orders = pd.DataFrame(order_rows)
    platforms = pd.DataFrame(platform_rows)
    orders = apply_region_continent_columns(orders)
    platforms = apply_region_continent_columns(platforms)
    if orders.empty:
        return orders, platforms

    for col in ["size_mw", "service_time_years", "delivery_days"]:
        orders[col] = pd.to_numeric(orders[col], errors="coerce")

    for col in ["turbines_qty", "rotor_m", "mw_rating", "slot_mw", "order_size_mw"]:
        if col in platforms.columns:
            platforms[col] = pd.to_numeric(platforms[col], errors="coerce")

    valid_ids = set(orders.loc[orders["size_mw"] > 0, "order_id"])
    valid_ids.update(platforms.loc[platforms["slot_mw"] > 0, "order_id"])
    orders = orders[orders["order_id"].isin(valid_ids)].copy()
    platforms = platforms[platforms["order_id"].isin(valid_ids)].copy()

    order_platform_mw = platforms.groupby("order_id")["slot_mw"].sum(min_count=1)
    order_turbines = platforms.groupby("order_id")["turbines_qty"].sum(min_count=1)
    orders["platform_mw_sum"] = orders["order_id"].map(order_platform_mw)
    orders["turbines_qty_sum"] = orders["order_id"].map(order_turbines)
    orders["mw_gap"] = orders["size_mw"] - orders["platform_mw_sum"]

    orders["delivery_time_years"] = orders["delivery_days"] / 365.25
    platforms["delivery_time_years"] = platforms["delivery_days"] / 365.25

    return orders, platforms

def find_default_workbook() -> Path | None:
    preferred = [
        Path("SG_data_new.xlsx"),
        Path("SG_data.xlsx"),
        Path("sg_data_new.xlsx"),
        Path("Siemens_Gamesa_data_new.xlsx"),
    ]
    for p in preferred:
        if p.exists() and not p.name.startswith("~$"):
            return p

    sgre_files = sorted(
        [
            f
            for f in Path(".").glob("*.xlsx")
            if not f.name.startswith("~$")
            and any(token in f.name.lower() for token in ["sg", "sgre", "siemens", "gamesa"])
        ]
    )
    if sgre_files:
        return sgre_files[0]

    patterns = ("*.xlsx", "*.xlsm", "*.xls")
    for pattern in patterns:
        files = sorted(Path(".").glob(pattern))
        files = [f for f in files if not f.name.startswith("~$")]
        if files:
            return files[0]
    return None

def empty_stock_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["date", "open", "high", "low", "close", "adj_close", "volume"])


def download_sgre_stock_monthly() -> pd.DataFrame:
    columns = ["date", "open", "high", "low", "close", "adj_close", "volume"]
    url = "https://companiesmarketcap.com/siemens-gamesa/stock-price-history/"
    try:
        resp = requests.get(url, timeout=25, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        page = resp.text
    except Exception:
        return pd.DataFrame(columns=columns)

    match = re.search(r"data\s*=\s*(\[\{.*?\}\]);\s*var\s+chart1", page, flags=re.IGNORECASE | re.DOTALL)
    if match is None:
        return pd.DataFrame(columns=columns)

    try:
        payload = json.loads(match.group(1))
    except Exception:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        ts = item.get("d")
        close = to_float(item.get("v"))
        if ts is None or math.isnan(close):
            continue
        try:
            date = pd.to_datetime(int(ts), unit="s", utc=True).tz_convert(None).normalize()
        except Exception:
            continue
        rows.append(
            {
                "date": date,
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "adj_close": close,
                "volume": 0.0,
            }
        )

    if not rows:
        return pd.DataFrame(columns=columns)

    stock = pd.DataFrame(rows)
    stock = stock.dropna(subset=["date", "close"]).copy()
    stock = stock.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return stock[columns].copy()


def refresh_sgre_stock_file(stock_path: Path) -> tuple[bool, str]:
    stock = download_sgre_stock_monthly()
    if stock.empty:
        return False, "Unable to download Siemens Gamesa stock history from the configured source."
    try:
        stock_path.parent.mkdir(parents=True, exist_ok=True)
        stock.to_json(stock_path, orient="records", date_format="iso")
    except Exception as exc:
        return False, f"Downloaded stock data but failed to save file: {exc}"
    return True, f"Saved {len(stock):,} monthly rows to `{stock_path}`."


def parse_stock_monthly_json(stock_path: Path) -> pd.DataFrame:
    if not stock_path.exists():
        return empty_stock_frame()

    try:
        raw = pd.read_json(stock_path)
    except Exception:
        return empty_stock_frame()

    if raw.empty:
        return empty_stock_frame()

    rename_map: dict[str, str] = {}
    for col in raw.columns:
        key = str(col).strip().lower()
        if "date" in key:
            rename_map[col] = "date"
        elif "adj" in key and "close" in key:
            rename_map[col] = "adj_close"
        elif "open" in key:
            rename_map[col] = "open"
        elif "high" in key:
            rename_map[col] = "high"
        elif "low" in key:
            rename_map[col] = "low"
        elif "close" in key:
            rename_map[col] = "close"
        elif "volume" in key:
            rename_map[col] = "volume"

    stock = raw.rename(columns=rename_map).copy()
    required_cols = ["date", "open", "high", "low", "close", "adj_close", "volume"]
    for col in required_cols:
        if col not in stock.columns:
            stock[col] = np.nan
    stock = stock[required_cols].copy()

    stock["date"] = pd.to_datetime(stock["date"], errors="coerce")
    for col in ["open", "high", "low", "close", "adj_close", "volume"]:
        stock[col] = pd.to_numeric(stock[col], errors="coerce")

    stock = stock.dropna(subset=["date", "open", "high", "low", "close"]).copy()
    stock = stock.sort_values("date")
    stock["date"] = stock["date"].dt.normalize()
    stock["volume"] = stock["volume"].fillna(0.0)
    return stock


def parse_market_monthly_json(market_path: Path) -> pd.DataFrame:
    if not market_path.exists():
        return pd.DataFrame(columns=["date", "steel_hrc_usd_per_short_ton", "copper_hg_usd_per_lb"])

    try:
        raw = pd.read_json(market_path)
    except Exception:
        return pd.DataFrame(columns=["date", "steel_hrc_usd_per_short_ton", "copper_hg_usd_per_lb"])

    if raw.empty:
        return pd.DataFrame(columns=["date", "steel_hrc_usd_per_short_ton", "copper_hg_usd_per_lb"])

    market = raw.copy()
    if "date" not in market.columns:
        for c in market.columns:
            if "date" in str(c).lower():
                market = market.rename(columns={c: "date"})
                break
    if "date" not in market.columns:
        return pd.DataFrame(columns=["date", "steel_hrc_usd_per_short_ton", "copper_hg_usd_per_lb"])

    for col in ["steel_hrc_usd_per_short_ton", "copper_hg_usd_per_lb"]:
        if col not in market.columns:
            market[col] = np.nan
        market[col] = pd.to_numeric(market[col], errors="coerce")

    market["date"] = pd.to_datetime(market["date"], errors="coerce").dt.normalize()
    market = market.dropna(subset=["date"]).sort_values("date")
    return market[["date", "steel_hrc_usd_per_short_ton", "copper_hg_usd_per_lb"]].copy()


@st.cache_data(show_spinner=False)
def load_data(
    cache_key: str,
    workbook_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cache_file = PARSED_DATA_FILE
    if cache_file.exists():
        try:
            payload = pd.read_pickle(cache_file)
            if isinstance(payload, dict) and payload.get("cache_key") == cache_key:
                return (
                    payload.get("economy", pd.DataFrame(columns=["metric", "year", "value"])),
                    payload.get("orders", pd.DataFrame()),
                    payload.get("platforms", pd.DataFrame()),
                    payload.get("economy_fy", pd.DataFrame(columns=["metric", "year", "value"])),
                    payload.get("quarterly_onshore_cost", pd.DataFrame(columns=["metric", "date", "year", "quarter", "value"])),
                )
        except Exception:
            pass

    workbook = Path(workbook_path)
    economy = parse_economy_sheet(workbook)
    economy_fy = parse_economy_fy_table(workbook)
    quarterly_onshore_cost = parse_quarterly_onshore_cost_per_mw(workbook)
    orders, platforms = parse_oi_sheets(workbook)

    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cache_key": cache_key,
            "workbook_path": str(workbook.resolve()),
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "economy": economy,
            "economy_fy": economy_fy,
            "quarterly_onshore_cost": quarterly_onshore_cost,
            "orders": orders,
            "platforms": platforms,
        }
        pd.to_pickle(payload, cache_file)
    except Exception:
        pass

    return economy, orders, platforms, economy_fy, quarterly_onshore_cost


@st.cache_data(show_spinner=False)
def load_stock_data(cache_key: str, stock_path: str) -> pd.DataFrame:
    _ = cache_key
    return parse_stock_monthly_json(Path(stock_path))


@st.cache_data(show_spinner=False)
def load_market_data(cache_key: str, market_path: str) -> pd.DataFrame:
    _ = cache_key
    return parse_market_monthly_json(Path(market_path))


def load_data_from_cache_only() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    if not PARSED_DATA_FILE.exists():
        return None
    try:
        payload = pd.read_pickle(PARSED_DATA_FILE)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return (
        payload.get("economy", pd.DataFrame(columns=["metric", "year", "value"])),
        payload.get("orders", pd.DataFrame()),
        payload.get("platforms", pd.DataFrame()),
        payload.get("economy_fy", pd.DataFrame(columns=["metric", "year", "value"])),
        payload.get("quarterly_onshore_cost", pd.DataFrame(columns=["metric", "date", "year", "quarter", "value"])),
    )


def parse_news_datetime(value: Any) -> pd.Timestamp | pd.NaT:
    text = clean_text(value)
    if text is None:
        return pd.NaT
    try:
        dt = parsedate_to_datetime(text)
        if dt is not None:
            return pd.Timestamp(dt).tz_convert("UTC").tz_localize(None) if dt.tzinfo else pd.Timestamp(dt)
    except Exception:
        pass
    ts = pd.to_datetime(text, errors="coerce", utc=True)
    if pd.isna(ts):
        return pd.NaT
    return ts.tz_convert("UTC").tz_localize(None)


def normalize_news_title(value: Any) -> str | None:
    text = clean_text(value)
    if text is None:
        return None
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def normalize_news_summary(value: Any) -> str | None:
    text = clean_text(value)
    if text is None:
        return None
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def short_summary(value: Any, max_chars: int = 320) -> str | None:
    text = normalize_news_summary(value)
    if text is None:
        return None
    if len(text) <= max_chars:
        return text
    clipped = text[:max_chars].rsplit(" ", 1)[0].strip()
    if not clipped:
        clipped = text[:max_chars].strip()
    return f"{clipped}..."


def is_nordex_relevant(
    title: str | None,
    summary: str | None,
    link: str | None,
    source_name: str | None = None,
) -> bool:
    source_txt = (source_name or "").lower()
    full = " ".join([title or "", summary or "", link or ""]).lower()
    if any(term in full for term in NON_TARGET_NEWS_TERMS):
        return False

    if any(bad in full for bad in NORDEX_NEWS_BLACKLIST):
        return False

    if source_txt.startswith("siemens gamesa (official"):
        return True

    title_txt = title or ""
    if not NORDEX_TITLE_RE.search(title_txt):
        return False

    # Filter broad roundup headlines where SGRE is one of many names.
    title_low = title_txt.lower()
    if title_low.count("|") >= 2 and not title_low.startswith(("siemens gamesa", "siemens-gamesa", "sgre")):
        return False
    if "siemens" in title_low and "gamesa" in title_low:
        return True
    return any(k in full for k in NORDEX_KEYWORDS)


def nordex_news_focus(news_type: Any) -> str:
    text = (clean_text(news_type) or "").lower()
    if "announcement" in text or "investor" in text:
        return "financial"
    if "press" in text:
        return "company"
    return "company"


def parse_nordex_company_news_archive() -> list[dict[str, Any]]:
    return []


def parse_rss_feed(feed_url: str, source_name: str, focus: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        resp = requests.get(feed_url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
    except Exception:
        return rows

    items = root.findall(".//item")
    atom_entries = root.findall(".//{http://www.w3.org/2005/Atom}entry")

    for node in items:
        title = normalize_news_title(node.findtext("title"))
        link = clean_text(node.findtext("link"))
        summary = normalize_news_summary(node.findtext("description"))
        pub_raw = (
            node.findtext("pubDate")
            or node.findtext("{http://purl.org/dc/elements/1.1/}date")
            or node.findtext("date")
        )
        published = parse_news_datetime(pub_raw)
        if title is None or link is None:
            continue
        source = source_name
        if source_name.startswith("Google News") and " - " in title:
            t_main, t_src = title.rsplit(" - ", 1)
            t_main = normalize_news_title(t_main)
            t_src = clean_text(t_src)
            if t_main is not None:
                title = t_main
            if t_src is not None:
                source = t_src
        if not is_nordex_relevant(title, summary, link, source_name=source):
            continue
        rows.append(
            {
                "source": source,
                "focus": focus,
                "title": title,
                "link": link,
                "summary": summary,
                "published": published,
            }
        )

    for node in atom_entries:
        title = normalize_news_title(node.findtext("{http://www.w3.org/2005/Atom}title"))
        link = None
        for link_node in node.findall("{http://www.w3.org/2005/Atom}link"):
            href = clean_text(link_node.attrib.get("href"))
            rel = clean_text(link_node.attrib.get("rel"))
            if href is None:
                continue
            if rel in {None, "alternate"}:
                link = href
                break
            if link is None:
                link = href

        pub_raw = (
            node.findtext("{http://www.w3.org/2005/Atom}updated")
            or node.findtext("{http://www.w3.org/2005/Atom}published")
        )
        summary = normalize_news_summary(
            node.findtext("{http://www.w3.org/2005/Atom}summary")
            or node.findtext("{http://www.w3.org/2005/Atom}content")
        )
        published = parse_news_datetime(pub_raw)
        if title is None or link is None:
            continue
        source = source_name
        if source_name.startswith("Google News") and " - " in title:
            t_main, t_src = title.rsplit(" - ", 1)
            t_main = normalize_news_title(t_main)
            t_src = clean_text(t_src)
            if t_main is not None:
                title = t_main
            if t_src is not None:
                source = t_src
        if not is_nordex_relevant(title, summary, link, source_name=source):
            continue
        rows.append(
            {
                "source": source,
                "focus": focus,
                "title": title,
                "link": link,
                "summary": summary,
                "published": published,
            }
        )
    return rows


def parse_nordex_orders_page() -> list[dict[str, Any]]:
    return []


@st.cache_data(show_spinner=False, ttl=1800)
def load_latest_news() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feed in NEWS_FEEDS:
        rows.extend(parse_rss_feed(feed["url"], feed["source"], feed["focus"]))

    if not rows:
        return pd.DataFrame(columns=["source", "focus", "title", "link", "summary", "published"])

    news = pd.DataFrame(rows)
    news = news[
        news.apply(
            lambda r: is_nordex_relevant(
                clean_text(r.get("title")),
                clean_text(r.get("summary")),
                clean_text(r.get("link")),
                clean_text(r.get("source")),
            ),
            axis=1,
        )
    ].copy()
    if news.empty:
        return pd.DataFrame(columns=["source", "focus", "title", "link", "summary", "published"])
    news["published"] = pd.to_datetime(news["published"], errors="coerce")
    news["title_key"] = news["title"].str.strip().str.lower()
    news["link_key"] = news["link"].str.strip().str.lower()
    news = news.sort_values(["published", "title"], ascending=[False, True])
    news = news.drop_duplicates(subset=["link_key"], keep="first")
    news = news.drop_duplicates(subset=["title_key"], keep="first")
    news = news.drop(columns=["title_key", "link_key"])

    max_per_source = 8
    selected_idx: list[int] = []
    source_counts: dict[str, int] = {}
    for idx, row in news.iterrows():
        src = str(row.get("source", "Unknown"))
        count = source_counts.get(src, 0)
        if count >= max_per_source:
            continue
        selected_idx.append(idx)
        source_counts[src] = count + 1
        if len(selected_idx) >= 20:
            break

    if len(selected_idx) < 20:
        for idx in news.index:
            if idx in selected_idx:
                continue
            selected_idx.append(idx)
            if len(selected_idx) >= 20:
                break

    return news.loc[selected_idx].copy()


@st.cache_data(show_spinner=False, ttl=43_200)
def load_latest_patents() -> pd.DataFrame:
    columns = [
        "title",
        "link",
        "publication_number",
        "publication_date",
        "filing_date",
        "priority_date",
        "inventor",
        "assignee",
        "summary",
    ]
    try:
        resp = requests.get(PATENTSCOPE_SEARCH_URL, timeout=25, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        page = resp.text
    except Exception:
        return pd.DataFrame(columns=columns)

    def html_text(value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = re.sub(r"<[^>]+>", " ", value)
        cleaned = html.unescape(cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned or None

    rows: list[dict[str, Any]] = []
    for row_html in re.findall(r"<tr data-ri=\"\d+\".*?</tr>", page, flags=re.IGNORECASE | re.DOTALL):
        pub_no_match = re.search(
            r"ps-patent-result--title--patent-number\">(.*?)</span>", row_html, flags=re.IGNORECASE | re.DOTALL
        )
        title_match = re.search(
            r"ps-patent-result--title--title[^>]*>(.*?)</span>\s*</span>",
            row_html,
            flags=re.IGNORECASE | re.DOTALL,
        )
        pub_date_match = re.search(
            r"resultListTableColumnPubDate\"[^>]*>(.*?)</span>", row_html, flags=re.IGNORECASE | re.DOTALL
        )
        link_match = re.search(r"<a href=\"([^\"]*detail\.jsf[^\"]+)\"", row_html, flags=re.IGNORECASE | re.DOTALL)
        applicant_match = re.search(
            r"ps-patent-result--applicant[^>]*>(.*?)</span>", row_html, flags=re.IGNORECASE | re.DOTALL
        )
        inventor_match = re.search(
            r"ps-patent-result--inventor[^>]*>(.*?)</span>", row_html, flags=re.IGNORECASE | re.DOTALL
        )
        abstract_match = re.search(
            r"ps-patent-result--abstract[^>]*>(.*?)</div>", row_html, flags=re.IGNORECASE | re.DOTALL
        )

        publication_number = html_text(pub_no_match.group(1)) if pub_no_match else None
        title = normalize_news_title(html_text(title_match.group(1)) if title_match else None)
        applicant = html_text(applicant_match.group(1)) if applicant_match else None
        inventor = html_text(inventor_match.group(1)) if inventor_match else None
        summary = normalize_news_summary(html_text(abstract_match.group(1)) if abstract_match else None)
        pub_date_raw = html_text(pub_date_match.group(1)) if pub_date_match else None
        publication_date = pd.to_datetime(pub_date_raw, format="%d.%m.%Y", errors="coerce")
        if pd.isna(publication_date):
            publication_date = pd.to_datetime(pub_date_raw, errors="coerce")

        if title is None or publication_number is None:
            continue
        if applicant is not None and not any(k in applicant.lower() for k in ["siemens", "gamesa", "sgre"]):
            continue

        link = None
        if link_match:
            href = html.unescape(link_match.group(1))
            href = re.sub(r";jsessionid=[^?]+", "", href)
            link = href if href.startswith("http") else f"https://patentscope.wipo.int/search/en/{href.lstrip('/')}"

        rows.append(
            {
                "title": title,
                "link": link,
                "publication_number": publication_number,
                "publication_date": publication_date,
                "filing_date": pd.NaT,
                "priority_date": pd.NaT,
                "inventor": inventor,
                "assignee": applicant,
                "summary": summary,
            }
        )

    if not rows:
        return pd.DataFrame(columns=columns)

    patents = pd.DataFrame(rows)
    patents = patents.drop_duplicates(subset=["publication_number", "title"], keep="first")
    patents = patents.sort_values(["publication_date", "title"], ascending=[False, True]).head(10)
    return patents


def latest_data_handled_text(
    economy: pd.DataFrame,
    orders: pd.DataFrame,
    stock_monthly: pd.DataFrame,
    market_monthly: pd.DataFrame,
) -> str:
    economy_max_year = int(economy["year"].max()) if economy is not None and not economy.empty else None
    orders_max_date = pd.to_datetime(orders["order_date"], errors="coerce").max() if orders is not None and not orders.empty else pd.NaT
    orders_max_year = int(orders["sheet_year"].max()) if orders is not None and not orders.empty else None
    stock_max = pd.to_datetime(stock_monthly["date"], errors="coerce").max() if stock_monthly is not None and not stock_monthly.empty else pd.NaT
    market_max = pd.to_datetime(market_monthly["date"], errors="coerce").max() if market_monthly is not None and not market_monthly.empty else pd.NaT

    orders_txt = "-"
    if pd.notna(orders_max_date):
        orders_txt = orders_max_date.strftime("%Y-%m-%d")
    elif orders_max_year is not None:
        orders_txt = str(orders_max_year)

    stock_txt = stock_max.strftime("%Y-%m-%d") if pd.notna(stock_max) else "-"
    market_txt = market_max.strftime("%Y-%m-%d") if pd.notna(market_max) else "-"
    econ_txt = str(economy_max_year) if economy_max_year is not None else "-"

    return (
        f"Latest handled data -> Economy: {econ_txt}; Orders: {orders_txt}; "
        f"Stock: {stock_txt}; Steel/Copper: {market_txt}. "
        "Not a live feed; updates occur when source files/feeds are refreshed."
    )


def mw_fmt(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{value:,.0f} MW"


def int_fmt(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{value:,.0f}"


def days_fmt(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{value:,.0f} days"


def top_n_by_mw(df: pd.DataFrame, col: str, value_col: str, n: int) -> list[str]:
    if df.empty:
        return []
    grouped = df.groupby(col, as_index=False)[value_col].sum()
    grouped = grouped.sort_values(value_col, ascending=False).head(n)
    return grouped[col].tolist()


def render_header_metrics(orders: pd.DataFrame, platforms: pd.DataFrame) -> None:
    total_orders = orders["order_id"].nunique()
    total_mw = orders["size_mw"].sum()
    total_platform_mw = platforms["slot_mw"].sum()
    avg_order_mw = orders["size_mw"].mean()
    avg_delivery_days = orders["delivery_days"].mean()
    countries = orders["country"].nunique()
    delivery_base = orders.dropna(subset=["delivery_days"]).copy()
    delivery_base_orders = delivery_base["order_id"].nunique()
    delivery_base_mw = delivery_base["size_mw"].sum()

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Orders", int_fmt(total_orders))
    m2.metric("Ordered MW", mw_fmt(total_mw))
    m3.metric("Platform-Mapped MW", mw_fmt(total_platform_mw))
    m4.metric(
        "Avg Order Size",
        mw_fmt(avg_order_mw),
        delta=f"based on {total_mw / 1000.0:,.1f} GW",
        delta_color="off",
    )
    m5.metric(
        "Avg Delivery Time",
        days_fmt(avg_delivery_days),
        delta=f"based on {delivery_base_mw / 1000.0:,.1f} GW / {delivery_base_orders:,.0f} orders",
        delta_color="off",
    )
    m6.metric("Countries", int_fmt(countries))


def render_information_page() -> None:
    st.subheader("Information")
    st.markdown("**Data source**")
    st.write("This dashboard is build on public available data.")

    st.markdown("**How to use and navigate**")
    st.write(
        "Use the sidebar filters to narrow years, countries, platforms, and minimum order size."
    )
    st.write(
        "Navigate tabs from high-level trends (Overall/Yearly/Quarterly) to deep dives (Across Years, Platform, Country, Delivery, Correlations)."
    )

    st.markdown("**Disclaimer**")
    st.write(
        "This dashboard is provided for informational purposes only. Data is compiled from public sources and may contain gaps, delays, or inaccuracies."
    )
    st.write(
        "Always verify critical figures with official company reporting before using the information for financial, legal, strategic, or investment decisions."
    )
    st.write(
        'The data and dashboard are provided "as is" without warranties of any kind. I make no guarantees of accuracy, completeness, or fitness for a particular purpose.'
    )
    st.write(
        "Use at your own risk. I am not responsible or liable for any losses or damages arising from use of the data or dashboard."
    )

    st.markdown("**Sources and attribution**")
    st.write(
        "All figures are compiled manually from Siemens Gamesa public investor communications (press releases/announcements) and annual reports."
    )
    st.write("Each datapoint should be verified against the original source documents.")

    st.markdown("**No affiliation / no endorsement**")
    st.write(
        "This is an unofficial, independent dashboard and is not affiliated with, endorsed by, or sponsored by Siemens Gamesa."
    )

    st.markdown("**IP and rights**")
    st.write(
        "The dataset in this repository is my own compilation (selection/structure) based on publicly available sources."
    )
    st.write("Underlying source documents remain the property of their respective owners.")

    st.markdown("**About me**")
    st.write(
        "Thomas Buhl is a wind-energy engineering leader with 20+ years across research and industry, including professor/department-head and director/VP roles."
    )
    st.write(
        "He combines deep technical expertise in wind turbine design and optimization with international people leadership and strategy execution."
    )


def render_latest_news() -> None:
    st.subheader("Latest News")
    st.caption("Showing the 20 latest items directly related to Siemens Gamesa.")
    st.caption("Sources include selected financial and wind-industry RSS/search feeds.")
    st.caption("Use links to open the original source articles.")

    news = load_latest_news()
    if news.empty:
        st.info("No news items could be loaded right now. Try again later.")
        return

    news = news.copy()
    news["published"] = pd.to_datetime(news["published"], errors="coerce")
    news["published_label"] = news["published"].dt.strftime("%Y-%m-%d %H:%M UTC")
    news["published_label"] = news["published_label"].fillna("Unknown date")

    for idx, row in enumerate(news.itertuples(index=False), start=1):
        title = clean_text(row.title) or "Untitled"
        link = clean_text(row.link) or ""
        source = clean_text(row.source) or "Unknown source"
        focus = clean_text(row.focus) or "news"
        when = clean_text(row.published_label) or "Unknown date"
        if link:
            st.markdown(f"{idx}. [{title}]({link})")
        else:
            st.write(f"{idx}. {title}")
        summary = short_summary(row.summary, max_chars=360)
        if summary is None or len(summary) < 80:
            summary = (
                f"This is a Siemens Gamesa-related {focus} update from {source}. "
                "The source feed did not provide a detailed excerpt, so open the original link for full context."
            )
        st.write(summary)
        st.caption(f"{source} | {focus} | {when}")


def render_latest_patents() -> None:
    st.subheader("Latest Patents")
    st.caption("Showing 10 most recent Siemens Gamesa patent publications from WIPO PATENTSCOPE.")
    st.caption("Inventor names are listed from filings; employment is inferred from applicant records.")

    patents = load_latest_patents()
    if patents.empty:
        st.info("No patent records could be loaded right now. Try again later.")
        return

    view = patents.copy()
    view["publication_label"] = pd.to_datetime(view["publication_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    view["filing_label"] = pd.to_datetime(view["filing_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    view["priority_label"] = pd.to_datetime(view["priority_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    view["publication_label"] = view["publication_label"].fillna("-")
    view["filing_label"] = view["filing_label"].fillna("-")
    view["priority_label"] = view["priority_label"].fillna("-")

    for idx, row in enumerate(view.itertuples(index=False), start=1):
        title = clean_text(row.title) or "Untitled patent record"
        link = clean_text(row.link) or ""
        publication_number = clean_text(row.publication_number) or "Unknown publication number"
        inventor = clean_text(row.inventor) or "Unknown inventor"
        assignee = clean_text(row.assignee) or "Unknown assignee"
        summary = short_summary(row.summary, max_chars=360)
        if summary is None:
            summary = "No abstract snippet available in the source response."

        if link:
            st.markdown(f"{idx}. [{title}]({link})")
        else:
            st.write(f"{idx}. {title}")
        st.write(summary)
        st.caption(
            f"{publication_number} | Published: {row.publication_label} | Filed: {row.filing_label} | Priority: {row.priority_label}"
        )
        st.caption(f"Inventor(s): {inventor} | Assignee: {assignee}")


def render_overall_economics(
    economy: pd.DataFrame,
    stock_monthly: pd.DataFrame,
    market_monthly: pd.DataFrame,
    economy_fy: pd.DataFrame,
    quarterly_onshore_cost: pd.DataFrame,
) -> None:
    st.subheader("Overall Economics")
    if economy.empty:
        st.info("No parsable data found in the economy sheet.")
        return

    years = sorted(economy["year"].unique().tolist())
    fy_years_all = sorted(economy_fy["year"].dropna().astype(int).unique().tolist()) if economy_fy is not None and not economy_fy.empty else []
    all_years = sorted(set(int(y) for y in years) | set(int(y) for y in fy_years_all))
    y_min, y_max = int(min(all_years)), int(max(all_years))
    year_range = st.slider("Year range (Economy + FY)", y_min, y_max, (max(y_min, 2007), y_max))
    econ = economy[economy["year"].between(year_range[0], year_range[1])].copy()
    if econ.empty:
        st.warning("No economic values available for the selected range.")
        return

    metric_options = sorted(econ["metric"].unique().tolist())

    def metric_key(text: str) -> str:
        norm = normalize_metric_name(text)
        if norm is None:
            return ""
        return re.sub(r"[^a-z0-9]+", "", norm.lower())

    def find_metric(required_tokens: list[str], any_tokens: list[str] | None = None) -> str | None:
        for metric in metric_options:
            key = metric_key(metric)
            if not all(tok in key for tok in required_tokens):
                continue
            if any_tokens is not None and not any(tok in key for tok in any_tokens):
                continue
            return metric
        return None

    def format_metric_value(metric: str, value: float) -> str:
        key = metric_key(metric)
        if "meurmw" in key:
            return f"{value:,.3f}"
        if "bneur" in key:
            return f"{value:,.2f}"
        if "mw" in key and "meurmw" not in key:
            return f"{value:,.0f}"
        return f"{value:,.0f}"

    def plot_metric_group(metrics: list[str | None], title: str, y_title: str, height: int = 380) -> None:
        chosen = [m for m in metrics if m is not None]
        if not chosen:
            st.info(f"No metrics found for: {title}")
            return
        frame = econ[econ["metric"].isin(chosen)].copy()
        if frame.empty:
            st.info(f"No values found for: {title}")
            return
        fig = px.line(
            frame,
            x="year",
            y="value",
            color="metric",
            markers=True,
            template=plotly_template(),
            title=title,
            height=height,
        )
        fig.update_layout(
            yaxis_title=y_title,
            legend_title_text="",
            margin=dict(l=10, r=10, t=60, b=10),
        )
        st.plotly_chart(fig, width="stretch")

    fy = economy_fy.copy() if economy_fy is not None else pd.DataFrame(columns=["metric", "year", "value"])
    if fy.empty:
        st.info("No parsable FY table detected in the economy sheet.")
    else:
        fy["year"] = pd.to_numeric(fy["year"], errors="coerce")
        fy["value"] = pd.to_numeric(fy["value"], errors="coerce")
        fy = fy.dropna(subset=["year", "value"]).copy()
        fy["year"] = fy["year"].astype(int)
        fy = fy.sort_values(["year", "metric"])
        fy = fy[fy["year"].between(year_range[0], year_range[1])].copy()

        fy_metrics = sorted(fy["metric"].unique().tolist())

        def find_fy_metric(required_tokens: list[str], any_tokens: list[str] | None = None) -> str | None:
            for metric in fy_metrics:
                key = metric_key(metric)
                if not all(tok in key for tok in required_tokens):
                    continue
                if any_tokens is not None and not any(tok in key for tok in any_tokens):
                    continue
                return metric
            return None

        fy_rev_total = find_fy_metric(["total", "revenue"])
        fy_rev_turbines = find_fy_metric(["revenue", "turbines"]) or find_fy_metric(["new", "units"])
        fy_rev_service = find_fy_metric(["revenue", "service"])
        fy_orders_total = find_fy_metric(["total", "order", "intake"])
        fy_orders_onshore = find_fy_metric(["total", "order", "onshore"])
        fy_orders_offshore = find_fy_metric(["total", "order", "offshore"])
        fy_orders_service = find_fy_metric(["order", "service"])
        fy_net_profit = find_fy_metric(["net", "profit"])
        fy_asp_onshore = find_fy_metric(["average", "selling", "price", "onshore"])
        fy_wtg_onshore = find_fy_metric(["wtg", "volume", "onshore"])
        fy_wtg_offshore = find_fy_metric(["wtg", "volume", "offshore"])

        fy_latest_year = int(fy["year"].max())
        fy_prev_year = fy_latest_year - 1
        fy_latest = fy[fy["year"] == fy_latest_year].set_index("metric")["value"]
        fy_prev = fy[fy["year"] == fy_prev_year].set_index("metric")["value"]

        key_cards = [
            ("Total revenue (EURm)", fy_rev_total),
            ("Revenue - Service (\u20acm)", fy_rev_service),
            ("Total order intake (EURm)", fy_orders_total),
            ("Net profit (EURm)", fy_net_profit),
        ]
        card_cols = st.columns(len(key_cards))
        for col, (label, metric_name) in zip(card_cols, key_cards, strict=False):
            latest_val = fy_latest.get(metric_name, np.nan) if metric_name is not None else np.nan
            prev_val = fy_prev.get(metric_name, np.nan) if metric_name is not None else np.nan
            delta_text = None
            if pd.notna(latest_val) and pd.notna(prev_val):
                delta_text = f"{latest_val - prev_val:+,.2f} vs FY{str(fy_prev_year)[-2:]}"
            if pd.isna(latest_val):
                value_html = "-"
            elif "net profit" in label.lower() and float(latest_val) < 0:
                value_html = f"<span style='color:#d64545'>{latest_val:,.2f}</span>"
            else:
                value_html = f"{latest_val:,.2f}"
            delta_html = html.escape(delta_text) if delta_text else "&nbsp;"
            col.markdown(
                f"""
<div class="fy-kpi-card">
  <div class="fy-kpi-title">{html.escape(label)}</div>
  <div class="fy-kpi-value">{value_html}</div>
  <div class="fy-kpi-delta">{delta_html}</div>
</div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height: 0.85rem;'></div>", unsafe_allow_html=True)

        fy_c1, fy_c2 = st.columns(2)
        with fy_c1:
            rev_metrics = [m for m in [fy_rev_total, fy_rev_turbines, fy_rev_service] if m is not None]
            if rev_metrics:
                rev_frame = fy[fy["metric"].isin(rev_metrics)].copy()
                fig_fy_rev = px.line(
                    rev_frame,
                    x="year",
                    y="value",
                    color="metric",
                    markers=True,
                    template=plotly_template(),
                    title="Revenue Mix by Financial Year",
                    height=390,
                    labels={"year": "FY", "value": "EURm", "metric": ""},
                )
                fig_fy_rev.update_layout(margin=dict(l=10, r=10, t=60, b=10))
                st.plotly_chart(fig_fy_rev, width="stretch")
            else:
                st.info("Revenue series not found in FY block.")

        with fy_c2:
            order_metrics = [m for m in [fy_orders_total, fy_orders_onshore, fy_orders_offshore, fy_orders_service] if m is not None]
            if order_metrics:
                order_frame = fy[fy["metric"].isin(order_metrics)].copy()
                fig_fy_orders = px.bar(
                    order_frame,
                    x="year",
                    y="value",
                    color="metric",
                    barmode="group",
                    template=plotly_template(),
                    title="Order Intake Breakdown by Financial Year",
                    height=390,
                    labels={"year": "FY", "value": "EURm", "metric": ""},
                )
                fig_fy_orders.update_layout(margin=dict(l=10, r=10, t=60, b=10))
                st.plotly_chart(fig_fy_orders, width="stretch")
            else:
                st.info("Order intake series not found in FY block.")

        fy_c3, fy_c4 = st.columns(2)
        with fy_c3:
            if fy_net_profit is not None:
                profit = fy[fy["metric"] == fy_net_profit].copy().sort_values("year")
                profit["sign"] = np.where(profit["value"] >= 0, "Positive", "Negative")
                fig_profit = px.bar(
                    profit,
                    x="year",
                    y="value",
                    color="sign",
                    template=plotly_template(),
                    title="Net Profit by Financial Year",
                    height=360,
                    labels={"year": "FY", "value": "EURm", "sign": ""},
                    color_discrete_map={"Positive": "#1f9d55", "Negative": "#d64545"},
                )
                fig_profit.update_layout(showlegend=False, margin=dict(l=10, r=10, t=60, b=10))
                st.plotly_chart(fig_profit, width="stretch")
            else:
                st.info("Net profit series not found in FY block.")

        with fy_c4:
            vol_metrics = [m for m in [fy_wtg_onshore, fy_wtg_offshore] if m is not None]
            if vol_metrics:
                vol_frame = fy[fy["metric"].isin(vol_metrics)].copy()
                fig_vol = px.area(
                    vol_frame,
                    x="year",
                    y="value",
                    color="metric",
                    template=plotly_template(),
                    title="WTG Delivered Volume (Onshore/Offshore)",
                    height=360,
                    labels={"year": "FY", "value": "MW", "metric": ""},
                )
                fig_vol.update_layout(margin=dict(l=10, r=10, t=60, b=10))
                st.plotly_chart(fig_vol, width="stretch")
            elif fy_asp_onshore is not None:
                asp = fy[fy["metric"] == fy_asp_onshore].copy().sort_values("year")
                fig_asp = px.line(
                    asp,
                    x="year",
                    y="value",
                    markers=True,
                    template=plotly_template(),
                    title="Onshore ASP by Financial Year",
                    height=360,
                    labels={"year": "FY", "value": "EURm/MW"},
                )
                fig_asp.update_layout(margin=dict(l=10, r=10, t=60, b=10))
                st.plotly_chart(fig_asp, width="stretch")
            else:
                st.info("No WTG volume or ASP series found in FY block.")

        st.markdown("**FY Custom Metric Explorer**")
        fy_default = [m for m in [fy_rev_total, fy_orders_total, fy_net_profit, fy_asp_onshore] if m is not None]
        fy_selected = st.multiselect(
            "Select FY metrics",
            options=fy_metrics,
            default=fy_default if fy_default else fy_metrics[:6],
            key="fy_metric_selector",
        )
        if fy_selected:
            fy_custom = fy[fy["metric"].isin(fy_selected)].copy()
            fig_fy_custom = px.line(
                fy_custom,
                x="year",
                y="value",
                color="metric",
                markers=True,
                template=plotly_template(),
                title="Custom FY Economics View",
                height=420,
            )
            fig_fy_custom.update_layout(legend_title_text="", margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig_fy_custom, width="stretch")

    q_cost = quarterly_onshore_cost.copy() if quarterly_onshore_cost is not None else pd.DataFrame(
        columns=["metric", "date", "year", "quarter", "value"]
    )
    if q_cost.empty:
        st.info("No quarterly onshore cost-per-MW series could be parsed.")
    else:
        q_cost["date"] = pd.to_datetime(q_cost["date"], errors="coerce")
        q_cost["year"] = pd.to_numeric(q_cost["year"], errors="coerce").astype("Int64")
        q_cost["quarter"] = pd.to_numeric(q_cost["quarter"], errors="coerce").astype("Int64")
        q_cost["value"] = pd.to_numeric(q_cost["value"], errors="coerce")
        q_cost = q_cost.dropna(subset=["date", "year", "quarter", "value"]).copy()
        q_cost["year"] = q_cost["year"].astype(int)
        q_cost["quarter"] = q_cost["quarter"].astype(int)
        q_cost = q_cost.sort_values("date")

        q_cost = q_cost[q_cost["year"].between(year_range[0], year_range[1])].copy()

        fig_q_line = px.line(
            q_cost,
            x="date",
            y="value",
            markers=True,
            template=plotly_template(),
            title="Onshore Cost per MW by Quarter",
            height=360,
            labels={"date": "Quarter", "value": "EURm/MW"},
        )
        fig_q_line.update_layout(margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig_q_line, width="stretch")

    revenue_metric = find_metric(["revenue"]) or find_metric(["gross", "revenue"])
    gross_profit_metric = find_metric(["gross", "profit"]) or find_metric(["ebit"])
    sales_projects_metric = find_metric(["sales", "projects"]) or find_metric(["sales"])
    sales_service_metric = find_metric(["sales", "service"]) or find_metric(["staff", "cost"])

    order_intake_beur_metric = find_metric(["order", "intake", "bneur"]) or find_metric(["order", "intake", "me"])
    order_backlog_wt_beur_metric = find_metric(["order", "backlog", "wind", "bneur"]) or find_metric(["order", "backlog"])
    order_backlog_service_beur_metric = find_metric(["order", "backlog", "service", "bneur"]) or find_metric(["ebit"])

    order_intake_mw_metric = find_metric(["order", "intake", "mw"]) or find_metric(["installed", "capacity", "mw"])
    order_backlog_mw_metric = find_metric(["order", "backlog", "wind", "mw"]) or find_metric(["number", "turbines"])
    deliveries_mw_metric = find_metric(["deliveries", "mw"]) or find_metric(["installed", "capacity", "mw"])

    avg_mw_price_metric = find_metric(["averagemwprice"]) or find_metric(["average", "mw", "price"])

    latest_year = int(econ["year"].max())
    base_year = int(year_range[0])
    latest = econ[econ["year"] == latest_year].set_index("metric")["value"]
    baseline = econ[econ["year"] == base_year].set_index("metric")["value"]
    card_metrics = [
        revenue_metric,
        gross_profit_metric,
        sales_projects_metric,
        order_intake_beur_metric,
        avg_mw_price_metric,
    ]
    card_metrics = [
        m
        for m in card_metrics
        if m is not None
        and "order backlog offshore" not in m.lower()
        and not ("revenue" in m.lower() and "service" in m.lower())
    ]

    if card_metrics:
        for idx in range(0, len(card_metrics), 4):
            row_metrics = card_metrics[idx : idx + 4]
            cols = st.columns(len(row_metrics))
            for col, metric_name in zip(cols, row_metrics, strict=False):
                val = latest.get(metric_name, np.nan)
                value_text = "-"
                if pd.notna(val):
                    value_text = f"{format_metric_value(metric_name, val)} ({latest_year})"
                base_val = baseline.get(metric_name, np.nan)
                delta_text = None
                if pd.notna(val) and pd.notna(base_val):
                    delta_value = val - base_val
                    delta_text = f"{delta_value:+,.2f} ({base_year})"
                col.metric(
                    metric_name,
                    value_text,
                    delta=delta_text,
                )

    g3, g4 = st.columns(2)
    with g3:
        plot_metric_group(
            [order_intake_mw_metric, order_backlog_mw_metric, deliveries_mw_metric],
            "MW Metrics (Order Intake / Capacity / Delivery)",
            "MW",
        )
    with g4:
        if avg_mw_price_metric is not None:
            plot_metric_group(
                [avg_mw_price_metric],
                "Average MW Price [mEUR/MW]",
                "mEUR/MW",
            )

    st.markdown("**Company Stock Price (Monthly OHLC)**")
    stock_col1, stock_col2 = st.columns([1, 3])
    with stock_col1:
        if st.button("Refresh SGRE Stock", key="refresh_sgre_stock_btn"):
            ok, msg = refresh_sgre_stock_file(STOCK_MONTHLY_FILE)
            if ok:
                st.cache_data.clear()
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)
    with stock_col2:
        st.caption("Source: companiesmarketcap.com (Siemens Gamesa stock history).")
    if stock_monthly is None or stock_monthly.empty:
        st.info("No local stock file found. Expected: `data/sgre_stock_monthly.json`.")
    else:
        stock = stock_monthly.copy()
        stock["year"] = stock["date"].dt.year
        stock = stock[stock["year"].between(year_range[0], year_range[1])].copy()
        if stock.empty:
            st.info("No stock data available for the selected economy year range.")
        else:
            overlay_options = [f"Economy: {m}" for m in sorted(metric_options)]
            market_series = {}
            steel_label = "Market: Steel price (HRC futures, USD/short ton)"
            copper_label = "Market: Copper price (HG futures, USD/lb)"
            if market_monthly is not None and not market_monthly.empty:
                market_series = {
                    steel_label: "steel_hrc_usd_per_short_ton",
                    copper_label: "copper_hg_usd_per_lb",
                }
                for label, col in list(market_series.items()):
                    if col not in market_monthly.columns or market_monthly[col].dropna().empty:
                        market_series.pop(label, None)
                overlay_options.extend(list(market_series.keys()))

            overlay_default: list[str] = []
            if "Economy: revenue (mEUR)" in overlay_options:
                overlay_default.append("Economy: revenue (mEUR)")
            if steel_label in overlay_options:
                overlay_default.append(steel_label)

            selected_overlays = st.multiselect(
                "Overlay on right axis (Y2/Y3)",
                options=overlay_options,
                default=overlay_default,
                key="stock_overlay_series",
            )

            fig_stock = go.Figure(
                data=[
                    go.Candlestick(
                        x=stock["date"],
                        open=stock["open"],
                        high=stock["high"],
                        low=stock["low"],
                        close=stock["close"],
                        name="SGRE",
                    )
                ]
            )
            y2_used = False
            y3_used = False
            for item in selected_overlays:
                if item.startswith("Economy: "):
                    metric_name = item.replace("Economy: ", "", 1)
                    metric_series = econ[econ["metric"] == metric_name][["year", "value"]].copy()
                    if metric_series.empty:
                        continue
                    metric_series["date"] = pd.to_datetime(metric_series["year"].astype(str) + "-12-31", errors="coerce")
                    metric_series = metric_series.dropna(subset=["date", "value"])
                    metric_series = metric_series[metric_series["date"].dt.year.between(year_range[0], year_range[1])]
                    if metric_series.empty:
                        continue
                    fig_stock.add_trace(
                        go.Scatter(
                            x=metric_series["date"],
                            y=metric_series["value"],
                            mode="lines+markers",
                            name=metric_name,
                            yaxis="y2",
                        )
                    )
                    y2_used = True
                elif item in market_series:
                    col = market_series[item]
                    market_f = market_monthly.copy()
                    market_f["date"] = pd.to_datetime(market_f["date"], errors="coerce")
                    market_f[col] = pd.to_numeric(market_f[col], errors="coerce")
                    market_f = market_f.dropna(subset=["date", col])
                    market_f = market_f[market_f["date"].dt.year.between(year_range[0], year_range[1])]
                    if market_f.empty:
                        continue
                    target_axis = "y3" if item == steel_label else "y2"
                    fig_stock.add_trace(
                        go.Scatter(
                            x=market_f["date"],
                            y=market_f[col],
                            mode="lines",
                            name=item.replace("Market: ", ""),
                            yaxis=target_axis,
                        )
                    )
                    if target_axis == "y3":
                        y3_used = True
                    else:
                        y2_used = True

            yaxis2_cfg: dict[str, Any] = {
                "title": "Overlay metrics (Y2)",
                "overlaying": "y",
                "side": "right",
                "showgrid": False,
                "visible": y2_used,
            }
            if y3_used:
                yaxis2_cfg["anchor"] = "free"
                yaxis2_cfg["position"] = 0.9

            fig_stock.update_layout(
                template=plotly_template(),
                title="Siemens Gamesa Monthly Price - Open/High/Low/Close",
                yaxis_title="Price",
                xaxis_title="Date",
                xaxis_rangeslider_visible=False,
                height=470,
                yaxis2=yaxis2_cfg,
                yaxis3=dict(
                    title="Steel price (Y3, USD/short ton)",
                    overlaying="y",
                    side="right",
                    showgrid=False,
                    visible=y3_used,
                    anchor="free",
                    position=0.995,
                ),
                margin=dict(l=10, r=10, t=60, b=10),
            )
            st.plotly_chart(fig_stock, width="stretch")
            st.caption("Source: local SGRE monthly history JSON + HRC=F (steel) and HG=F (copper).")

    st.markdown("**Custom Metric Explorer**")
    default_custom = [
        m
        for m in [
            revenue_metric,
            gross_profit_metric,
            sales_projects_metric,
            order_intake_beur_metric,
            order_intake_mw_metric,
            avg_mw_price_metric,
        ]
        if m is not None
    ]
    selected_metrics = st.multiselect(
        "Select custom metrics",
        options=metric_options,
        default=default_custom if default_custom else metric_options[:8],
    )
    custom_frame = econ[econ["metric"].isin(selected_metrics)].copy()
    if not custom_frame.empty:
        fig_custom = px.line(
            custom_frame,
            x="year",
            y="value",
            color="metric",
            markers=True,
            template=plotly_template(),
            height=460,
            title="Custom Economics View",
        )
        fig_custom.update_layout(legend_title_text="", margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig_custom, width="stretch")

def render_yearly_overview(orders: pd.DataFrame, platforms: pd.DataFrame) -> None:
    st.subheader("Year-by-Year Order Intake Overview")
    if orders.empty:
        st.info("No order intake rows after filtering.")
        return

    yearly = (
        orders.groupby("order_year", as_index=False)
        .agg(
            ordered_mw=("size_mw", "sum"),
            orders=("order_id", "nunique"),
            avg_order_mw=("size_mw", "mean"),
            countries=("country", "nunique"),
            customers=("customer", "nunique"),
        )
        .sort_values("order_year")
    )
    platform_yearly = platforms.groupby("order_year", as_index=False).agg(
        platform_mw=("slot_mw", "sum"),
        platforms=("platform", "nunique"),
    )
    yearly = yearly.merge(platform_yearly, on="order_year", how="left")

    c1, c2 = st.columns(2)
    with c1:
        fig1 = px.bar(
            yearly,
            x="order_year",
            y="ordered_mw",
            template=plotly_template(),
            labels={"order_year": "Year", "ordered_mw": "MW"},
            title="Ordered MW by Year",
            height=410,
        )
        fig1.update_layout(margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig1, width="stretch")

    with c2:
        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(
                x=yearly["order_year"],
                y=yearly["orders"],
                mode="lines+markers",
                name="Orders",
                line=dict(width=3),
            )
        )
        fig2.add_trace(
            go.Scatter(
                x=yearly["order_year"],
                y=yearly["avg_order_mw"],
                mode="lines+markers",
                name="Avg order MW",
                yaxis="y2",
                line=dict(width=3),
            )
        )
        fig2.update_layout(
            template=plotly_template(),
            title="Order Count and Avg Order Size",
            height=410,
            yaxis=dict(title="Orders / MW"),
            yaxis2=dict(title="Avg MW", overlaying="y", side="right"),
            margin=dict(l=10, r=10, t=60, b=10),
        )
        st.plotly_chart(fig2, width="stretch")

    continent_year = (
        orders.groupby(["order_year", "continent"], as_index=False)["size_mw"]
        .sum()
        .sort_values(["continent", "order_year"])
    )
    if not continent_year.empty:
        continent_year["cum_sold_mw"] = continent_year.groupby("continent")["size_mw"].cumsum()
        yearly_totals = (
            continent_year.groupby("order_year", as_index=False)["size_mw"]
            .sum()
            .rename(columns={"size_mw": "year_total_mw"})
        )
        continent_share = continent_year.merge(yearly_totals, on="order_year", how="left")
        continent_share = continent_share[continent_share["year_total_mw"] > 0].copy()
        continent_share["market_share_pct"] = 100.0 * continent_share["size_mw"] / continent_share["year_total_mw"]

        c5, c6 = st.columns(2)
        with c5:
            fig_continent_cum = px.area(
                continent_year,
                x="order_year",
                y="cum_sold_mw",
                color="continent",
                template=plotly_template(),
                labels={"order_year": "Year", "cum_sold_mw": "Accumulated MW", "continent": ""},
                title="Accumulated Sold MW by Continent",
                height=430,
            )
            fig_continent_cum.update_layout(margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig_continent_cum, width="stretch")

        with c6:
            fig_continent_share = px.bar(
                continent_share,
                x="order_year",
                y="market_share_pct",
                color="continent",
                barmode="stack",
                template=plotly_template(),
                labels={"order_year": "Year", "market_share_pct": "Market share (%)", "continent": ""},
                title="Continent Market Share by Year",
                hover_data={"size_mw": ":,.0f", "year_total_mw": ":,.0f", "market_share_pct": ":.1f"},
                height=430,
            )
            fig_continent_share.update_layout(
                yaxis=dict(range=[0, 100], ticksuffix="%"),
                margin=dict(l=10, r=10, t=60, b=10),
            )
            st.plotly_chart(fig_continent_share, width="stretch")

        st.caption("Accumulated chart uses announced order MW by continent.")

    st.dataframe(
        yearly.rename(
            columns={
                "order_year": "Year",
                "ordered_mw": "Ordered MW",
                "platform_mw": "Platform-Mapped MW",
                "avg_order_mw": "Avg Order MW",
                "orders": "Orders",
                "countries": "Countries",
                "customers": "Customers",
                "platforms": "Platforms",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    chosen_year = st.selectbox("Year detail", options=sorted(yearly["order_year"].unique().tolist(), reverse=True))
    yr_orders = orders[orders["order_year"] == chosen_year]
    yr_platforms = platforms[platforms["order_year"] == chosen_year]

    c3, c4 = st.columns(2)
    with c3:
        top_country = (
            yr_orders.groupby("country", as_index=False)["size_mw"]
            .sum()
            .sort_values("size_mw", ascending=False)
            .head(12)
        )
        fig3 = px.bar(
            top_country,
            x="size_mw",
            y="country",
            orientation="h",
            template=plotly_template(),
            title=f"Top Countries in {chosen_year}",
            labels={"size_mw": "MW", "country": ""},
            height=430,
        )
        fig3.update_layout(yaxis=dict(categoryorder="total ascending"), margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig3, width="stretch")

    with c4:
        top_platform = (
            yr_platforms.groupby("platform", as_index=False)["slot_mw"]
            .sum()
            .sort_values("slot_mw", ascending=False)
            .head(12)
        )
        fig4 = px.bar(
            top_platform,
            x="slot_mw",
            y="platform",
            orientation="h",
            template=plotly_template(),
            title=f"Top Platforms in {chosen_year}",
            labels={"slot_mw": "MW", "platform": ""},
            height=430,
        )
        fig4.update_layout(yaxis=dict(categoryorder="total ascending"), margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig4, width="stretch")


def render_quarterly_analytics(orders: pd.DataFrame) -> None:
    st.subheader("Quarterly Analytics")
    if orders.empty:
        st.info("No quarterly data available.")
        return

    announced = orders.dropna(subset=["order_date"]).copy()
    if not announced.empty:
        announced["year"] = announced["order_date"].dt.year.astype(int)
        announced["quarter"] = announced["order_date"].dt.quarter.astype(int)
        announced_q = (
            announced.groupby(["year", "quarter"], as_index=False)
            .agg(
                announced_mw=("size_mw", "sum"),
                orders=("order_id", "nunique"),
                avg_order_mw=("size_mw", "mean"),
                avg_service_time=("service_time_years", "mean"),
                avg_delivery_days=("delivery_days", "mean"),
            )
        )
    else:
        announced_q = pd.DataFrame(columns=["year", "quarter", "announced_mw", "orders", "avg_order_mw", "avg_service_time", "avg_delivery_days"])

    quarter_df = announced_q.copy()
    if quarter_df.empty:
        st.info("No quarterly data available.")
        return

    quarter_df["announced_mw"] = pd.to_numeric(quarter_df["announced_mw"], errors="coerce").fillna(0.0)
    quarter_df["orders"] = pd.to_numeric(quarter_df["orders"], errors="coerce").fillna(0).astype(int)
    quarter_df["total_mw"] = quarter_df["announced_mw"]
    quarter_df["quarter"] = pd.to_numeric(quarter_df["quarter"], errors="coerce").astype("Int64")
    quarter_df = quarter_df.dropna(subset=["year", "quarter"]).copy()
    quarter_df["year"] = quarter_df["year"].astype(int)
    quarter_df["quarter"] = quarter_df["quarter"].astype(int)
    quarter_df = quarter_df.sort_values(["year", "quarter"])
    quarter_df["quarter_label"] = quarter_df["quarter"].apply(lambda q: f"Q{q}")
    quarter_df["year_quarter"] = quarter_df["year"].astype(str) + "-" + quarter_df["quarter_label"]

    c1, c2 = st.columns(2)
    with c1:
        fig1 = px.bar(
            quarter_df,
            x="year_quarter",
            y="announced_mw",
            template=plotly_template(),
            labels={"year_quarter": "Quarter", "announced_mw": "MW"},
            title="Quarterly Ordered MW",
            height=420,
        )
        fig1.update_layout(xaxis_tickangle=-45, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig1, width="stretch")

    with c2:
        pivot = quarter_df.pivot(index="year", columns="quarter_label", values="total_mw").fillna(0)
        fig2 = px.imshow(
            pivot,
            text_auto=".0f",
            color_continuous_scale="YlOrRd",
            aspect="auto",
            title="Total MW Heatmap (Year x Quarter)",
            height=420,
        )
        fig2.update_layout(margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig2, width="stretch")

    c3, c4 = st.columns(2)
    with c3:
        by_quarter = (
            quarter_df.groupby("quarter_label", as_index=False)
            .agg(
                avg_announced_mw=("announced_mw", "mean"),
                avg_total_mw=("total_mw", "mean"),
            )
            .sort_values("quarter_label")
        )
        fig3 = px.bar(
            by_quarter,
            x="quarter_label",
            y=["avg_announced_mw", "avg_total_mw"],
            barmode="group",
            template=plotly_template(),
            title="Average Quarterly MW by Quarter Number",
            labels={"value": "Average MW", "quarter_label": "Quarter"},
            height=420,
        )
        fig3.update_layout(legend_title_text="", margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig3, width="stretch")

    with c4:
        corr_cols = ["announced_mw", "total_mw", "orders", "avg_order_mw", "avg_service_time", "avg_delivery_days"]
        corr_src = quarter_df[corr_cols].dropna()
        if len(corr_src) >= 4:
            corr = corr_src.corr(numeric_only=True)
            fig4 = px.imshow(
                corr,
                text_auto=".2f",
                zmin=-1,
                zmax=1,
                color_continuous_scale="RdBu",
                title="Quarterly Correlation Matrix",
                height=420,
            )
            fig4.update_layout(margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig4, width="stretch")
        else:
            st.info("Not enough complete quarterly rows for a correlation matrix.")

    st.dataframe(
        quarter_df.rename(
            columns={
                "year": "Year",
                "quarter_label": "Quarter",
                "announced_mw": "Announced MW",
                "total_mw": "Total MW",
                "orders": "Orders",
                "avg_order_mw": "Avg Order MW",
                "avg_service_time": "Avg Service Time",
                "avg_delivery_days": "Avg Delivery Days",
            }
        ),
        width="stretch",
        hide_index=True,
    )


def render_across_years(orders: pd.DataFrame, platforms: pd.DataFrame) -> None:
    st.subheader("Across-Years Analytics")
    if orders.empty:
        st.info("No order intake rows after filtering.")
        return

    tab_country, tab_platform, tab_service_time, tab_rotor, tab_customer = st.tabs(
        ["Country Stats", "Platform Stats", "Contract Time", "Rotor and MW Rating", "Customer Stats"]
    )

    with tab_country:
        top_n = st.slider("Top countries to display", 5, 20, 12, key="top_country_years")
        top_countries = top_n_by_mw(orders, "country", "size_mw", top_n)
        country_year = (
            orders[orders["country"].isin(top_countries)]
            .groupby(["order_year", "country"], as_index=False)["size_mw"]
            .sum()
        )
        fig = px.area(
            country_year,
            x="order_year",
            y="size_mw",
            color="country",
            template=plotly_template(),
            labels={"size_mw": "MW", "order_year": "Year"},
            title="Country MW Trends Across Years",
            height=440,
        )
        fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig, width="stretch")

        c_geo1, c_geo2 = st.columns(2)
        with c_geo1:
            continent_year = (
                orders.groupby(["order_year", "continent"], as_index=False)["size_mw"]
                .sum()
                .sort_values("order_year")
            )
            if continent_year.empty:
                st.info("No continent data available for current filters.")
            else:
                fig_cont = px.bar(
                    continent_year,
                    x="order_year",
                    y="size_mw",
                    color="continent",
                    template=plotly_template(),
                    labels={"size_mw": "MW", "order_year": "Year"},
                    title="Continent MW Trends Across Years",
                    height=430,
                )
                fig_cont.update_layout(margin=dict(l=10, r=10, t=60, b=10))
                st.plotly_chart(fig_cont, width="stretch")

        with c_geo2:
            top_regions = top_n_by_mw(orders[orders["region"] != "Unknown"], "region", "size_mw", 10)
            region_year = (
                orders[orders["region"].isin(top_regions)]
                .groupby(["order_year", "region"], as_index=False)["size_mw"]
                .sum()
                .sort_values("order_year")
            )
            if region_year.empty:
                st.info("No region data available for current filters.")
            else:
                fig_reg = px.area(
                    region_year,
                    x="order_year",
                    y="size_mw",
                    color="region",
                    template=plotly_template(),
                    labels={"size_mw": "MW", "order_year": "Year"},
                    title="Top Regions MW Trends Across Years",
                    height=430,
                )
                fig_reg.update_layout(margin=dict(l=10, r=10, t=60, b=10))
                st.plotly_chart(fig_reg, width="stretch")

        table = (
            orders.groupby(["country", "region", "continent"], as_index=False)
            .agg(total_mw=("size_mw", "sum"), orders=("order_id", "nunique"), avg_order_mw=("size_mw", "mean"))
            .sort_values("total_mw", ascending=False)
            .head(20)
        )
        st.dataframe(table, width="stretch", hide_index=True)

    with tab_platform:
        top_n = st.slider("Top platforms to display", 5, 20, 12, key="top_platform_years")
        p_base = platforms[platforms["platform"] != "Unknown"]
        top_platforms = top_n_by_mw(p_base, "platform", "slot_mw", top_n)
        platform_year = (
            p_base[p_base["platform"].isin(top_platforms)]
            .groupby(["order_year", "platform"], as_index=False)["slot_mw"]
            .sum()
        )
        fig = px.area(
            platform_year,
            x="order_year",
            y="slot_mw",
            color="platform",
            template=plotly_template(),
            labels={"slot_mw": "MW", "order_year": "Year"},
            title="Platform MW Trends Across Years",
            height=440,
        )
        fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig, width="stretch")

        table = (
            p_base.groupby("platform", as_index=False)
            .agg(
                total_mw=("slot_mw", "sum"),
                orders=("order_id", "nunique"),
                avg_delivery_days=("delivery_days", "mean"),
            )
            .sort_values("total_mw", ascending=False)
            .head(20)
        )
        st.dataframe(table, width="stretch", hide_index=True)

    with tab_service_time:
        c1, c2 = st.columns(2)

        with c1:
            time_year = (
                orders.dropna(subset=["service_time_years"])
                .groupby("order_year", as_index=False)
                .agg(
                    avg_service_time=("service_time_years", "mean"),
                    min_service_time=("service_time_years", "min"),
                    max_service_time=("service_time_years", "max"),
                )
            )
            fig = px.line(
                time_year,
                x="order_year",
                y=["avg_service_time", "min_service_time", "max_service_time"],
                template=plotly_template(),
                markers=True,
                labels={"value": "Years", "order_year": "Year"},
                title="Contract Length Across Years (avg/min/max)",
                height=430,
            )
            fig.update_layout(legend_title_text="", margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig, width="stretch")

        with c2:
            svc_time = orders.dropna(subset=["service_time_years"]).copy()
            if svc_time.empty:
                st.info("No contract-time values available for current filters.")
            else:
                fig_box = px.box(
                    svc_time,
                    x="order_year",
                    y="service_time_years",
                    template=plotly_template(),
                    title="Contract Time Distribution by Year",
                    labels={"service_time_years": "Years", "order_year": "Year"},
                    height=430,
                )
                fig_box.update_layout(margin=dict(l=10, r=10, t=60, b=10))
                st.plotly_chart(fig_box, width="stretch")

    with tab_rotor:
        rot = platforms[(platforms["platform"] != "Unknown") & (platforms["slot_mw"] > 0)].copy()
        rotor_stats = (
            rot.dropna(subset=["rotor_m"])
            .groupby("order_year", as_index=False)
            .agg(rotor_avg=("rotor_m", "mean"), rotor_min=("rotor_m", "min"), rotor_max=("rotor_m", "max"))
        )
        mw_stats = (
            rot.dropna(subset=["mw_rating"])
            .groupby("order_year", as_index=False)
            .agg(mw_avg=("mw_rating", "mean"), mw_min=("mw_rating", "min"), mw_max=("mw_rating", "max"))
        )

        c1, c2 = st.columns(2)
        with c1:
            fig_r = go.Figure()
            fig_r.add_trace(go.Scatter(x=rotor_stats["order_year"], y=rotor_stats["rotor_max"], mode="lines", line=dict(width=0), showlegend=False))
            fig_r.add_trace(
                go.Scatter(
                    x=rotor_stats["order_year"],
                    y=rotor_stats["rotor_min"],
                    mode="lines",
                    fill="tonexty",
                    fillcolor="rgba(43,119,173,0.22)",
                    line=dict(width=0),
                    name="Min-Max band",
                )
            )
            fig_r.add_trace(
                go.Scatter(
                    x=rotor_stats["order_year"],
                    y=rotor_stats["rotor_avg"],
                    mode="lines+markers",
                    line=dict(width=3, color="#1f5b86"),
                    name="Average",
                )
            )
            fig_r.update_layout(
                template=plotly_template(),
                title="Rotor Size Across Years (min/avg/max)",
                yaxis_title="Rotor diameter (m)",
                height=430,
                margin=dict(l=10, r=10, t=60, b=10),
            )
            st.plotly_chart(fig_r, width="stretch")

        with c2:
            fig_m = go.Figure()
            fig_m.add_trace(go.Scatter(x=mw_stats["order_year"], y=mw_stats["mw_max"], mode="lines", line=dict(width=0), showlegend=False))
            fig_m.add_trace(
                go.Scatter(
                    x=mw_stats["order_year"],
                    y=mw_stats["mw_min"],
                    mode="lines",
                    fill="tonexty",
                    fillcolor="rgba(241,120,50,0.20)",
                    line=dict(width=0),
                    name="Min-Max band",
                )
            )
            fig_m.add_trace(
                go.Scatter(
                    x=mw_stats["order_year"],
                    y=mw_stats["mw_avg"],
                    mode="lines+markers",
                    line=dict(width=3, color="#ca6318"),
                    name="Average",
                )
            )
            fig_m.update_layout(
                template=plotly_template(),
                title="Platform MW Rating Across Years (min/avg/max)",
                yaxis_title="MW rating",
                height=430,
                margin=dict(l=10, r=10, t=60, b=10),
            )
            st.plotly_chart(fig_m, width="stretch")

    with tab_customer:
        cust_base = orders[orders["customer"] != "Unknown"].copy()
        top_n = st.slider("Top customers to display", 5, 25, 12, key="top_customers_years")
        top_customers = top_n_by_mw(cust_base, "customer", "size_mw", top_n)
        by_year = (
            cust_base[cust_base["customer"].isin(top_customers)]
            .groupby(["order_year", "customer"], as_index=False)["size_mw"]
            .sum()
        )
        fig = px.bar(
            by_year,
            x="order_year",
            y="size_mw",
            color="customer",
            template=plotly_template(),
            title="Customer MW Bought Across Years",
            labels={"size_mw": "MW", "order_year": "Year"},
            height=430,
        )
        fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig, width="stretch")

        table = (
            cust_base.groupby("customer", as_index=False)
            .agg(total_mw=("size_mw", "sum"), orders=("order_id", "nunique"), avg_order_mw=("size_mw", "mean"))
            .sort_values("total_mw", ascending=False)
            .head(25)
        )
        st.dataframe(table, width="stretch", hide_index=True)


def render_platform_lens(platforms: pd.DataFrame) -> None:
    st.subheader("Across Platforms")
    if platforms.empty:
        st.info("No platform-level rows after filtering.")
        return

    base = platforms[(platforms["platform"] != "Unknown") & (platforms["slot_mw"] > 0)].copy()
    if base.empty:
        st.info("No platform rows with MW available.")
        return

    options = sorted(base["platform"].unique().tolist())
    default_sel = top_n_by_mw(base, "platform", "slot_mw", min(6, len(options)))
    selected = st.multiselect("Platforms to focus", options=options, default=default_sel)
    view = base[base["platform"].isin(selected)] if selected else base

    timeline = view.groupby(["order_year", "platform"], as_index=False)["slot_mw"].sum()
    fig = px.line(
        timeline,
        x="order_year",
        y="slot_mw",
        color="platform",
        template=plotly_template(),
        markers=True,
        title="Timeline: Sold MW per Platform per Year",
        labels={"slot_mw": "MW", "order_year": "Year"},
        height=430,
    )
    fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(fig, width="stretch")

    svc_time = view.dropna(subset=["service_time_years"])
    if not svc_time.empty:
        fig_b = px.box(
            svc_time,
            x="platform",
            y="service_time_years",
            template=plotly_template(),
            title="Across Platforms: Contract Time Distribution",
            labels={"service_time_years": "Contract time (years)", "platform": ""},
            height=460,
        )
        fig_b.update_layout(xaxis_tickangle=-45, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig_b, width="stretch")

    c3, c4 = st.columns(2)
    with c3:
        platform_country = view.groupby(["platform", "country"], as_index=False)["slot_mw"].sum()
        top_pc = platform_country.sort_values("slot_mw", ascending=False).head(140)
        fig_t = px.treemap(
            top_pc,
            path=["platform", "country"],
            values="slot_mw",
            template=plotly_template(),
            title="Across Platforms: Country Mix",
            height=520,
        )
        fig_t.update_layout(margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig_t, width="stretch")

    with c4:
        focus_platform = st.selectbox("Customer view by platform", options=sorted(view["platform"].unique().tolist()))
        customer_data = (
            view[(view["platform"] == focus_platform) & (view["customer"] != "Unknown")]
            .groupby("customer", as_index=False)["slot_mw"]
            .sum()
            .sort_values("slot_mw", ascending=False)
            .head(15)
        )
        fig_c = px.bar(
            customer_data,
            x="slot_mw",
            y="customer",
            orientation="h",
            template=plotly_template(),
            title=f"Across Platforms: Top Customers for {focus_platform}",
            labels={"slot_mw": "MW", "customer": ""},
            height=520,
        )
        fig_c.update_layout(yaxis=dict(categoryorder="total ascending"), margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig_c, width="stretch")

    platform_delivery = (
        view.dropna(subset=["delivery_days"])
        .groupby("platform", as_index=False)
        .agg(avg_days=("delivery_days", "mean"), median_days=("delivery_days", "median"), orders=("order_id", "nunique"))
        .sort_values("avg_days", ascending=False)
    )
    if not platform_delivery.empty:
        fig_d = px.bar(
            platform_delivery.head(20),
            x="avg_days",
            y="platform",
            orientation="h",
            template=plotly_template(),
            title="Across Platforms: Average Delivery Time",
            labels={"avg_days": "Days", "platform": ""},
            height=470,
        )
        fig_d.update_layout(yaxis=dict(categoryorder="total ascending"), margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig_d, width="stretch")


def render_country_maps(
    orders: pd.DataFrame,
    platforms: pd.DataFrame,
    selected_orders: pd.DataFrame,
    selected_platforms: pd.DataFrame,
) -> None:
    st.markdown("### Zoomable Country Maps")
    map_scope = st.radio(
        "Map scope",
        options=["All countries", "Selected countries"],
        index=0,
        horizontal=True,
        key="country_map_scope",
    )
    if map_scope == "Selected countries":
        map_orders = selected_orders.copy()
        map_platforms = selected_platforms.copy()
    else:
        map_orders = orders.copy()
        map_platforms = platforms.copy()

    map_orders["size_mw"] = pd.to_numeric(map_orders["size_mw"], errors="coerce")
    map_orders["service_time_years"] = pd.to_numeric(map_orders["service_time_years"], errors="coerce")
    map_orders["delivery_days"] = pd.to_numeric(map_orders["delivery_days"], errors="coerce")

    map_base = (
        map_orders.groupby("country", as_index=False)
        .agg(
            total_mw=("size_mw", "sum"),
            orders=("order_id", "nunique"),
            avg_order_mw=("size_mw", "mean"),
            avg_service_time=("service_time_years", "mean"),
            min_service_time=("service_time_years", "min"),
            max_service_time=("service_time_years", "max"),
            avg_delivery_days=("delivery_days", "mean"),
            median_delivery_days=("delivery_days", "median"),
        )
        .copy()
    )
    rc = map_base["country"].apply(country_region_continent)
    map_base["region"] = rc.apply(lambda t: t[0])
    map_base["continent"] = rc.apply(lambda t: t[1])

    valid_map_platforms = map_platforms[(map_platforms["platform"] != "Unknown") & (map_platforms["slot_mw"] > 0)].copy()
    platform_rank = (
        valid_map_platforms.groupby(["country", "platform"], as_index=False)["slot_mw"]
        .sum()
        .sort_values(["country", "slot_mw"], ascending=[True, False])
        .drop_duplicates("country")
        .rename(columns={"platform": "dominant_platform", "slot_mw": "dominant_platform_mw"})
    )
    map_base = map_base.merge(
        platform_rank[["country", "dominant_platform", "dominant_platform_mw"]],
        on="country",
        how="left",
    )
    map_base["dominant_platform"] = map_base["dominant_platform"].fillna("Unknown")

    platform_options = ["All platforms"]
    if not valid_map_platforms.empty:
        platform_options.extend(sorted(valid_map_platforms["platform"].dropna().unique().tolist()))
    selected_platform = st.selectbox(
        "Platform overlay for maps",
        options=platform_options,
        index=0,
        key="country_map_platform_overlay",
    )

    if valid_map_platforms.empty:
        overlay = pd.DataFrame(columns=["country", "platform_overlay_mw"])
    elif selected_platform == "All platforms":
        overlay = (
            valid_map_platforms.groupby("country", as_index=False)["slot_mw"]
            .sum()
            .rename(columns={"slot_mw": "platform_overlay_mw"})
        )
    else:
        overlay = (
            valid_map_platforms[valid_map_platforms["platform"] == selected_platform]
            .groupby("country", as_index=False)["slot_mw"]
            .sum()
            .rename(columns={"slot_mw": "platform_overlay_mw"})
        )

    map_base = map_base.merge(overlay, on="country", how="left")
    map_base["platform_overlay_mw"] = pd.to_numeric(map_base["platform_overlay_mw"], errors="coerce").fillna(0.0)
    map_base["iso3"] = map_base["country"].map(map_country_for_geo)
    map_geo = map_base[map_base["iso3"].notna()].copy()

    if map_geo.empty:
        st.info("No mappable countries available for the current scope.")
        return

    map_metric_labels = {
        "total_mw": "Total sold MW",
        "platform_overlay_mw": f"Sold MW for {selected_platform}",
        "orders": "Number of orders",
        "avg_service_time": "Average service length (years)",
        "avg_delivery_days": "Average delivery time (days)",
    }
    size_metric_labels = {
        "total_mw": "Total sold MW",
        "platform_overlay_mw": f"Sold MW for {selected_platform}",
        "orders": "Number of orders",
    }
    color_metric_labels = {
        "avg_service_time": "Average service length (years)",
        "avg_delivery_days": "Average delivery time (days)",
        "orders": "Number of orders",
    }

    map_style = st.selectbox(
        "Map style",
        options=["Bubble map", "Choropleth map"],
        index=0,
        key="country_map_style",
    )

    map_geo["orders"] = pd.to_numeric(map_geo["orders"], errors="coerce")

    if map_style == "Bubble map":
        c5, c6 = st.columns(2)
        with c5:
            bubble_size_metric = st.selectbox(
                "Bubble size metric",
                options=list(size_metric_labels.keys()),
                format_func=lambda k: size_metric_labels[k],
                key="country_map_bubble_size_metric",
            )
        with c6:
            bubble_color_metric = st.selectbox(
                "Bubble color metric",
                options=list(color_metric_labels.keys()),
                format_func=lambda k: color_metric_labels[k],
                key="country_map_bubble_color_metric",
            )

        map_geo[bubble_size_metric] = pd.to_numeric(map_geo[bubble_size_metric], errors="coerce")
        map_geo[bubble_color_metric] = pd.to_numeric(map_geo[bubble_color_metric], errors="coerce")

        bubble_data = map_geo.dropna(subset=[bubble_size_metric]).copy()
        bubble_data = bubble_data[bubble_data[bubble_size_metric] > 0]
        if bubble_data[bubble_color_metric].notna().any():
            bubble_data = bubble_data.dropna(subset=[bubble_color_metric]).copy()
        else:
            bubble_color_metric = "orders"
            bubble_data = bubble_data.dropna(subset=[bubble_color_metric]).copy()

        if bubble_data.empty:
            st.info("No values available for the selected bubble map metrics.")
        else:
            bubble_data["_bubble_size"] = bubble_data[bubble_size_metric].clip(lower=0.1)
            fig_bubble = px.scatter_geo(
                bubble_data,
                locations="iso3",
                locationmode="ISO-3",
                size="_bubble_size",
                color=bubble_color_metric,
                hover_name="country",
                hover_data={
                    "total_mw": ":,.0f",
                    "platform_overlay_mw": ":,.0f",
                    "orders": ":,.0f",
                    "avg_service_time": ":.2f",
                    "avg_delivery_days": ":,.0f",
                    "region": True,
                    "continent": True,
                    "dominant_platform": True,
                    "_bubble_size": False,
                    "iso3": False,
                },
                size_max=45,
                projection="natural earth",
                template=plotly_template(),
                color_continuous_scale="Turbo",
                title=(
                    f"Zoomable Bubble Map: size={size_metric_labels[bubble_size_metric]}, "
                    f"color={color_metric_labels[bubble_color_metric]}"
                ),
                height=600,
            )
            fig_bubble.update_geos(showframe=False, showcoastlines=True, fitbounds="locations", bgcolor="rgba(0,0,0,0)")
            fig_bubble.update_layout(
                coloraxis_colorbar_title=color_metric_labels[bubble_color_metric],
                margin=dict(l=10, r=10, t=60, b=10),
            )
            st.plotly_chart(fig_bubble, width="stretch")
    else:
        choropleth_metric = st.selectbox(
            "Choropleth metric",
            options=list(map_metric_labels.keys()),
            format_func=lambda k: map_metric_labels[k],
            key="country_map_choropleth_metric",
        )
        map_geo[choropleth_metric] = pd.to_numeric(map_geo[choropleth_metric], errors="coerce")
        choropleth_data = map_geo.dropna(subset=[choropleth_metric]).copy()
        if choropleth_metric in {"total_mw", "platform_overlay_mw", "orders"}:
            choropleth_data = choropleth_data[choropleth_data[choropleth_metric] > 0]

        if choropleth_data.empty:
            st.info("No values available for the selected choropleth metric.")
        else:
            fig_map = px.choropleth(
                choropleth_data,
                locations="iso3",
                locationmode="ISO-3",
                color=choropleth_metric,
                hover_name="country",
                hover_data={
                    "total_mw": ":,.0f",
                    "platform_overlay_mw": ":,.0f",
                    "orders": ":,.0f",
                    "avg_service_time": ":.2f",
                    "avg_delivery_days": ":,.0f",
                    "region": True,
                    "continent": True,
                    "dominant_platform": True,
                    "iso3": False,
                },
                template=plotly_template(),
                color_continuous_scale="YlGnBu",
                title=f"Zoomable Choropleth: {map_metric_labels[choropleth_metric]}",
                height=600,
            )
            fig_map.update_geos(showframe=False, showcoastlines=True, fitbounds="locations", bgcolor="rgba(0,0,0,0)")
            fig_map.update_layout(
                coloraxis_colorbar_title=map_metric_labels[choropleth_metric],
                margin=dict(l=10, r=10, t=60, b=10),
            )
            st.plotly_chart(fig_map, width="stretch")


def render_country_lens(orders: pd.DataFrame, platforms: pd.DataFrame) -> None:
    st.subheader("Across Countries")
    if orders.empty:
        st.info("No country-level rows after filtering.")
        return

    country_options = sorted(orders["country"].unique().tolist())
    default = top_n_by_mw(orders, "country", "size_mw", min(6, len(country_options)))
    selected_countries = st.multiselect("Countries to focus", options=country_options, default=default)
    if selected_countries:
        o = orders[orders["country"].isin(selected_countries)].copy()
        p = platforms[platforms["country"].isin(selected_countries)].copy()
    else:
        o = orders.copy()
        p = platforms.copy()

    if o.empty:
        st.info("No rows available for selected countries.")
        return

    render_country_maps(orders, platforms, o, p)

    c1, c2 = st.columns(2)
    with c1:
        p_valid = p[(p["platform"] != "Unknown") & (p["slot_mw"] > 0)]
        top_platforms = top_n_by_mw(p_valid, "platform", "slot_mw", 10)
        mix = (
            p_valid[p_valid["platform"].isin(top_platforms)]
            .groupby(["country", "platform"], as_index=False)["slot_mw"]
            .sum()
        )
        fig = px.bar(
            mix,
            x="country",
            y="slot_mw",
            color="platform",
            template=plotly_template(),
            title="Across Country: Platform Mix",
            labels={"slot_mw": "MW", "country": ""},
            height=430,
        )
        fig.update_layout(xaxis_tickangle=-30, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig, width="stretch")

    with c2:
        scheme = o.groupby("country", as_index=False).agg(orders=("order_id", "nunique"), size_mw=("size_mw", "sum"))
        fig2 = px.bar(
            scheme,
            x="country",
            y="orders",
            template=plotly_template(),
            title="Across Country: Number of Orders",
            labels={"orders": "Orders", "country": ""},
            height=430,
        )
        fig2.update_layout(xaxis_tickangle=-30, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig2, width="stretch")

    c3, c4 = st.columns(2)
    with c3:
        service_time = (
            o.dropna(subset=["service_time_years"])
            .groupby("country", as_index=False)
            .agg(avg_service_time=("service_time_years", "mean"), median_service_time=("service_time_years", "median"))
            .sort_values("avg_service_time", ascending=False)
        )
        fig3 = px.bar(
            service_time,
            x="avg_service_time",
            y="country",
            orientation="h",
            template=plotly_template(),
            title="Across Country: Avg Service Time",
            labels={"avg_service_time": "Years", "country": ""},
            height=430,
        )
        fig3.update_layout(yaxis=dict(categoryorder="total ascending"), margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig3, width="stretch")

    with c4:
        delivery = (
            o.dropna(subset=["delivery_days"])
            .groupby("country", as_index=False)
            .agg(avg_delivery_days=("delivery_days", "mean"), median_delivery_days=("delivery_days", "median"), orders=("order_id", "nunique"))
            .sort_values("avg_delivery_days", ascending=False)
        )
        fig4 = px.bar(
            delivery,
            x="avg_delivery_days",
            y="country",
            orientation="h",
            template=plotly_template(),
            title="Across Country: Delivery Time",
            labels={"avg_delivery_days": "Days", "country": ""},
            height=430,
        )
        fig4.update_layout(yaxis=dict(categoryorder="total ascending"), margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig4, width="stretch")

    c5, c6 = st.columns(2)
    with c5:
        continent_mw = (
            o.groupby("continent", as_index=False)
            .agg(total_mw=("size_mw", "sum"), orders=("order_id", "nunique"))
            .sort_values("total_mw", ascending=False)
        )
        fig5 = px.bar(
            continent_mw,
            x="continent",
            y="total_mw",
            color="continent",
            template=plotly_template(),
            title="Across Countries: Continent Totals",
            labels={"total_mw": "MW", "continent": ""},
            height=430,
        )
        fig5.update_layout(showlegend=False, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig5, width="stretch")

    with c6:
        region_stats = (
            o.groupby("region", as_index=False)
            .agg(
                total_mw=("size_mw", "sum"),
                avg_service_time=("service_time_years", "mean"),
                avg_delivery_days=("delivery_days", "mean"),
            )
            .sort_values("total_mw", ascending=False)
            .head(12)
        )
        region_stats = region_stats.dropna(subset=["avg_service_time", "avg_delivery_days"])
        if region_stats.empty:
            st.info("No complete region-level service/delivery data for current filters.")
        else:
            fig6 = px.scatter(
                region_stats,
                x="avg_service_time",
                y="avg_delivery_days",
                size="total_mw",
                color="region",
                template=plotly_template(),
                title="Across Countries: Region Service vs Delivery",
                labels={"avg_service_time": "Avg service time (years)", "avg_delivery_days": "Avg delivery days"},
                height=430,
            )
            fig6.update_layout(margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig6, width="stretch")


def render_delivery_capacity(orders: pd.DataFrame) -> None:
    st.subheader("Installed Capacity and Delivery")
    if orders.empty:
        st.info("No delivery rows after filtering.")
        return

    delivery = orders.dropna(subset=["delivery_year", "size_mw"]).copy()
    if delivery.empty:
        st.info("No delivery year information available in filtered data.")
        return

    delivery["delivery_year"] = delivery["delivery_year"].astype(int)
    if "delivery_q" in delivery.columns:
        delivery["delivery_q"] = pd.to_numeric(delivery["delivery_q"], errors="coerce")

    by_year = delivery.groupby("delivery_year", as_index=False)["size_mw"].sum()
    by_quarter = (
        delivery.dropna(subset=["delivery_q"])
        .groupby(["delivery_year", "delivery_q"], as_index=False)["size_mw"]
        .sum()
        .sort_values(["delivery_year", "delivery_q"])
    )

    c1, c2 = st.columns(2)
    with c1:
        fig1 = px.bar(
            by_year,
            x="delivery_year",
            y="size_mw",
            template=plotly_template(),
            title="Across Years: Installed Capacity (by delivery year)",
            labels={"size_mw": "MW", "delivery_year": "Delivery year"},
            height=430,
        )
        fig1.update_layout(margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig1, width="stretch")

    with c2:
        if not by_quarter.empty:
            by_quarter["delivery_q"] = by_quarter["delivery_q"].astype(int).astype(str).radd("Q")
            fig2 = px.bar(
                by_quarter,
                x="delivery_year",
                y="size_mw",
                color="delivery_q",
                template=plotly_template(),
                title="Across Years: Installed Capacity by Quarter",
                labels={"size_mw": "MW", "delivery_year": "Delivery year", "delivery_q": "Quarter"},
                height=430,
            )
            fig2.update_layout(margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig2, width="stretch")

    if not by_quarter.empty:
        pivot = by_quarter.pivot(index="delivery_year", columns="delivery_q", values="size_mw").fillna(0)
        fig_h = px.imshow(
            pivot,
            text_auto=".0f",
            aspect="auto",
            color_continuous_scale="YlGnBu",
            title="Installed Capacity Heatmap (Year x Quarter)",
            height=420,
        )
        fig_h.update_layout(margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig_h, width="stretch")

    delivery_time = (
        orders.dropna(subset=["delivery_days"])
        .groupby("order_year", as_index=False)
        .agg(
            avg_delivery_days=("delivery_days", "mean"),
            median_delivery_days=("delivery_days", "median"),
            p90_delivery_days=("delivery_days", lambda s: s.quantile(0.9)),
            orders=("order_id", "nunique"),
        )
        .sort_values("order_year")
    )
    if not delivery_time.empty:
        fig_t = px.line(
            delivery_time,
            x="order_year",
            y=["avg_delivery_days", "median_delivery_days", "p90_delivery_days"],
            template=plotly_template(),
            markers=True,
            title="Across Years: Delivery Time Statistics",
            labels={"value": "Days", "order_year": "Order year"},
            height=430,
        )
        fig_t.update_layout(legend_title_text="", margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig_t, width="stretch")


def render_correlations(orders: pd.DataFrame) -> None:
    st.subheader("Additional Correlations")
    if orders.empty:
        st.info("No rows for correlation analysis.")
        return

    numeric_cols = ["size_mw", "service_time_years", "delivery_days", "platform_mw_sum", "turbines_qty_sum"]
    corr_data = orders[numeric_cols].dropna()
    if len(corr_data) >= 12:
        corr = corr_data.corr(numeric_only=True)
        fig = px.imshow(
            corr,
            text_auto=".2f",
            zmin=-1,
            zmax=1,
            color_continuous_scale="RdBu",
            title="Numeric Correlation Matrix",
            height=420,
        )
        fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig, width="stretch")

    scatter_data = orders.dropna(subset=["size_mw", "delivery_days"]).copy()
    if not scatter_data.empty:
        scatter_data["service_time_years"] = pd.to_numeric(scatter_data["service_time_years"], errors="coerce")
        scatter_data["continent"] = scatter_data["continent"].fillna("Unknown")

        scatter_args: dict[str, Any] = {
            "data_frame": scatter_data,
            "x": "size_mw",
            "y": "delivery_days",
            "color": "continent",
            "hover_data": ["country", "customer", "primary_platform", "order_year", "region"],
            "template": plotly_template(),
            "title": "Delivery Time vs Order Size",
            "labels": {"size_mw": "Order size (MW)", "delivery_days": "Delivery days"},
            "height": 500,
        }

        if scatter_data["service_time_years"].notna().any():
            positive_vals = scatter_data.loc[scatter_data["service_time_years"] > 0, "service_time_years"]
            fallback_size = float(positive_vals.median()) if not positive_vals.empty else 1.0
            scatter_data["_bubble_size"] = scatter_data["service_time_years"].fillna(fallback_size).clip(lower=0.1)
            scatter_args["size"] = "_bubble_size"
            scatter_args["size_max"] = 38

        try:
            fig2 = px.scatter(**scatter_args)
        except ValueError:
            scatter_args.pop("size", None)
            scatter_args.pop("size_max", None)
            fig2 = px.scatter(**scatter_args)
            st.caption("Bubble size disabled for this view due missing/invalid service-time values.")
        fig2.update_layout(margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig2, width="stretch")

    if len(corr_data) >= 12:
        flat_corr = (
            corr.where(~np.tril(np.ones(corr.shape)).astype(bool))
            .stack()
            .reset_index()
            .rename(columns={"level_0": "Metric A", "level_1": "Metric B", 0: "Correlation"})
        )
        flat_corr["AbsCorr"] = flat_corr["Correlation"].abs()
        flat_corr = flat_corr.sort_values("AbsCorr", ascending=False).drop(columns=["AbsCorr"]).head(10)
        st.dataframe(flat_corr, width="stretch", hide_index=True)

def apply_global_filters(
    orders: pd.DataFrame, platforms: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if orders.empty:
        return orders, platforms

    orders = apply_region_continent_columns(orders)
    platforms = apply_region_continent_columns(platforms)

    st.sidebar.header("Global Filters")
    y_min, y_max = int(orders["order_year"].min()), int(orders["order_year"].max())
    year_range = st.sidebar.slider("Order year range", y_min, y_max, (y_min, y_max))

    continents = sorted([c for c in orders["continent"].dropna().unique().tolist() if c != "Unknown"])
    regions = sorted([r for r in orders["region"].dropna().unique().tolist() if r != "Unknown"])
    countries = sorted(orders["country"].dropna().unique().tolist())
    platform_options = sorted(platforms["platform"].dropna().unique().tolist()) if not platforms.empty else []

    selected_continents = st.sidebar.multiselect("Continents", options=continents, default=[])
    selected_regions = st.sidebar.multiselect("Regions", options=regions, default=[])
    selected_countries = st.sidebar.multiselect("Countries", options=countries, default=[])
    selected_platforms = st.sidebar.multiselect("Platforms", options=platform_options, default=[])

    min_order_mw = float(np.nanmin(orders["size_mw"])) if orders["size_mw"].notna().any() else 0.0
    max_order_mw = float(np.nanmax(orders["size_mw"])) if orders["size_mw"].notna().any() else 1.0
    mw_floor = st.sidebar.slider("Minimum order MW", min_value=float(min_order_mw), max_value=float(max_order_mw), value=float(min_order_mw))

    o = orders[
        orders["order_year"].between(year_range[0], year_range[1]) & (orders["size_mw"].fillna(0) >= mw_floor)
    ].copy()

    if selected_continents:
        o = o[o["continent"].isin(selected_continents)]
    if selected_regions:
        o = o[o["region"].isin(selected_regions)]
    if selected_countries:
        o = o[o["country"].isin(selected_countries)]

    p = platforms[platforms["order_id"].isin(o["order_id"])].copy()

    if selected_platforms:
        p = p[p["platform"].isin(selected_platforms)]
        o = o[o["order_id"].isin(p["order_id"])]

    st.sidebar.caption(
        "Siemens-Gamesa is as of 2023 no longer reporting seperate financial annual report but is rather part of Siemens Energy"
    )

    return o, p


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="W", layout="wide")
    if "dark_mode" not in st.session_state:
        st.session_state["dark_mode"] = False
    st.sidebar.toggle("Dark mode", key="dark_mode")
    dark_mode = bool(st.session_state.get("dark_mode", False))
    apply_page_style(dark_mode)

    st.title(APP_TITLE)
    st.caption("Build on public available data.")

    workbook = find_default_workbook()
    if workbook is None:
        cached = load_data_from_cache_only()
        if cached is None:
            st.error("No Excel workbook found and no parsed cache available (`data_cache/sgre_parsed_data.pkl`).")
            return
        economy, orders, platforms, economy_fy, quarterly_onshore_cost = cached
    else:
        stat = workbook.stat()
        cache_key = f"{workbook.resolve()}::{stat.st_mtime_ns}::{stat.st_size}"
        economy, orders, platforms, economy_fy, quarterly_onshore_cost = load_data(cache_key, str(workbook))

    if not STOCK_MONTHLY_FILE.exists():
        refresh_sgre_stock_file(STOCK_MONTHLY_FILE)

    if STOCK_MONTHLY_FILE.exists():
        stock_stat = STOCK_MONTHLY_FILE.stat()
        stock_cache_key = f"{STOCK_MONTHLY_FILE.resolve()}::{stock_stat.st_mtime_ns}::{stock_stat.st_size}"
    else:
        stock_cache_key = "missing_stock_file"
    stock_monthly = load_stock_data(stock_cache_key, str(STOCK_MONTHLY_FILE))
    if MARKET_MONTHLY_FILE.exists():
        market_stat = MARKET_MONTHLY_FILE.stat()
        market_cache_key = f"{MARKET_MONTHLY_FILE.resolve()}::{market_stat.st_mtime_ns}::{market_stat.st_size}"
    else:
        market_cache_key = "missing_market_file"
    market_monthly = load_market_data(market_cache_key, str(MARKET_MONTHLY_FILE))
    st.caption(latest_data_handled_text(economy, orders, stock_monthly, market_monthly))

    if orders.empty:
        st.error("No order intake records were parsed from the yearly order sheets.")
        return

    orders_f, platforms_f = apply_global_filters(orders, platforms)
    render_sidebar_footer()

    if orders_f.empty:
        st.warning("Filters returned no rows. Adjust the sidebar filters.")
        return

    render_header_metrics(orders_f, platforms_f)
    tabs = st.tabs(
        [
            "Overall Economics",
            "Year-by-Year Overview",
            "Quarterly Analytics",
            "Across Years",
            "Platform Analytics",
            "Country Analytics",
            "Delivery and Capacity",
            "Correlations",
            "Latest News",
            "Latest Patents",
            "Information",
        ]
    )

    with tabs[0]:
        render_overall_economics(economy, stock_monthly, market_monthly, economy_fy, quarterly_onshore_cost)
    with tabs[1]:
        render_yearly_overview(orders_f, platforms_f)
    with tabs[2]:
        render_quarterly_analytics(orders_f)
    with tabs[3]:
        render_across_years(orders_f, platforms_f)
    with tabs[4]:
        render_platform_lens(platforms_f)
    with tabs[5]:
        render_country_lens(orders_f, platforms_f)
    with tabs[6]:
        render_delivery_capacity(orders_f)
    with tabs[7]:
        render_correlations(orders_f)
    with tabs[8]:
        render_latest_news()
    with tabs[9]:
        render_latest_patents()
    with tabs[10]:
        render_information_page()


if __name__ == "__main__":
    main()

