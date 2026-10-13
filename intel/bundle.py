"""One-shot computation of every analytic the intelligence tabs need (so each rerun is cheap)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from . import economics, outlook, quality, structure
from .common import (
    as_of_date,
    load_segment_results,
    prepare_orders,
    prepare_platforms,
)


@dataclass
class Intel:
    """Precomputed analytics for the full (unfiltered) dataset."""

    as_of: pd.Timestamp
    orders: pd.DataFrame
    platforms: pd.DataFrame
    economy_fy: pd.DataFrame
    results: dict[str, Any] | None
    seg: pd.DataFrame
    fy: pd.DataFrame
    to_date: pd.DataFrame
    ann_fq: pd.DataFrame
    ann_fy: pd.DataFrame
    ytd: pd.DataFrame
    ltm: dict[str, Any]
    projection: dict[str, Any]
    qe_overall: dict[str, float]
    qe_by_year: pd.DataFrame
    quarter_share: pd.DataFrame
    month_share: pd.Series
    concentration: pd.DataFrame
    mix: pd.DataFrame
    series: pd.DataFrame
    rating_mix: pd.DataFrame
    leading: pd.DataFrame
    service: pd.DataFrame
    lead_time: pd.DataFrame
    radar: pd.DataFrame
    markets: pd.DataFrame
    econ: pd.DataFrame
    anomalies: pd.DataFrame
    completeness: pd.DataFrame
    extras: dict[str, Any] = field(default_factory=dict)


def build_intel(
    economy_fy: pd.DataFrame,
    orders: pd.DataFrame,
    platforms: pd.DataFrame,
    stock: pd.DataFrame | None = None,
    market: pd.DataFrame | None = None,
    results: dict[str, Any] | None = None,
) -> Intel:
    o = prepare_orders(orders)
    p = prepare_platforms(platforms)
    as_of = as_of_date(o)
    if results is None:
        results = load_segment_results()
    seg = outlook.segment_table(results)
    fy = outlook.fiscal_year_table(seg)
    n_latest = int(seg.iloc[-1]["quarter"]) if len(seg) else 0
    return Intel(
        as_of=as_of,
        orders=o,
        platforms=p,
        economy_fy=economy_fy,
        results=results,
        seg=seg,
        fy=fy,
        to_date=outlook.to_date_table(seg, n_latest) if n_latest else pd.DataFrame(),
        ann_fq=outlook.announced_by_fiscal_quarter(o, seg),
        ann_fy=outlook.announced_by_fiscal_year(o, fy),
        ytd=outlook.ytd_table(o, as_of),
        ltm=outlook.ltm_summary(o, seg, as_of),
        projection=outlook.project_fiscal_year(results, seg),
        qe_overall=outlook.quarter_end_overall(o),
        qe_by_year=outlook.quarter_end_share_by_year(o),
        quarter_share=outlook.quarter_share_table(o),
        month_share=outlook.month_in_quarter_share(o),
        concentration=structure.concentration_by_year(o, as_of),
        mix=structure.mix_by_year(p),
        series=structure.series_mix(p),
        rating_mix=structure.rating_bucket_mix(p),
        leading=structure.leading_platforms(p),
        service=structure.service_by_year(o),
        lead_time=structure.lead_time_by_year(o),
        radar=structure.account_radar(o, as_of),
        markets=structure.market_status(o, as_of),
        econ=economics.economics_table(economy_fy, fy, market),
        anomalies=quality.anomaly_summary(o, p, results, economy_fy, stock, market),
        completeness=quality.completeness_by_year(o),
    )
