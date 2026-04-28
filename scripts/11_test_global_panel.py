"""Diagnostic experiment: does training the ML estimators on the full
64-country panel (study + non-study) improve out-of-sample accuracy?

The European study panel uses Eurostat sectoral micro-data (beds,
establishments, employees - travellers, overnight stays) that is not
available for the 31 non-study countries. To make the two panels
comparable we re-run DEA on a smaller World-Bank-only specification:

    inputs  = population_total · protected_hectares · airports_intl
    outputs = tourism_arrivals · tourism_receipts_usd

The resulting CCR-proxy scores are used as the new target for the same
five ML estimators. If LOO R² improves materially over the EU-only
baseline (kNN ≈ 0.47), the global panel is worth adopting; if not, the
current strategy is left in place.

This script is intentionally read-only: it neither overwrites artefacts
nor changes the live pipeline. It only reports the comparison on stdout.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import dea, ml_models
from src.utils import DATA_PROC, TABLES

warnings.filterwarnings("ignore")

PROXY_INPUTS  = ["population_total", "protected_hectares", "airports_intl"]
PROXY_OUTPUTS = ["tourism_arrivals", "tourism_receipts_usd"]


def _impute(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
        med = df[c].median()
        df[c] = df[c].fillna(med).astype(float)
        df[c] = np.where(df[c] <= 0, med if med > 0 else 1.0, df[c])
    return df


def _run_pipeline(df: pd.DataFrame, target_col: str, features: list[str],
                  tag: str) -> pd.DataFrame:
    sub = df.dropna(subset=[target_col]).copy()
    X = sub[features]
    y = sub[target_col]
    print(f"  - {tag}: n = {len(sub)}, target_mean = {y.mean():.3f}")
    models = ml_models.build_models(columns=features)
    cv = ml_models.loo_cross_validate(models, X, y)
    return cv.metrics.assign(panel=tag)


def main() -> None:
    eu = pd.read_csv(DATA_PROC / "country_panel.csv", index_col=0)
    nonstudy = pd.read_csv(DATA_PROC / "non_study_country_features.csv",
                           index_col=0)

    # ---------------------------------------------------------------- panel
    full = pd.concat([eu, nonstudy], axis=0).copy()
    full = _impute(full, PROXY_INPUTS + PROXY_OUTPUTS)

    # Compute proxy CCR on the combined 64-country panel.
    Xin = full[PROXY_INPUTS].values.astype(float)
    Yout = full[PROXY_OUTPUTS].values.astype(float)
    full["CCR_proxy"] = dea.ccr(Xin, Yout)
    print("Combined-panel proxy DEA:")
    print(full["CCR_proxy"].describe().round(3).to_string())

    # ---------------------------------------------------------------- features
    structural = [
        "gdp_per_capita_usd", "gdp_per_capita_ppp", "gdp_growth_pct",
        "services_value_added", "agriculture_value_added",
        "tertiary_enrolment", "internet_users_pct", "mobile_subs_p100",
        "forest_area_pct", "agri_land_pct", "protected_area_pct",
        "hospital_beds_p1k", "logistics_perf_idx",
        "tourism_receipts_usd", "tourism_arrivals",
        "rural_pop_pct", "population_total",
        "unesco_sites", "airports_intl",
    ]

    # ---------------------------------------------------------------- runs
    print("\n=== EU only · CCR_proxy target ===")
    eu_idx = eu.index
    metrics_eu = _run_pipeline(
        full.loc[eu_idx], "CCR_proxy", structural, "eu_only_proxy")

    print("\n=== Combined 64-country · CCR_proxy target ===")
    metrics_all = _run_pipeline(
        full, "CCR_proxy", structural, "combined_proxy")

    print("\n=== Combined 64-country · WB-only feature subset (drop strongest" \
          " sectoral) ===")
    structural_lite = [c for c in structural
                       if c not in ("tourism_receipts_usd",
                                    "tourism_arrivals")]
    metrics_lite = _run_pipeline(
        full, "CCR_proxy", structural_lite, "combined_proxy_lite")

    out = pd.concat([metrics_eu, metrics_all, metrics_lite])
    out = out.reset_index().rename(columns={"index": "model"})
    out_path = TABLES / "global_panel_experiment.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved - {out_path}")
    print("\n=== Summary ===")
    pivot = out.pivot(index="model", columns="panel", values="r2_loo").round(4)
    print(pivot.to_string())


if __name__ == "__main__":
    main()
