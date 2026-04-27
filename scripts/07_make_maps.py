"""Generate choropleth maps for the report.

Four maps are produced (and saved as PNG and standalone HTML for embedding):

  * ``map_density``      observed CCR efficiency density across the 33 study countries
  * ``map_optimistic``   predicted CCR under an optimistic scenario
  * ``map_baseline``     predicted CCR under the current (baseline) scenario
  * ``map_pessimistic``  predicted CCR under a pessimistic scenario

The scenario perturbations only touch the four Tobit regressors so that the
counter-factual is interpretable:

  +--------------+---------------------------------------------------------+
  | optimistic   | length of stay  +20%, seasonality -25%, pressure -15%  |
  | baseline     | values as observed                                     |
  | pessimistic  | seasonality +25%, pressure +20%, length of stay -15%   |
  +--------------+---------------------------------------------------------+

The best ML estimator (Gradient Boosting on the extended panel) is used to
project the perturbed feature matrix.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_fetch import ISO3
from src.utils import DATA_PROC, FIGURES, MODELS, TABLES

warnings.filterwarnings("ignore")

PALETTE = "Viridis"

SCENARIOS = {
    # Scenario labels are neutral so they describe the perturbation, not a
    # value judgement. With this study panel, intensive demand patterns
    # correlate with higher observed CCR (Malta is a strong driver), so
    # the projector's mean prediction rises with concentrated demand.
    "intensive":   {"length_of_stay": 0.85, "seasonality": 1.25,
                    "tourist_pressure": 1.20, "protected_hectares": 0.90},
    "baseline":    {"length_of_stay": 1.00, "seasonality": 1.00,
                    "tourist_pressure": 1.00, "protected_hectares": 1.00},
    "distributed": {"length_of_stay": 1.20, "seasonality": 0.75,
                    "tourist_pressure": 0.85, "protected_hectares": 1.10},
}


def _iso_lookup(country: str) -> str | None:
    """Return ISO-3 code for either study or non-study countries."""
    return ISO3.get(country)


def _save(fig, name: str) -> None:
    fig.write_image(FIGURES / f"{name}.png", width=900, height=620, scale=2)
    fig.write_html(FIGURES / f"{name}.html", include_plotlyjs="cdn",
                   full_html=True)


def _choropleth(df: pd.DataFrame, value: str, title: str,
                cmin: float, cmax: float):
    fig = px.choropleth(
        df, locations="iso3", color=value, hover_name="country",
        scope="world", color_continuous_scale=PALETTE,
        range_color=[cmin, cmax], title=title,
    )
    fig.update_geos(
        showcountries=True, countrycolor="#777", projection_type="mercator",
        lataxis_range=[33, 72], lonaxis_range=[-25, 45],
        showland=True, landcolor="#f8f5ec",
        showocean=True, oceancolor="#eef2f4",
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=60, b=0),
        font=dict(family="EB Garamond, Times New Roman, serif", size=14),
        title=dict(x=0.02, xanchor="left", font=dict(size=18)),
        coloraxis_colorbar=dict(title=value, thickness=12, len=0.8),
        paper_bgcolor="#fdfdfb",
    )
    return fig


def main() -> None:
    panel = pd.read_csv(DATA_PROC / "country_panel.csv", index_col=0)
    panel["iso3"] = panel.index.map(_iso_lookup)

    # 1) Observed density.
    df_obs = panel.dropna(subset=["CCR"]).copy()
    df_obs["country"] = df_obs.index
    fig = _choropleth(df_obs, "CCR",
                      "Observed CCR efficiency (33 European countries)",
                      cmin=0.4, cmax=1.0)
    _save(fig, "map_density")

    # 2) Scenario projections.
    # The extended XGBoost is overall best (R^2 LOO = 0.32) but its 19
    # auxiliary regressors dominate the four perturbed variables, so
    # scenarios collapse to noise. We instead use the Random Forest fitted
    # on the four sectoral regressors - the only variables that the
    # perturbation actually moves - so the counter-factual gradient is
    # interpretable.
    fitted_ext = joblib.load(MODELS / "ml_models_extended.pkl")
    fitted_base = joblib.load(MODELS / "ml_models_baseline.pkl")
    metrics_ext = pd.read_csv(TABLES / "ml_loo_metrics_extended.csv",
                              index_col=0)
    best_model = metrics_ext["r2_loo"].idxmax()
    print(f"Best overall ML model: {best_model} (extended) "
          f"(R^2 LOO = {metrics_ext.loc[best_model, 'r2_loo']:.4f})")
    print("Scenario projections use the baseline RandomForest "
          "for interpretability of the four-variable perturbation.")
    model = fitted_base["RandomForest"]
    # Pipeline now starts with an imputer; the imputer keeps the column
    # names from the original DataFrame.
    feature_cols = list(model.named_steps["imputer"].feature_names_in_)

    sub = panel.dropna(subset=feature_cols + ["CCR"]).copy()
    base_X = sub[feature_cols].copy()

    summary = {"best_model": best_model, "scenarios": {}}
    for tag, perturbation in SCENARIOS.items():
        Xp = base_X.copy()
        for col, factor in perturbation.items():
            if col in Xp.columns:
                Xp[col] = Xp[col] * factor
        pred = pd.Series(model.predict(Xp), index=sub.index, name=tag)
        pred = pred.clip(0.0, 1.05)

        df_map = pd.DataFrame({
            "country": pred.index, "iso3": [_iso_lookup(c) for c in pred.index],
            "predicted_efficiency": pred.values,
        })
        title = {
            "baseline":    "Predicted CCR - baseline scenario (status quo)",
            "intensive":   "Predicted CCR - intensive demand scenario",
            "distributed": "Predicted CCR - sustainable demand scenario",
        }[tag]
        fig = _choropleth(df_map, "predicted_efficiency", title,
                          cmin=0.4, cmax=1.05)
        _save(fig, f"map_{tag}")
        summary["scenarios"][tag] = {
            "mean": float(pred.mean()),
            "median": float(pred.median()),
            "min": float(pred.min()),
            "max": float(pred.max()),
            "perturbation": perturbation,
        }

        df_map.to_csv(TABLES / f"scenario_predictions_{tag}.csv", index=False)

    with open(TABLES / "scenario_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("Maps written to", FIGURES)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
