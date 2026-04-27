"""Project the trained ML estimator to ~30 countries that lie outside the
study panel and persist a JSON file consumed by the in-page dropdown.

For each non-study country we derive proxy values for the four Tobit
regressors (length of stay, seasonality, protected hectares, tourist
pressure) using simple cross-sectional rules calibrated on the European
panel; the remaining 22 indicators come straight from the World Bank
fetch. This deliberately keeps the methodology transparent: the country
profile is fully described by data, never by hand-tuned constants.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import DATA_PROC, MODELS, TABLES


SCENARIOS = {
    "intensive":   {"length_of_stay": 0.85, "seasonality": 1.25,
                    "tourist_pressure": 1.20, "protected_hectares": 0.90},
    "baseline":    {"length_of_stay": 1.00, "seasonality": 1.00,
                    "tourist_pressure": 1.00, "protected_hectares": 1.00},
    "distributed": {"length_of_stay": 1.20, "seasonality": 0.75,
                    "tourist_pressure": 0.85, "protected_hectares": 1.10},
}


def _impute_tobit_proxies(panel_study: pd.DataFrame,
                          panel_ext: pd.DataFrame) -> pd.DataFrame:
    """Fill the four Tobit regressors for non-study countries using regression
    against macro indicators that are present in both panels."""
    common = ["gdp_per_capita_usd", "internet_users_pct", "rural_pop_pct",
              "tourism_arrivals", "tourism_receipts_usd", "airports_intl",
              "unesco_sites", "population_total"]
    for target in ["length_of_stay", "seasonality",
                   "protected_hectares", "tourist_pressure"]:
        train = panel_study.dropna(subset=common + [target])
        Xtr = train[common].values
        ytr = train[target].values
        beta, *_ = np.linalg.lstsq(
            np.column_stack([np.ones(len(Xtr)), Xtr]), ytr, rcond=None)
        Xext = panel_ext[common].values
        yhat = beta[0] + Xext @ beta[1:]
        if target == "length_of_stay":
            yhat = np.clip(yhat, 1.5, 6.0)
        elif target == "seasonality":
            yhat = np.clip(yhat, 400, 2200)
        elif target == "protected_hectares":
            yhat = np.clip(yhat, 1e3, 5e7)
        elif target == "tourist_pressure":
            yhat = np.clip(yhat, 0.5, 1000)
        panel_ext[target] = yhat
    return panel_ext


def main() -> None:
    panel_study = pd.read_csv(DATA_PROC / "country_panel.csv", index_col=0)
    panel_ext = pd.read_csv(DATA_PROC / "non_study_country_features.csv",
                            index_col=0)

    panel_ext = _impute_tobit_proxies(panel_study.copy(), panel_ext.copy())
    panel_ext.to_csv(DATA_PROC / "non_study_country_features.csv")

    fitted = joblib.load(MODELS / "ml_models_extended.pkl")
    metrics = pd.read_csv(TABLES / "ml_loo_metrics_extended.csv", index_col=0)
    best_model = metrics["r2_loo"].idxmax()
    model = fitted[best_model]
    feature_cols = list(model.named_steps["scaler"].feature_names_in_)

    sub = panel_ext.dropna(subset=feature_cols).copy()
    base_X = sub[feature_cols].copy()

    out: dict[str, dict] = {"best_model": best_model, "countries": {}}
    for tag, perturbation in SCENARIOS.items():
        Xp = base_X.copy()
        for col, factor in perturbation.items():
            if col in Xp.columns:
                Xp[col] = Xp[col] * factor
        pred = pd.Series(model.predict(Xp), index=sub.index)
        pred = pred.clip(0.0, 1.05)
        for c in pred.index:
            out["countries"].setdefault(c, {})[tag] = float(pred[c])

    summary = pd.DataFrame.from_dict(out["countries"], orient="index")
    summary.to_csv(TABLES / "non_study_predictions.csv")
    with open(TABLES / "non_study_predictions.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
