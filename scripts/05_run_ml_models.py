"""Train, evaluate and explain the machine-learning models that predict
country-level rural-tourism efficiency.

The script is run after ``02_run_dea.py``, ``03_run_tobit.py`` and
``04_fetch_external_data.py``. Two feature sets are evaluated:

  * baseline - the four determinants used in the original Tobit (Simar &
    Wilson, 2007 second-stage variables).
  * extended - the four baseline regressors plus eleven structural
    indicators harvested from the World Bank, UNESCO and CIA Factbook.

Results stored under ``results/``:

    tables/ml_loo_metrics.csv          out-of-sample R^2, RMSE, MAE
    tables/ml_permutation_importance.csv
    tables/ml_shap_summary.csv
    tables/ml_predictions.csv
    models/full_models.pkl             fitted pipelines
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import ml_models
from src.utils import DATA_PROC, MODELS, TABLES

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


BASELINE_FEATURES = ["seasonality", "length_of_stay",
                     "protected_hectares", "tourist_pressure"]

EXTENDED_FEATURES = BASELINE_FEATURES + [
    # Macro context.
    "gdp_per_capita_usd", "gdp_per_capita_ppp", "gdp_growth_pct",
    "services_value_added", "agriculture_value_added",
    # Human capital and connectivity.
    "tertiary_enrolment", "internet_users_pct", "mobile_subs_p100",
    # Territory and environment.
    "forest_area_pct", "agri_land_pct", "protected_area_pct",
    # Infrastructure.
    "hospital_beds_p1k", "logistics_perf_idx",
    # Tourism direct indicators.
    "tourism_receipts_usd", "tourism_arrivals",
    # Demographics.
    "rural_pop_pct", "population_total",
    # Heritage and connectivity.
    "unesco_sites", "airports_intl",
]


def _load_panel() -> pd.DataFrame:
    return pd.read_csv(DATA_PROC / "country_panel.csv", index_col=0)


def _evaluate(df: pd.DataFrame, features: list[str], tag: str) -> None:
    # We only require CCR to be present; missing regressors are imputed by
    # the in-pipeline median imputer, which keeps the full sample.
    sub = df.dropna(subset=["CCR"]).copy()
    X = sub[features]
    y = sub["CCR"]
    print(f"\n=== {tag.upper()} feature set ({len(features)} variables, "
          f"n = {len(sub)}) ===")

    models = ml_models.build_models(columns=features)
    cv = ml_models.loo_cross_validate(models, X, y)
    cv.metrics.to_csv(TABLES / f"ml_loo_metrics_{tag}.csv")
    cv.predictions.to_csv(TABLES / f"ml_predictions_{tag}.csv")
    print("Leave-one-out metrics:")
    print(cv.metrics.round(4).to_string())

    fitted = ml_models.fit_full(ml_models.build_models(columns=features),
                                X, y)

    perm = ml_models.permutation_table(fitted, X, y, n_repeats=50)
    perm["feature_set"] = tag
    perm.to_csv(TABLES / f"ml_permutation_importance_{tag}.csv", index=False)

    shap_rows = []
    for name in ["RandomForest", "XGBoost", "GradientBoosting"]:
        try:
            s = ml_models.shap_summary(fitted[name], X)
            s.insert(0, "model", name)
            shap_rows.append(s)
        except Exception as exc:
            print(f"  SHAP failed for {name}: {exc}")
    if shap_rows:
        shap_df = pd.concat(shap_rows, ignore_index=True)
        shap_df["feature_set"] = tag
        shap_df.to_csv(TABLES / f"ml_shap_summary_{tag}.csv", index=False)

    joblib.dump(fitted, MODELS / f"ml_models_{tag}.pkl")

    # Stacked ensemble with a Ridge meta-learner.
    stack = ml_models.stacked_ensemble(
        ml_models.build_models(columns=features))
    stack.fit(X, y)
    joblib.dump(stack, MODELS / f"ml_stack_{tag}.pkl")
    print(f"Stacked ensemble training R^2: {stack.score(X, y):.4f}")


def main() -> None:
    df = _load_panel()
    _evaluate(df, BASELINE_FEATURES, "baseline")
    _evaluate(df, EXTENDED_FEATURES, "extended")


if __name__ == "__main__":
    main()
