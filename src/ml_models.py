"""
Machine-learning models that explain rural-tourism efficiency.

Six estimators are wrapped behind a common interface:

    ElasticNetCV, RandomForestRegressor, GradientBoostingRegressor,
    XGBRegressor, LGBMRegressor and a multilayer perceptron (sklearn).

The models are evaluated through a leave-one-out cross-validation loop,
which is the only sensible scheme given the small sample (n=33). Results
are reported in terms of out-of-sample R-squared, root mean squared
error and mean absolute error.

Permutation importance and SHAP values are computed on the full sample
to surface the variables that drive each model's prediction. A simple
stacked ensemble (linear meta-learner) is also provided.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import (GradientBoostingRegressor, RandomForestRegressor,
                              StackingRegressor)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNetCV, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from xgboost import XGBRegressor

RANDOM_STATE = 42

# Heavily right-skewed variables for which a log-transform helps the linear
# baseline and stabilises the tree splits.
LOG_FEATURES = {
    "gdp_per_capita_usd", "gdp_per_capita_ppp", "tourism_receipts_usd",
    "tourism_arrivals", "tourism_expenditures", "air_passengers",
    "population_total", "protected_hectares", "airports_intl",
    "unesco_sites",
}


def _log_transform(X: np.ndarray, columns: list[str]) -> np.ndarray:
    """Apply log1p to columns that appear in LOG_FEATURES and pass the rest
    through unchanged. ``columns`` is the column order of the input frame."""
    Xv = np.asarray(X, dtype=float).copy()
    for j, col in enumerate(columns):
        if col in LOG_FEATURES:
            Xv[:, j] = np.log1p(np.maximum(Xv[:, j], 0))
    return Xv


def _pipeline(model, columns: list[str] | None = None) -> Pipeline:
    """Median-impute → optional log-transform → standard-scale → estimator."""
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if columns is not None and any(c in LOG_FEATURES for c in columns):
        steps.append(("logger", FunctionTransformer(
            _log_transform, kw_args={"columns": columns}, validate=False)))
    steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def build_models(columns: list[str] | None = None) -> Dict[str, Pipeline]:
    """Five small-sample-friendly learners.

    Hyperparameters are chosen to favour regularisation over capacity:
    shallow trees, leaf-size lower bounds, strong L2 penalties on linear
    and boosted estimators, and a slow learning rate compensated by more
    trees.
    """
    return {
        "ElasticNet": _pipeline(ElasticNetCV(
            alphas=np.logspace(-3, 1, 30),
            l1_ratio=[.05, .2, .4, .6, .8, .95],
            cv=5, random_state=RANDOM_STATE, max_iter=30000), columns),
        "RandomForest": _pipeline(RandomForestRegressor(
            n_estimators=900, max_depth=4, min_samples_leaf=3,
            max_features=0.7, random_state=RANDOM_STATE, n_jobs=-1), columns),
        "GradientBoosting": _pipeline(GradientBoostingRegressor(
            n_estimators=600, learning_rate=0.03, max_depth=2,
            min_samples_leaf=3, subsample=0.85,
            random_state=RANDOM_STATE), columns),
        "XGBoost": _pipeline(XGBRegressor(
            n_estimators=900, learning_rate=0.03, max_depth=3,
            min_child_weight=3, subsample=0.85, colsample_bytree=0.75,
            reg_alpha=0.5, reg_lambda=2.0,
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0), columns),
        "kNN": _pipeline(KNeighborsRegressor(
            n_neighbors=5, weights="distance"), columns),
    }


@dataclass
class CVResult:
    metrics: pd.DataFrame
    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)


def loo_cross_validate(models: Dict[str, Pipeline],
                       X: pd.DataFrame, y: pd.Series) -> CVResult:
    loo = LeaveOneOut()
    rows = []
    preds = {name: np.empty(len(y)) for name in models}
    for name, model in models.items():
        y_pred = np.empty(len(y))
        for tr, te in loo.split(X):
            mod = model.__class__(**model.get_params(deep=False))
            mod.set_params(**model.get_params(deep=False))
            mod = type(model)(model.steps)
            mod.fit(X.iloc[tr], y.iloc[tr])
            y_pred[te] = mod.predict(X.iloc[te])
        preds[name] = y_pred
        rows.append({
            "model": name,
            "r2_loo": r2_score(y, y_pred),
            "rmse_loo": float(np.sqrt(mean_squared_error(y, y_pred))),
            "mae_loo": mean_absolute_error(y, y_pred),
        })
    metrics = pd.DataFrame(rows).set_index("model")
    pred_df = pd.DataFrame(preds, index=X.index).assign(actual=y.values)
    return CVResult(metrics=metrics, predictions=pred_df)


def fit_full(models: Dict[str, Pipeline],
             X: pd.DataFrame, y: pd.Series) -> Dict[str, Pipeline]:
    return {name: model.fit(X, y) for name, model in models.items()}


def permutation_table(models: Dict[str, Pipeline],
                      X: pd.DataFrame, y: pd.Series,
                      n_repeats: int = 50) -> pd.DataFrame:
    out = []
    for name, m in models.items():
        r = permutation_importance(m, X, y, n_repeats=n_repeats,
                                   random_state=RANDOM_STATE, n_jobs=-1)
        for i, col in enumerate(X.columns):
            out.append({"model": name, "feature": col,
                        "importance_mean": r.importances_mean[i],
                        "importance_std": r.importances_std[i]})
    return pd.DataFrame(out)


def shap_summary(model: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Mean absolute SHAP value per feature for tree-based estimators.

    Replays the preprocessing chain (impute → optional log → scale)
    before handing the matrix to the explainer.
    """
    estimator = model.named_steps["model"]
    pre = Pipeline(model.steps[:-1])
    Xs = pre.transform(X)
    try:
        explainer = shap.TreeExplainer(estimator)
        sv = explainer.shap_values(Xs)
    except Exception:
        explainer = shap.Explainer(estimator.predict, Xs)
        sv = explainer(Xs).values
    return pd.DataFrame({
        "feature": X.columns,
        "mean_abs_shap": np.abs(sv).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)


def stacked_ensemble(models: Dict[str, Pipeline]) -> StackingRegressor:
    base = [(n, m) for n, m in models.items()
            if n in {"RandomForest", "XGBoost", "GradientBoosting", "kNN"}]
    return StackingRegressor(estimators=base, final_estimator=Ridge(alpha=2.0),
                             cv=5, n_jobs=-1)
