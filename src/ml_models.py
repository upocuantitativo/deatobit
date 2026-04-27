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
from lightgbm import LGBMRegressor
from sklearn.ensemble import (GradientBoostingRegressor, RandomForestRegressor,
                              StackingRegressor)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNetCV, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

RANDOM_STATE = 42


def _pipeline(model) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def build_models() -> Dict[str, Pipeline]:
    return {
        "ElasticNet": _pipeline(ElasticNetCV(
            alphas=np.logspace(-3, 1, 25), l1_ratio=[.1, .3, .5, .7, .9],
            cv=5, random_state=RANDOM_STATE, max_iter=20000)),
        "RandomForest": _pipeline(RandomForestRegressor(
            n_estimators=600, max_depth=None, min_samples_leaf=2,
            random_state=RANDOM_STATE, n_jobs=-1)),
        "GradientBoosting": _pipeline(GradientBoostingRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=3,
            random_state=RANDOM_STATE)),
        "XGBoost": _pipeline(XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=3,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)),
        "LightGBM": _pipeline(LGBMRegressor(
            n_estimators=600, learning_rate=0.05, max_depth=-1,
            num_leaves=15, subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1)),
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
    """Mean absolute SHAP value per feature for tree-based estimators."""
    estimator = model.named_steps["model"]
    Xs = model.named_steps["scaler"].transform(X)
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
            if n in {"RandomForest", "XGBoost", "LightGBM",
                     "GradientBoosting", "ElasticNet"}]
    return StackingRegressor(estimators=base, final_estimator=Ridge(alpha=1.0),
                             cv=5, n_jobs=-1)
