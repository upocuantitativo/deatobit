"""Render all figures used in the README and HTML report.

Each figure is produced twice: ``<name>_en.png`` (English) and
``<name>_es.png`` (Spanish). The HTML report swaps the ``src`` of every
``<img>`` based on the language toggle.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import partial_dependence
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import (DATA_PROC, FIGURES, MODELS, TABLES, academic_style,
                       newfig, palette)

warnings.filterwarnings("ignore")
academic_style()
COLORS = palette()


# ---------------------------------------------------------------------------
# Translation tables
# ---------------------------------------------------------------------------
LABELS = {
    "en": {
        "ccr": "CCR efficiency", "bcc": "BCC efficiency",
        "supereff": "Super-efficiency",
        "country": "Country",
        "score": "Score",
        "frequency": "Frequency",
        "predicted": "Predicted CCR efficiency",
        "actual": "Observed CCR efficiency",
        "feature": "Feature",
        "importance": "Mean |SHAP value|",
        "perm_importance": "Permutation importance (R^2 drop)",
        "model": "Model",
        "metric": "Metric",
        "value": "Value",
        "title_ranking": "Country ranking - DEA super-efficiency scores",
        "title_dist": "Distribution of CCR efficiency scores",
        "title_corr": "Pearson correlations among DEA scores",
        "title_models": "Out-of-sample performance (leave-one-out)",
        "title_pred_actual": "Predicted vs observed efficiency (XGBoost, LOO)",
        "title_shap": "Global SHAP feature importance",
        "title_pdp": "Partial-dependence profile",
        "title_country_shap": "Country-level SHAP explanation",
        "feature_seasonality": "Seasonality",
        "feature_length_of_stay": "Length of stay",
        "feature_protected_hectares": "Protected hectares",
        "feature_tourist_pressure": "Tourist pressure",
        "feature_gdp_per_capita_usd": "GDP per capita (USD)",
        "feature_gdp_per_capita_ppp": "GDP per capita (PPP)",
        "feature_tertiary_enrolment": "Tertiary enrolment",
        "feature_internet_users_pct": "Internet users (%)",
        "feature_urban_pop_large_pct": "Large-city population (%)",
        "feature_forest_area_pct": "Forest area (%)",
        "feature_hospital_beds_p1k": "Hospital beds / 1k",
        "feature_logistics_perf_idx": "Logistics Performance Index",
        "feature_rural_pop_pct": "Rural population (%)",
        "feature_unesco_sites": "UNESCO sites",
        "feature_airports_intl": "International airports",
    },
    "es": {
        "ccr": "Eficiencia CCR", "bcc": "Eficiencia BCC",
        "supereff": "Supereficiencia",
        "country": "País",
        "score": "Puntuación",
        "frequency": "Frecuencia",
        "predicted": "Eficiencia CCR predicha",
        "actual": "Eficiencia CCR observada",
        "feature": "Variable",
        "importance": "|SHAP| medio",
        "perm_importance": "Importancia por permutación (caída en R^2)",
        "model": "Modelo",
        "metric": "Métrica",
        "value": "Valor",
        "title_ranking": "Ranking de países - puntuaciones DEA de supereficiencia",
        "title_dist": "Distribución de las puntuaciones CCR",
        "title_corr": "Correlaciones de Pearson entre puntuaciones DEA",
        "title_models": "Desempeño fuera de muestra (validación leave-one-out)",
        "title_pred_actual": "Predicho vs. observado (XGBoost, LOO)",
        "title_shap": "Importancia global de las variables (SHAP)",
        "title_pdp": "Perfil de dependencia parcial",
        "title_country_shap": "Explicación SHAP a nivel país",
        "feature_seasonality": "Estacionalidad",
        "feature_length_of_stay": "Duración de la estancia",
        "feature_protected_hectares": "Hectáreas protegidas",
        "feature_tourist_pressure": "Presión turística",
        "feature_gdp_per_capita_usd": "PIB per cápita (USD)",
        "feature_gdp_per_capita_ppp": "PIB per cápita (PPA)",
        "feature_tertiary_enrolment": "Matrícula terciaria",
        "feature_internet_users_pct": "Usuarios de internet (%)",
        "feature_urban_pop_large_pct": "Población en grandes ciudades (%)",
        "feature_forest_area_pct": "Superficie forestal (%)",
        "feature_hospital_beds_p1k": "Camas hospitalarias / 1k",
        "feature_logistics_perf_idx": "Índice de Desempeño Logístico",
        "feature_rural_pop_pct": "Población rural (%)",
        "feature_unesco_sites": "Sitios UNESCO",
        "feature_airports_intl": "Aeropuertos internacionales",
    },
}


def label(key: str, lang: str) -> str:
    return LABELS[lang].get(key, key)


def feature_label(name: str, lang: str) -> str:
    return label(f"feature_{name}", lang)


def save(fig, name: str) -> None:
    fig.savefig(FIGURES / name, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def figure_ranking(scores: pd.DataFrame, lang: str) -> None:
    s = scores.sort_values("SuperEff", ascending=True)
    fig, ax = newfig(figsize=(7, 8))
    bars = ax.barh(s.index, s["SuperEff"], color=COLORS[1], edgecolor="white")
    ax.axvline(1.0, color=COLORS[4], linestyle="--", linewidth=1)
    ax.set_xlabel(label("supereff", lang))
    ax.set_ylabel(label("country", lang))
    ax.set_title(label("title_ranking", lang))
    for bar, v in zip(bars, s["SuperEff"]):
        ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{v:.2f}", va="center", fontsize=8)
    save(fig, f"01_ranking_{lang}.png")


def figure_distribution(scores: pd.DataFrame, lang: str) -> None:
    fig, ax = newfig(figsize=(7, 4.2))
    ax.hist(scores["CCR"], bins=12, color=COLORS[0], edgecolor="white")
    ax.axvline(scores["CCR"].mean(), color=COLORS[4], linestyle="--",
               label=f"mean = {scores['CCR'].mean():.3f}")
    ax.set_xlabel(label("ccr", lang))
    ax.set_ylabel(label("frequency", lang))
    ax.set_title(label("title_dist", lang))
    ax.legend()
    save(fig, f"02_distribution_{lang}.png")


def figure_correlations(scores: pd.DataFrame, lang: str) -> None:
    cols = ["CCR", "BCC", "SuperEff", "SBM"]
    corr = scores[cols].corr().values
    fig, ax = newfig(figsize=(5.5, 4.5))
    im = ax.imshow(corr, cmap="RdYlBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=30)
    ax.set_yticks(range(len(cols)), cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                    color="white" if abs(corr[i, j]) > 0.5 else "black",
                    fontsize=10)
    ax.set_title(label("title_corr", lang))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save(fig, f"03_corr_dea_{lang}.png")


def figure_model_comparison(metrics_b: pd.DataFrame, metrics_e: pd.DataFrame,
                            lang: str) -> None:
    fig, ax = newfig(figsize=(8, 4.2))
    x = np.arange(len(metrics_b))
    width = 0.35
    ax.bar(x - width / 2, metrics_b["r2_loo"], width,
           color=COLORS[0], label="baseline (4 vars)")
    ax.bar(x + width / 2, metrics_e["r2_loo"], width,
           color=COLORS[2], label="extended (15 vars)")
    ax.set_xticks(x, metrics_b.index, rotation=20)
    ax.set_ylabel("R^2 (LOO)")
    ax.set_title(label("title_models", lang))
    ax.axhline(0, color="grey", linewidth=0.6)
    ax.legend()
    save(fig, f"04_model_comparison_{lang}.png")


def figure_pred_vs_actual(pred_df: pd.DataFrame, lang: str) -> None:
    fig, ax = newfig(figsize=(5.5, 5.2))
    ax.scatter(pred_df["actual"], pred_df["XGBoost"], color=COLORS[0],
               edgecolor="white", s=70)
    for c in pred_df.index:
        ax.text(pred_df.loc[c, "actual"] + 0.005,
                pred_df.loc[c, "XGBoost"], c, fontsize=7)
    lo, hi = 0.35, 1.05
    ax.plot([lo, hi], [lo, hi], color=COLORS[4], linestyle="--", linewidth=1)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(label("actual", lang))
    ax.set_ylabel(label("predicted", lang))
    ax.set_title(label("title_pred_actual", lang))
    save(fig, f"05_pred_vs_actual_{lang}.png")


def figure_shap_global(shap_df: pd.DataFrame, lang: str, tag: str) -> None:
    df = shap_df[shap_df["model"] == "XGBoost"].copy()
    df["display"] = df["feature"].apply(lambda f: feature_label(f, lang))
    df = df.sort_values("mean_abs_shap")
    fig, ax = newfig(figsize=(7, 4 + 0.18 * len(df)))
    ax.barh(df["display"], df["mean_abs_shap"], color=COLORS[1],
            edgecolor="white")
    ax.set_xlabel(label("importance", lang))
    ax.set_title(f"{label('title_shap', lang)} ({tag})")
    save(fig, f"06_shap_global_{tag}_{lang}.png")


def figure_permutation(perm_df: pd.DataFrame, lang: str, tag: str) -> None:
    df = perm_df[perm_df["model"] == "RandomForest"].copy()
    df["display"] = df["feature"].apply(lambda f: feature_label(f, lang))
    df = df.sort_values("importance_mean")
    fig, ax = newfig(figsize=(7, 4 + 0.18 * len(df)))
    ax.barh(df["display"], df["importance_mean"],
            xerr=df["importance_std"], color=COLORS[3], edgecolor="white",
            error_kw={"ecolor": "grey", "elinewidth": 0.8})
    ax.set_xlabel(label("perm_importance", lang))
    ax.set_title(label("title_shap", lang).replace("SHAP", "permutation")
                 .replace("(SHAP)", "(permutación)"))
    save(fig, f"07_permutation_{tag}_{lang}.png")


def figure_pdp(model, X: pd.DataFrame, feature: str, lang: str,
               tag: str) -> None:
    pdp = partial_dependence(model, X, [feature], grid_resolution=50)
    grid = pdp["grid_values"][0]
    avg = pdp["average"][0]
    fig, ax = newfig(figsize=(6, 4))
    ax.plot(grid, avg, color=COLORS[0], linewidth=2)
    ax.scatter(X[feature], np.full(len(X), avg.min() - 0.005),
               marker="|", color=COLORS[4], alpha=0.7)
    ax.set_xlabel(feature_label(feature, lang))
    ax.set_ylabel(label("predicted", lang))
    ax.set_title(f"{label('title_pdp', lang)}: {feature_label(feature, lang)}")
    save(fig, f"08_pdp_{feature}_{tag}_{lang}.png")


def figure_comparison_bar(comp_df: pd.DataFrame, lang: str) -> None:
    """Horizontal bar chart of every estimator on a unified scoreboard."""
    df = comp_df.copy()
    df["metric_value_num"] = pd.to_numeric(df["metric_value"], errors="coerce")
    ml = df[df["block"].str.startswith("Machine learning")].copy()
    ml = ml.sort_values("metric_value_num")
    label_x = "R² LOO" if lang == "en" else "R² LOO"
    fig, ax = newfig(figsize=(8, 6))
    colors = []
    for _, r in ml.iterrows():
        if r["block"].endswith("(extended)"):
            colors.append(COLORS[1])
        else:
            colors.append(COLORS[3])
    bars = ax.barh(ml["model"] + " · " + ml["block"].str.replace(
                       "Machine learning ", ""),
                   ml["metric_value_num"], color=colors, edgecolor="white")
    for bar, v in zip(bars, ml["metric_value_num"]):
        ax.text(v + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=8)
    ax.axvline(0, color="grey", linewidth=0.6)
    ax.set_xlabel(label_x)
    title = ("Comparative scoreboard of ML estimators"
             if lang == "en"
             else "Comparativa de los estimadores de aprendizaje automático")
    ax.set_title(title)
    save(fig, f"10_comparison_{lang}.png")


def figure_scenario_panel(scenario_summary: dict, lang: str) -> None:
    """Bar chart of mean predicted efficiency per scenario."""
    tags = ["intensive", "baseline", "distributed"]
    means = [scenario_summary["scenarios"][t]["mean"] for t in tags]
    labels = {
        "en": ["Intensive demand", "Baseline", "Sustainable demand"],
        "es": ["Demanda intensiva", "Línea base", "Demanda sostenible"],
    }[lang]
    colors = [COLORS[4], COLORS[6], COLORS[1]]
    fig, ax = newfig(figsize=(6, 4))
    ax.bar(labels, means, color=colors, edgecolor="white")
    for i, v in enumerate(means):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean predicted CCR" if lang == "en"
                  else "CCR media predicha")
    ax.set_title("Three-scenario projection" if lang == "en"
                 else "Proyección a tres escenarios")
    save(fig, f"11_scenarios_{lang}.png")


def figure_country_shap(model, X: pd.DataFrame, country: str, lang: str,
                        tag: str) -> None:
    estimator = model.named_steps["model"]
    pre = Pipeline(model.steps[:-1])
    Xs = pre.transform(X)
    Xs_df = pd.DataFrame(Xs, columns=X.columns, index=X.index)
    explainer = shap.TreeExplainer(estimator)
    sv = explainer.shap_values(Xs_df.loc[[country]])
    contrib = pd.Series(sv[0], index=X.columns).sort_values()
    contrib.index = [feature_label(f, lang) for f in contrib.index]
    fig, ax = newfig(figsize=(7, 4 + 0.18 * len(contrib)))
    colors = [COLORS[4] if v > 0 else COLORS[1] for v in contrib.values]
    ax.barh(contrib.index, contrib.values, color=colors, edgecolor="white")
    ax.axvline(0, color="grey", linewidth=0.6)
    ax.set_xlabel("SHAP")
    ax.set_title(f"{label('title_country_shap', lang)} - {country}")
    save(fig, f"09_country_shap_{country}_{tag}_{lang}.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    scores = pd.read_csv(DATA_PROC / "dea_scores.csv", index_col=0)
    metrics_b = pd.read_csv(TABLES / "ml_loo_metrics_baseline.csv",
                            index_col=0)
    metrics_e = pd.read_csv(TABLES / "ml_loo_metrics_extended.csv",
                            index_col=0)
    pred_b = pd.read_csv(TABLES / "ml_predictions_baseline.csv", index_col=0)
    shap_b = pd.read_csv(TABLES / "ml_shap_summary_baseline.csv")
    shap_e = pd.read_csv(TABLES / "ml_shap_summary_extended.csv")
    perm_b = pd.read_csv(TABLES / "ml_permutation_importance_baseline.csv")
    perm_e = pd.read_csv(TABLES / "ml_permutation_importance_extended.csv")

    panel = pd.read_csv(DATA_PROC / "country_panel.csv", index_col=0)
    baseline_feats = ["seasonality", "length_of_stay",
                      "protected_hectares", "tourist_pressure"]
    panel_b = panel.dropna(subset=baseline_feats + ["CCR"])
    Xb = panel_b[baseline_feats]
    yb = panel_b["CCR"]

    fitted_b = joblib.load(MODELS / "ml_models_baseline.pkl")
    fitted_e = joblib.load(MODELS / "ml_models_extended.pkl")

    comp = pd.read_csv(TABLES / "comparative_analyses.csv")
    try:
        with open(TABLES / "scenario_summary.json", encoding="utf-8") as fh:
            scen = json.load(fh)
    except FileNotFoundError:
        scen = None

    for lang in ("en", "es"):
        figure_ranking(scores, lang)
        figure_distribution(scores, lang)
        figure_correlations(scores, lang)
        figure_model_comparison(metrics_b, metrics_e, lang)
        figure_pred_vs_actual(pred_b, lang)
        figure_shap_global(shap_b, lang, "baseline")
        figure_shap_global(shap_e, lang, "extended")
        figure_permutation(perm_b, lang, "baseline")
        figure_permutation(perm_e, lang, "extended")
        for feat in baseline_feats:
            figure_pdp(fitted_b["XGBoost"], Xb, feat, lang, "baseline")
        for country in ("Spain", "Italy", "Croatia", "NorthMacedonia"):
            if country in Xb.index:
                figure_country_shap(fitted_b["XGBoost"], Xb, country,
                                    lang, "baseline")
        figure_comparison_bar(comp, lang)
        if scen is not None:
            figure_scenario_panel(scen, lang)

    print("Saved figures (en + es) under", FIGURES)
    # Persist label dictionary so the HTML can reuse it.
    with open(FIGURES / "labels.json", "w", encoding="utf-8") as fh:
        json.dump(LABELS, fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
