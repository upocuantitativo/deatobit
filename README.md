# European Rural Tourism Efficiency — Explainable Machine Learning

Reproducible pipeline that predicts and explains the technical efficiency of the rural-tourism sector in 33 European countries, projects three forward-looking scenarios on a choropleth map, and lets the visitor query the model on countries that lie outside the study panel.

> 🌐 **Live bilingual report (EN / ES):** https://upocuantitativo.github.io/deatobit/

---

[English](#english) · [Español](#español)

---

## English

### What this repository does

* Estimates rural-tourism efficiency on the European study panel as the upstream target.
* Trains six machine-learning estimators plus a Ridge-stacked ensemble on a 23-indicator structural panel built from the World Bank API, the UNESCO World Heritage list and the CIA World Factbook.
* Explains the predictions with permutation importance, SHAP values, partial-dependence and per-country decomposition charts.
* Builds a comparative scoreboard that ranks every estimator on a single metric.
* Projects three counter-factual scenarios (intensive demand, status quo, distributed demand) and renders them on Plotly choropleth maps.
* Lets the visitor select a country outside the study panel and a scenario in the GitHub Pages report; the chart and predicted score update in the browser.

### Pipeline

```bash
pip install -r requirements.txt

python scripts/01_prepare_data.py        # tidy CSVs
python scripts/02_run_dea.py             # CCR / BCC / SuperEff / SBM (target variable)
python scripts/03_run_tobit.py           # censored regression (cross-check)
python scripts/04_fetch_external_data.py # 23 World Bank indicators + non-study panel
python scripts/05_run_ml_models.py       # ML + LOO + permutation + SHAP
python scripts/06_make_figures.py        # bilingual EN / ES figures
python scripts/07_make_maps.py           # density + 3-scenario choropleth
python scripts/08_predict_non_study.py   # JSON for the in-page predictor
python scripts/09_make_comparison.py     # comparative scoreboard
```

### Project layout

```
deatobit/
├── data/
│   ├── raw/              # original Eurostat panel
│   └── processed/        # tidy CSVs + extended panel + non-study panel
├── docs/                 # GitHub Pages site (bilingual EN / ES)
│   ├── index.html        # report with language toggle and predictor
│   ├── style.css         # academic stylesheet
│   ├── data/             # JSON consumed by the predictor and scoreboard
│   └── figures/          # bilingual PNGs + choropleth HTML
├── results/
│   ├── tables/           # ML metrics, comparative scoreboard, scenarios
│   ├── figures/          # generated bilingual figures
│   └── models/           # joblib-pickled fitted models
├── scripts/              # reproducible pipeline (01 → 09)
└── src/
    ├── ml_models.py      # ML learners + LOO + SHAP
    ├── data_fetch.py     # World Bank API client (study + non-study)
    ├── dea.py            # LP-based DEA estimators (target generator)
    ├── tobit.py          # MLE Tobit + Simar-Wilson bootstrap (cross-check)
    └── utils.py          # paths, plotting, loaders
```

### Estimators in one line each

* **Elastic Net** — linear baseline with combined L1/L2 penalty.
* **Random Forest, Gradient Boosting, XGBoost, LightGBM** — tree ensembles tuned for small samples.
* **Multilayer Perceptron** — 64-32 hidden units with ReLU activations.
* **Stacked ensemble** — Ridge meta-learner over the five base estimators (out-of-fold predictions).

### Scenarios

| Scenario | Length of stay | Seasonality | Tourist pressure | Protected hectares |
| --- | --- | --- | --- | --- |
| Intensive demand | −15 % | +25 % | +20 % | −10 % |
| Baseline (status quo) | 0 % | 0 % | 0 % | 0 % |
| Distributed demand | +20 % | −25 % | −15 % | +10 % |

The labels describe the perturbation, not a value judgement. With this study panel, intensive-demand patterns correlate with higher reported CCR scores (Malta is a strong driver), so the projector's mean prediction rises with concentration.

### Country predictor (out-of-sample)

The site ships with pre-computed predictions for 31 countries that lie outside the European study panel — North America, Latin America, Asia, the Middle East, Oceania, Africa and non-EU Europe. The dropdown in `#predictor` swaps the country and scenario; the page renders the predicted score, ranking and a Chart.js bar chart.

### Licence

MIT — see `LICENSE`.

---

## Español

### Qué hace este repositorio

* Estima la eficiencia del turismo rural en el panel europeo como variable objetivo.
* Entrena seis algoritmos de aprendizaje automático y un ensemble apilado con Ridge sobre un panel estructural de 23 indicadores construido a partir del API del Banco Mundial, la lista del Patrimonio Mundial UNESCO y el CIA World Factbook.
* Explica las predicciones con importancia por permutación, valores SHAP, dependencia parcial y descomposición a nivel país.
* Construye un cuadro comparativo que ordena todos los estimadores con una única métrica.
* Proyecta tres escenarios contrafactuales (demanda intensiva, status quo, demanda distribuida) sobre mapas coropléticos generados con Plotly.
* Permite seleccionar un país fuera del panel europeo y un escenario en la web; el gráfico y la puntuación se actualizan en el navegador.

### Canalización

```bash
pip install -r requirements.txt

python scripts/01_prepare_data.py        # CSV limpios
python scripts/02_run_dea.py             # CCR / BCC / SuperEff / SBM (variable objetivo)
python scripts/03_run_tobit.py           # regresión censurada (control)
python scripts/04_fetch_external_data.py # 23 indicadores del BM + panel externo
python scripts/05_run_ml_models.py       # IA + LOO + permutación + SHAP
python scripts/06_make_figures.py        # figuras bilingües EN / ES
python scripts/07_make_maps.py           # densidad + 3 mapas de escenario
python scripts/08_predict_non_study.py   # JSON para el predictor interactivo
python scripts/09_make_comparison.py     # cuadro comparativo
```

### Estimadores

* **Elastic Net** · baseline lineal con penalización L1+L2.
* **Random Forest, Gradient Boosting, XGBoost, LightGBM** · ensambles de árboles ajustados para muestras pequeñas.
* **MLP** · 64-32 unidades ocultas con activación ReLU.
* **Ensemble apilado** · meta-aprendiz Ridge entrenado sobre las predicciones out-of-fold de los cinco estimadores base.

### Escenarios

| Escenario | Duración estancia | Estacionalidad | Presión turística | Hectáreas protegidas |
| --- | --- | --- | --- | --- |
| Demanda intensiva | −15 % | +25 % | +20 % | −10 % |
| Línea base (status quo) | 0 % | 0 % | 0 % | 0 % |
| Demanda distribuida | +20 % | −25 % | −15 % | +10 % |

Los nombres describen la perturbación, no un juicio de valor. En este panel europeo los patrones de demanda intensiva correlacionan con mayor CCR observada (Malta es un caso atípico relevante), por lo que la predicción media sube con la concentración.

### Predictor de país (fuera de muestra)

La web incluye predicciones pre-calculadas para 31 países fuera del panel europeo — Norteamérica, Latinoamérica, Asia, Oriente Medio, Oceanía, África y la Europa no UE. El selector del bloque `#predictor` cambia el país y el escenario; la página actualiza la puntuación predicha, el ranking y un gráfico de barras Chart.js.

### Licencia

MIT — véase `LICENSE`.
