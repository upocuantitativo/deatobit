# European Rural Tourism Efficiency — DEA, Tobit & Machine Learning

Reproducible pipeline that estimates the technical efficiency of the rural-tourism sector in 33 European countries and explains its determinants through censored regression and explainable machine-learning models.

> 🌐 Interactive bilingual report (EN / ES): **https://upocuantitativo.github.io/deatobit/**

---

[English](#english) · [Español](#español)

---

## English

### Overview

| Block | Method | Software |
| --- | --- | --- |
| Frontier estimation | CCR · BCC · Andersen-Petersen super-efficiency · Tone SBM | `scipy.optimize.linprog` |
| Determinants | Type-I Tobit (right-censored at 1) · Simar-Wilson (2007) bootstrap-truncated regression | custom MLE + bootstrap |
| Machine learning | Elastic Net · Random Forest · Gradient Boosting · XGBoost · LightGBM · Multilayer Perceptron · Ridge-stacked ensemble | scikit-learn, xgboost, lightgbm |
| Explainability | Permutation importance · SHAP (TreeExplainer) · partial-dependence plots · per-country SHAP | shap, sklearn.inspection |

### Data

| Block | Variables | Source |
| --- | --- | --- |
| DEA inputs | beds, establishments, employees | Eurostat |
| DEA outputs | travellers, overnight stays | Eurostat |
| Tobit regressors | seasonality, length of stay, protected hectares, tourist pressure | authors' compilation |
| Structural extension (15 indicators) | GDP per capita (USD & PPP), tertiary enrolment, internet usage, urban concentration, forest area, hospital beds, Logistics Performance Index, rural-population share, UNESCO sites, international airports | World Bank API · UNESCO · CIA Factbook |

### Pipeline

```bash
pip install -r requirements.txt

python scripts/01_prepare_data.py        # tidy CSVs
python scripts/02_run_dea.py             # CCR / BCC / SuperEff / SBM
python scripts/03_run_tobit.py           # Tobit + bootstrap-truncated
python scripts/04_fetch_external_data.py # World Bank features
python scripts/05_run_ml_models.py       # ML + permutation + SHAP
python scripts/06_make_figures.py        # figures in EN and ES
```

All artefacts are stored under `results/` (`tables/`, `figures/`, `models/`).

### Project layout

```
deatobit/
├── data/
│   ├── raw/               # original Eurostat panel (xlsx)
│   └── processed/         # tidy CSVs + extended panel
├── docs/                  # GitHub Pages site (bilingual EN / ES)
├── results/
│   ├── tables/            # DEA scores, Tobit, ML metrics, SHAP
│   ├── figures/           # bilingual PNGs (_en / _es)
│   └── models/            # joblib-pickled fitted models
├── scripts/               # reproducible pipeline (01 → 06)
└── src/
    ├── dea.py             # LP-based DEA estimators
    ├── tobit.py           # MLE Tobit + Simar-Wilson bootstrap
    ├── ml_models.py       # learners + LOO + SHAP
    ├── data_fetch.py      # World Bank API client
    └── utils.py           # paths, plotting, loaders
```

### Models, in one line each

* **CCR** — input-oriented LP, constant returns to scale, the "standard" DEA score.
* **BCC** — adds a convexity constraint, isolates pure technical efficiency from scale efficiency.
* **Super-efficiency (Andersen-Petersen)** — drops the evaluated DMU from the reference set so efficient countries can be ranked above 1.
* **SBM (Tone, 2001)** — non-radial measure that penalises input excesses and output shortfalls.
* **Tobit** — right-censored MLE; standard errors from outer-product-of-gradients.
* **Bootstrap-truncated (Simar-Wilson, 2007)** — Algorithm 1, 1 000 bootstrap replications, bias-corrected coefficients.
* **Elastic Net** — linear baseline with combined L1/L2 penalty, hyperparameters chosen by 5-fold CV.
* **Random Forest, Gradient Boosting, XGBoost, LightGBM** — tree ensembles tuned for small samples.
* **Multilayer Perceptron** — 64-32 hidden units with ReLU activations and Adam optimiser.
* **Stacked ensemble** — Ridge meta-learner trained on out-of-fold predictions of the five base estimators.

### Explainability

The pipeline does not stop at predictions: it explains them.

* **Permutation importance** for every model (`results/tables/ml_permutation_importance_*.csv`).
* **SHAP TreeExplainer** for every tree-based learner (`results/tables/ml_shap_summary_*.csv`).
* **Partial-dependence plots** for the four baseline regressors.
* **Country-level SHAP charts** decomposing the prediction for individual countries (Croatia, Italy, Spain, North Macedonia).

All figures are generated twice — `*_en.png` and `*_es.png` — and the GitHub Pages report swaps them through a language toggle.

### Citation

If you use this code, please cite this repository:

```
upocuantitativo (2026). deatobit: DEA, Tobit and machine-learning estimators
for European rural-tourism efficiency. https://github.com/upocuantitativo/deatobit
```

### Licence

MIT — see `LICENSE`.

---

## Español

### Visión general

| Bloque | Método | Software |
| --- | --- | --- |
| Estimación de frontera | CCR · BCC · Supereficiencia de Andersen-Petersen · SBM de Tone | `scipy.optimize.linprog` |
| Determinantes | Tobit Tipo I (censurado por la derecha en 1) · Regresión truncada con bootstrap (Simar-Wilson, 2007) | MLE propio + bootstrap |
| Aprendizaje automático | Elastic Net · Random Forest · Gradient Boosting · XGBoost · LightGBM · Perceptrón multicapa · Ensemble apilado con Ridge | scikit-learn, xgboost, lightgbm |
| Explicabilidad | Importancia por permutación · SHAP (TreeExplainer) · gráficos de dependencia parcial · SHAP a nivel país | shap, sklearn.inspection |

### Datos

| Bloque | Variables | Fuente |
| --- | --- | --- |
| Inputs DEA | plazas, establecimientos, empleados | Eurostat |
| Outputs DEA | viajeros, pernoctaciones | Eurostat |
| Regresores Tobit | estacionalidad, duración de la estancia, hectáreas protegidas, presión turística | Elaboración propia |
| Extensión estructural (15 indicadores) | PIB per cápita (USD y PPA), matrícula terciaria, uso de internet, concentración urbana, superficie forestal, camas hospitalarias, Índice de Desempeño Logístico, % población rural, sitios UNESCO y aeropuertos internacionales | API del Banco Mundial · UNESCO · CIA Factbook |

### Canalización

```bash
pip install -r requirements.txt

python scripts/01_prepare_data.py        # CSV limpios
python scripts/02_run_dea.py             # CCR / BCC / SuperEff / SBM
python scripts/03_run_tobit.py           # Tobit + bootstrap truncado
python scripts/04_fetch_external_data.py # Variables del Banco Mundial
python scripts/05_run_ml_models.py       # IA + permutación + SHAP
python scripts/06_make_figures.py        # Figuras en EN y ES
```

Todos los artefactos quedan en `results/` (`tables/`, `figures/`, `models/`).

### Modelos, en una frase

* **CCR** — programa lineal input-orientado, rendimientos constantes a escala, la métrica DEA estándar.
* **BCC** — añade restricción de convexidad, separa eficiencia técnica pura de la eficiencia de escala.
* **Supereficiencia (Andersen-Petersen)** — elimina la DMU evaluada del conjunto de referencia, permite valores superiores a 1.
* **SBM (Tone, 2001)** — medida no radial que penaliza los excesos en inputs y los déficits en outputs.
* **Tobit** — MLE censurado; errores estándar mediante el estimador OPG.
* **Bootstrap truncado (Simar-Wilson, 2007)** — Algoritmo 1, 1 000 réplicas bootstrap, coeficientes corregidos por sesgo.
* **Elastic Net** — baseline lineal con penalización L1+L2, hiperparámetros por validación cruzada de 5 folds.
* **Random Forest, Gradient Boosting, XGBoost, LightGBM** — ensambles de árboles ajustados para muestras pequeñas.
* **MLP** — 64-32 unidades ocultas con activación ReLU y optimizador Adam.
* **Ensemble apilado** — meta-aprendiz Ridge entrenado sobre las predicciones out-of-fold de los cinco estimadores base.

### Explicabilidad

* **Importancia por permutación** para cada modelo.
* **SHAP TreeExplainer** para cada algoritmo basado en árboles.
* **Gráficos de dependencia parcial** para los cuatro regresores.
* **Gráficos SHAP a nivel país** que descomponen la predicción para Croacia, Italia, España y Macedonia del Norte.

Todas las figuras se generan dos veces (`*_en.png` y `*_es.png`); la web de GitHub Pages las intercambia mediante el selector de idioma.

### Cita

```
upocuantitativo (2026). deatobit: DEA, Tobit y aprendizaje automático
para la eficiencia del turismo rural europeo. https://github.com/upocuantitativo/deatobit
```

### Licencia

MIT — véase `LICENSE`.
