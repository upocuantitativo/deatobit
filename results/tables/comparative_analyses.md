| Block | Model | Metric | Value | Notes |
| --- | --- | --- | --- | --- |
| DEA | CCR | Mean efficiency | 0.8357 | 7/33 efficient |
| DEA | BCC | Mean efficiency | 0.9042 | 17/33 efficient |
| DEA | SuperEff | Mean efficiency | 0.8906 | 0/33 efficient |
| DEA | SBM | Mean efficiency | 0.6884 | 7/33 efficient |
| Censored regression | Tobit (right-censored at 1) | Significant regressors (p<0.05) | 1.0000 | MLE |
| Censored regression | Simar-Wilson bootstrap-truncated | Coefs with 95% CI excluding 0 | 3.0000 | B = 1000 |
| Machine learning (baseline) | ElasticNet | R^2 LOO | 0.0527 | RMSE 0.1592 |
| Machine learning (baseline) | RandomForest | R^2 LOO | 0.1055 | RMSE 0.1547 |
| Machine learning (baseline) | GradientBoosting | R^2 LOO | 0.0784 | RMSE 0.1571 |
| Machine learning (baseline) | XGBoost | R^2 LOO | 0.0248 | RMSE 0.1616 |
| Machine learning (baseline) | LightGBM | R^2 LOO | -0.0656 | RMSE 0.1689 |
| Machine learning (extended) | ElasticNet | R^2 LOO | 0.2524 | RMSE 0.1473 |
| Machine learning (extended) | RandomForest | R^2 LOO | 0.2349 | RMSE 0.1490 |
| Machine learning (extended) | GradientBoosting | R^2 LOO | -0.0067 | RMSE 0.1709 |
| Machine learning (extended) | XGBoost | R^2 LOO | 0.3231 | RMSE 0.1402 |
| Machine learning (extended) | LightGBM | R^2 LOO | -0.0784 | RMSE 0.1769 |

**Best estimator** — Machine learning (extended) · `XGBoost` · R^2 LOO = 0.3231