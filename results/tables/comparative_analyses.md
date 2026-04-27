| Block | Model | Metric | Value | Notes |
| --- | --- | --- | --- | --- |
| DEA | CCR | Mean efficiency | 0.8357 | 7/33 efficient |
| DEA | BCC | Mean efficiency | 0.9042 | 17/33 efficient |
| DEA | SuperEff | Mean efficiency | 0.8906 | 0/33 efficient |
| DEA | SBM | Mean efficiency | 0.6884 | 7/33 efficient |
| Censored regression | Tobit (right-censored at 1) | Significant regressors (p<0.05) | 1.0000 | MLE |
| Censored regression | Simar-Wilson bootstrap-truncated | Coefs with 95% CI excluding 0 | 3.0000 | B = 1000 |
| Machine learning (baseline) | ElasticNet | R^2 LOO | 0.0638 | RMSE 0.1645 |
| Machine learning (baseline) | RandomForest | R^2 LOO | 0.1564 | RMSE 0.1561 |
| Machine learning (baseline) | GradientBoosting | R^2 LOO | 0.1185 | RMSE 0.1596 |
| Machine learning (baseline) | XGBoost | R^2 LOO | 0.0937 | RMSE 0.1618 |
| Machine learning (baseline) | kNN | R^2 LOO | 0.1503 | RMSE 0.1567 |
| Machine learning (extended) | ElasticNet | R^2 LOO | -0.0115 | RMSE 0.1710 |
| Machine learning (extended) | RandomForest | R^2 LOO | 0.2582 | RMSE 0.1464 |
| Machine learning (extended) | GradientBoosting | R^2 LOO | 0.2552 | RMSE 0.1467 |
| Machine learning (extended) | XGBoost | R^2 LOO | 0.2940 | RMSE 0.1428 |
| Machine learning (extended) | kNN | R^2 LOO | 0.4681 | RMSE 0.1240 |

**Best estimator** — Machine learning (extended) · `kNN` · R^2 LOO = 0.4681