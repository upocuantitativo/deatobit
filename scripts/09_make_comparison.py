"""Build a single comparative table that summarises every estimator
(DEA, censored regression and machine learning) and selects the best one.

Output: ``results/tables/comparative_analyses.csv`` and a Markdown
companion ``results/tables/comparative_analyses.md`` that the HTML report
renders verbatim.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import DATA_PROC, TABLES


def main() -> None:
    rows = []

    # DEA estimators - reported as the average score on the panel.
    dea_summary = pd.read_csv(TABLES / "dea_summary.csv", index_col=0)
    for name, row in dea_summary.iterrows():
        rows.append({
            "block": "DEA",
            "model": name,
            "metric_label": "Mean efficiency",
            "metric_value": row["mean"],
            "metric_secondary": f"{int(row['n_efficient'])}/33 efficient",
        })

    # Tobit and bootstrap-truncated.
    tobit_csv = pd.read_csv(TABLES / "tobit_coefficients.csv", index_col=0)
    sig = (tobit_csv["p_value"] < 0.05).sum()
    rows.append({
        "block": "Censored regression",
        "model": "Tobit (right-censored at 1)",
        "metric_label": "Significant regressors (p<0.05)",
        "metric_value": int(sig),
        "metric_secondary": "MLE",
    })

    boot = pd.read_csv(TABLES / "tobit_bootstrap_truncated.csv", index_col=0)
    boot_sig = ((boot["ci_lower"] > 0) | (boot["ci_upper"] < 0)).sum()
    rows.append({
        "block": "Censored regression",
        "model": "Simar-Wilson bootstrap-truncated",
        "metric_label": "Coefs with 95% CI excluding 0",
        "metric_value": int(boot_sig),
        "metric_secondary": "B = 1000",
    })

    # Machine learning - LOO R^2 for both feature sets.
    for tag in ["baseline", "extended"]:
        m = pd.read_csv(TABLES / f"ml_loo_metrics_{tag}.csv", index_col=0)
        for name, row in m.iterrows():
            rows.append({
                "block": f"Machine learning ({tag})",
                "model": name,
                "metric_label": "R^2 LOO",
                "metric_value": row["r2_loo"],
                "metric_secondary": f"RMSE {row['rmse_loo']:.4f}",
            })

    df = pd.DataFrame(rows)
    df["metric_value_num"] = pd.to_numeric(df["metric_value"], errors="coerce")

    # Best ML estimator.
    ml_only = df[df["block"].str.startswith("Machine learning")].copy()
    best = ml_only.loc[ml_only["metric_value_num"].idxmax()]
    summary = {
        "best_block": str(best["block"]),
        "best_model": str(best["model"]),
        "best_metric_label": str(best["metric_label"]),
        "best_metric_value": float(best["metric_value_num"]),
    }

    df.to_csv(TABLES / "comparative_analyses.csv", index=False)
    with open(TABLES / "comparative_analyses_best.json", "w",
              encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    md_lines = [
        "| Block | Model | Metric | Value | Notes |",
        "| --- | --- | --- | --- | --- |",
    ]
    for _, r in df.iterrows():
        val = r["metric_value"]
        if isinstance(val, float):
            val = f"{val:.4f}"
        md_lines.append(
            f"| {r['block']} | {r['model']} | {r['metric_label']} | "
            f"{val} | {r['metric_secondary']} |"
        )
    md_lines.append("")
    md_lines.append(
        f"**Best estimator** — {summary['best_block']} · "
        f"`{summary['best_model']}` · "
        f"{summary['best_metric_label']} = {summary['best_metric_value']:.4f}"
    )
    (TABLES / "comparative_analyses.md").write_text(
        "\n".join(md_lines), encoding="utf-8")

    print(df.to_string())
    print("\nBest estimator:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
