"""Estimate the four DEA models and persist the country-level scores."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import dea
from src.utils import DATA_PROC, TABLES, load_inputs_outputs


def main() -> None:
    df = load_inputs_outputs()
    X = df[["beds", "establishments", "employees"]].values.astype(float)
    Y = df[["travellers", "overnight_stays"]].values.astype(float)

    scores = pd.DataFrame(index=df.index)
    scores["CCR"] = dea.ccr(X, Y)
    scores["BCC"] = dea.bcc(X, Y)
    scores["SuperEff"] = dea.super_efficiency(X, Y)
    scores["SBM"] = dea.sbm(X, Y)
    scores["scale_eff"] = scores["CCR"] / scores["BCC"]
    scores = scores.sort_values("SuperEff", ascending=False)

    out_csv = DATA_PROC / "dea_scores.csv"
    scores.to_csv(out_csv)
    print(scores.round(4).to_string())

    summary = pd.DataFrame({m: dea.descriptive(scores[m].values)
                            for m in ["CCR", "BCC", "SuperEff", "SBM"]}).T
    summary.to_csv(TABLES / "dea_summary.csv")
    print("\nDescriptive summary:")
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
