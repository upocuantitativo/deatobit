"""Estimate the Tobit and bootstrap-truncated regressions on DEA scores."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import tobit
from src.utils import DATA_PROC, TABLES, load_tobit_inputs


def main() -> None:
    df = load_tobit_inputs().dropna()
    y = df["ccr_score"]
    X = df[["seasonality", "length_of_stay",
            "protected_hectares", "tourist_pressure"]]

    # Type-I Tobit, right-censored at 1.
    res = tobit.fit_tobit(X, y, lower=-np.inf, upper=1.0)
    res.summary_table.to_csv(TABLES / "tobit_coefficients.csv")
    print("Tobit (right-censored at 1):")
    print(res.summary_table.round(6).to_string())
    print(f"\nsigma = {res.sigma:.6f}, log-lik = {res.loglik:.4f}, "
          f"R2 = {res.r2_pseudo:.4f}, R2_adj = {res.r2_adjusted:.4f}, "
          f"McFadden = {res.r2_mcfadden:.4f}, n = {res.n_obs}")

    # Simar-Wilson (2007) bootstrap-truncated regression.
    print("\nSimar-Wilson bootstrap-truncated regression (B = 1000):")
    boot = tobit.bootstrap_truncated(X, y, n_boot=1000, seed=42)
    boot.to_csv(TABLES / "tobit_bootstrap_truncated.csv")
    print(boot.round(6).to_string())


if __name__ == "__main__":
    main()
