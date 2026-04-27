"""Pull additional country-level explanatory variables from the World Bank
indicators API and merge them with the original Tobit regressors."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_fetch import fetch_world_bank
from src.utils import DATA_PROC


def main() -> None:
    print("Querying World Bank indicators for 33 European countries...")
    wb = fetch_world_bank(year=2022)
    wb.to_csv(DATA_PROC / "external_country_features.csv")
    print(f"Saved external features: {wb.shape} -> "
          f"{DATA_PROC / 'external_country_features.csv'}")

    base = pd.read_csv(DATA_PROC / "tobit_determinants.csv", index_col=0)
    dea_scores = pd.read_csv(DATA_PROC / "dea_scores.csv", index_col=0)
    merged = (base.join(wb, how="left")
                  .join(dea_scores[["CCR", "BCC", "SuperEff", "SBM"]],
                        how="left"))
    merged.to_csv(DATA_PROC / "country_panel.csv")
    print(f"Saved merged panel: {merged.shape} -> "
          f"{DATA_PROC / 'country_panel.csv'}")
    print("\nMissing-value count per column:")
    print(merged.isna().sum().to_string())


if __name__ == "__main__":
    main()
