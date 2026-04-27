"""Clean the raw Excel into tidy CSVs stored under ``data/processed``."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import DATA_PROC, load_inputs_outputs, load_tobit_inputs


def main() -> None:
    io_df = load_inputs_outputs()
    io_df.to_csv(DATA_PROC / "inputs_outputs.csv")
    print(f"Saved inputs/outputs: {io_df.shape} -> "
          f"{DATA_PROC / 'inputs_outputs.csv'}")

    tobit_df = load_tobit_inputs()
    tobit_df.to_csv(DATA_PROC / "tobit_determinants.csv")
    print(f"Saved Tobit determinants: {tobit_df.shape} -> "
          f"{DATA_PROC / 'tobit_determinants.csv'}")


if __name__ == "__main__":
    main()
