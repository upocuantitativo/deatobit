"""Helpers shared across scripts: paths, plotting style, data loaders."""
from __future__ import annotations

import unicodedata
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
MODELS = RESULTS / "models"

for p in (DATA_PROC, TABLES, FIGURES, MODELS):
    p.mkdir(parents=True, exist_ok=True)


def academic_style() -> None:
    """Set a sober, journal-ready Matplotlib style."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.alpha": 0.5,
        "legend.frameon": False,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def normalise_country(name: str) -> str:
    """Strip mojibake artefacts that appear when Excel is exported as cp1252."""
    if not isinstance(name, str):
        return name
    cleaned = name.replace("Pa�ses", "Países").replace("T�rkiye", "Türkiye")
    cleaned = unicodedata.normalize("NFC", cleaned).strip()
    return cleaned


def load_inputs_outputs() -> pd.DataFrame:
    df = pd.read_excel(DATA_RAW / "Datos_DEA_Tobit.xlsx", sheet_name="DEA",
                       header=1)
    df = df.rename(columns={df.columns[0]: "country"})
    df["country"] = df["country"].apply(normalise_country)
    df = df.set_index("country")
    df.columns = ["beds", "establishments", "employees",
                  "travellers", "overnight_stays"]
    df = df.apply(pd.to_numeric, errors="coerce")
    return df


def load_tobit_inputs() -> pd.DataFrame:
    df = pd.read_excel(DATA_RAW / "Datos_DEA_Tobit.xlsx", sheet_name="Tobit")
    df.columns = ["country", "ccr_score", "protected_hectares",
                  "tourist_pressure", "length_of_stay", "seasonality"]
    df["country"] = df["country"].apply(normalise_country)
    return df.set_index("country")


def palette() -> list[str]:
    return ["#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51",
            "#8AB17D", "#6A4C93", "#1D3557"]


def newfig(figsize=(7, 4.2)):
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax
