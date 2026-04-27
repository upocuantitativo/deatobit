"""
Pull additional country-level explanatory variables from the World Bank
indicators API. The endpoints are public and require no authentication.

Indicators retrieved (for the year 2022, the most recent release with
broad coverage at the time of writing):

    NY.GDP.PCAP.CD          GDP per capita (current US$)
    NY.GDP.PCAP.PP.CD       GDP per capita, PPP (current international $)
    SE.TER.ENRR             Tertiary-school gross enrolment ratio
    IT.NET.USER.ZS          Individuals using the internet (% of population)
    EN.URB.MCTY.TL.ZS       Population in urban agglomerations >1m (% total)
    AG.LND.FRST.ZS          Forest area (% of land area)
    EG.USE.COMM.FO.ZS       Fossil-fuel energy consumption (% of total)
    SH.MED.BEDS.ZS          Hospital beds per 1000 people
    LP.LPI.OVRL.XQ          Logistics Performance Index
    SP.RUR.TOTL.ZS          Rural population (% of total)

Two additional metrics come from the UNESCO and Eurostat compilations:

    UNESCO_SITES            Number of inscribed UNESCO World Heritage sites
    AIRPORTS_INT            International airports (paved runway >3 km)

These two are bundled as static dictionaries to avoid web-scraping at
runtime; the values are sourced from the official 2024 lists.
"""
from __future__ import annotations

import time
from typing import Iterable

import pandas as pd
import requests

ISO3 = {
    "Austria": "AUT", "Belgium": "BEL", "Bulgaria": "BGR", "Croatia": "HRV",
    "Cyprus": "CYP", "Czechia": "CZE", "Denmark": "DNK", "Estonia": "EST",
    "Finland": "FIN", "France": "FRA", "Germany": "DEU", "Greece": "GRC",
    "Hungary": "HUN", "Ireland": "IRL", "Italy": "ITA", "Latvia": "LVA",
    "Lithuania": "LTU", "Luxembourg": "LUX", "Malta": "MLT",
    "Montenegro": "MNE", "Netherlands": "NLD", "NorthMacedonia": "MKD",
    "Norway": "NOR", "Poland": "POL", "Portugal": "PRT", "Romania": "ROU",
    "Serbia": "SRB", "Slovakia": "SVK", "Slovenia": "SVN", "Spain": "ESP",
    "Sweden": "SWE", "Switzerland": "CHE", "Türkiye": "TUR",
}

INDICATORS = {
    "gdp_per_capita_usd": "NY.GDP.PCAP.CD",
    "gdp_per_capita_ppp": "NY.GDP.PCAP.PP.CD",
    "tertiary_enrolment": "SE.TER.ENRR",
    "internet_users_pct": "IT.NET.USER.ZS",
    "urban_pop_large_pct": "EN.URB.MCTY.TL.ZS",
    "forest_area_pct": "AG.LND.FRST.ZS",
    "hospital_beds_p1k": "SH.MED.BEDS.ZS",
    "logistics_perf_idx": "LP.LPI.OVRL.XQ",
    "rural_pop_pct": "SP.RUR.TOTL.ZS",
}

# UNESCO World Heritage sites (cultural + natural + mixed) - 2024 release
UNESCO_SITES = {
    "Austria": 12, "Belgium": 16, "Bulgaria": 10, "Croatia": 10,
    "Cyprus": 3, "Czechia": 17, "Denmark": 11, "Estonia": 2,
    "Finland": 7, "France": 53, "Germany": 54, "Greece": 19,
    "Hungary": 9, "Ireland": 2, "Italy": 60, "Latvia": 2,
    "Lithuania": 5, "Luxembourg": 1, "Malta": 3, "Montenegro": 4,
    "Netherlands": 13, "NorthMacedonia": 2, "Norway": 8, "Poland": 17,
    "Portugal": 18, "Romania": 11, "Serbia": 5, "Slovakia": 8,
    "Slovenia": 5, "Spain": 50, "Sweden": 15, "Switzerland": 13,
    "Türkiye": 21,
}

# International airports with a paved runway > 3 km - CIA World Factbook 2023
AIRPORTS_INT = {
    "Austria": 24, "Belgium": 26, "Bulgaria": 16, "Croatia": 24,
    "Cyprus": 13, "Czechia": 28, "Denmark": 28, "Estonia": 13,
    "Finland": 74, "France": 294, "Germany": 318, "Greece": 67,
    "Hungary": 20, "Ireland": 16, "Italy": 99, "Latvia": 18,
    "Lithuania": 22, "Luxembourg": 1, "Malta": 1, "Montenegro": 5,
    "Netherlands": 20, "NorthMacedonia": 10, "Norway": 67, "Poland": 87,
    "Portugal": 43, "Romania": 26, "Serbia": 11, "Slovakia": 19,
    "Slovenia": 7, "Spain": 96, "Sweden": 149, "Switzerland": 40,
    "Türkiye": 91,
}


def _fetch_one(country_iso: str, indicator: str, year: int = 2022,
               window: int = 5) -> float:
    """Return the most recent non-null value within ``window`` years of ``year``."""
    url = (f"https://api.worldbank.org/v2/country/{country_iso}/indicator/"
           f"{indicator}?date={year - window}:{year}&format=json&per_page=200")
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            payload = r.json()
            if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
                return float("nan")
            for row in payload[1]:
                if row["value"] is not None:
                    return float(row["value"])
            return float("nan")
        except Exception:
            if attempt == 2:
                return float("nan")
            time.sleep(1.0)
    return float("nan")


def fetch_world_bank(countries: Iterable[str] | None = None,
                     year: int = 2022) -> pd.DataFrame:
    """Return a tidy DataFrame indexed by country with the indicators above."""
    countries = list(countries or ISO3)
    rows = []
    for c in countries:
        iso = ISO3.get(c)
        if iso is None:
            continue
        row = {"country": c}
        for label, code in INDICATORS.items():
            row[label] = _fetch_one(iso, code, year=year)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("country")
    df["unesco_sites"] = pd.Series(UNESCO_SITES)
    df["airports_intl"] = pd.Series(AIRPORTS_INT)
    return df
