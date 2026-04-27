"""Render an editable flowchart that summarises the project's methodology.

The graph is intentionally written as plain Python data structures so that
nodes, edges, colours, fonts and layout offsets can be edited without
touching any visual library calls. Re-running the script regenerates two
images under ``results/figures/`` (English version) and prints a textual
summary on stdout that mirrors what the figure shows.

  * Inputs  : the lists ``PHASES`` and ``EDGES`` defined below.
  * Outputs : results/figures/methodology_graph.png
              results/figures/methodology_graph.svg

The drawing uses pure matplotlib; networkx is used only for topological
ordering. Tweak ``LAYOUT`` to relocate any node manually.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import FIGURES, academic_style

academic_style()

# ---------------------------------------------------------------------------
# Editable methodology specification
# ---------------------------------------------------------------------------
PHASES: list[dict] = [
    {"id": "data",      "label": "1. Data ingestion",
     "subtitle": "Eurostat panel + WB API + UNESCO + CIA Factbook",
     "stage": "Inputs"},
    {"id": "dea",       "label": "2. DEA frontier",
     "subtitle": "CCR · BCC · Super-efficiency · SBM",
     "stage": "Estimation"},
    {"id": "tobit",     "label": "3. Censored regression",
     "subtitle": "Tobit (MLE + OPG) · Simar-Wilson bootstrap",
     "stage": "Estimation"},
    {"id": "extend",    "label": "4. Structural extension",
     "subtitle": "23 country-level indicators",
     "stage": "Inputs"},
    {"id": "ml",        "label": "5. Machine-learning estimators",
     "subtitle": "Elastic Net · RF · GBM · XGBoost · LightGBM · Stack",
     "stage": "Estimation"},
    {"id": "explain",   "label": "6. Explainability",
     "subtitle": "Permutation · SHAP · PDP · per-country SHAP",
     "stage": "Diagnostics"},
    {"id": "compare",   "label": "7. Comparative scoreboard",
     "subtitle": "Single metric across DEA, Tobit and ML",
     "stage": "Diagnostics"},
    {"id": "scenarios", "label": "8. Scenario projection",
     "subtitle": "Intensive · baseline · distributed",
     "stage": "Projection"},
    {"id": "predictor", "label": "9. Out-of-sample predictor",
     "subtitle": "31 non-study countries · Chart.js dropdown",
     "stage": "Projection"},
    {"id": "report",    "label": "10. Bilingual report",
     "subtitle": "GitHub Pages site (EN / ES) + Canva poster",
     "stage": "Outputs"},
]

EDGES: list[tuple[str, str]] = [
    ("data", "dea"),
    ("dea", "tobit"),
    ("data", "extend"),
    ("dea", "ml"),
    ("extend", "ml"),
    ("ml", "explain"),
    ("ml", "compare"),
    ("tobit", "compare"),
    ("dea", "compare"),
    ("ml", "scenarios"),
    ("scenarios", "predictor"),
    ("explain", "report"),
    ("compare", "report"),
    ("predictor", "report"),
]

LAYOUT: dict[str, tuple[float, float]] = {
    "data":      (0.0, 4.0),
    "extend":    (0.0, 2.5),
    "dea":       (1.6, 4.0),
    "tobit":     (1.6, 5.2),
    "ml":        (3.4, 3.2),
    "explain":   (5.2, 4.6),
    "compare":   (5.2, 2.8),
    "scenarios": (7.0, 3.8),
    "predictor": (8.6, 3.8),
    "report":    (10.0, 3.2),
}

STAGE_COLORS = {
    "Inputs":      "#2A9D8F",
    "Estimation":  "#264653",
    "Diagnostics": "#E9C46A",
    "Projection":  "#F4A261",
    "Outputs":     "#E76F51",
}


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def build_graph() -> nx.DiGraph:
    g = nx.DiGraph()
    for p in PHASES:
        g.add_node(p["id"], **p)
    g.add_edges_from(EDGES)
    return g


def draw(g: nx.DiGraph, save_to: Path, fmt: str = "png") -> None:
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(-0.7, 11.0)
    ax.set_ylim(1.6, 6.2)
    ax.axis("off")

    # Edges first so they sit beneath the nodes.
    for u, v in g.edges():
        x1, y1 = LAYOUT[u]
        x2, y2 = LAYOUT[v]
        ax.annotate("",
                    xy=(x2 - 0.55, y2), xytext=(x1 + 0.55, y1),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color="#5d6168", lw=1.2, shrinkA=0, shrinkB=4,
                        connectionstyle="arc3,rad=0.06"))

    # Nodes.
    for node, attrs in g.nodes(data=True):
        x, y = LAYOUT[node]
        color = STAGE_COLORS[attrs["stage"]]
        box = mpatches.FancyBboxPatch(
            (x - 0.65, y - 0.36), 1.3, 0.72,
            boxstyle="round,pad=0.06",
            linewidth=1.2, edgecolor=color,
            facecolor="white",
        )
        ax.add_patch(box)
        ax.text(x, y + 0.10, attrs["label"],
                ha="center", va="center",
                fontsize=10, fontweight="bold", color="#1d1d1f")
        ax.text(x, y - 0.18, attrs["subtitle"],
                ha="center", va="center",
                fontsize=7.6, color="#4f545b", style="italic", wrap=True)

    # Title and stage legend.
    ax.text(5.15, 6.05,
            "European rural-tourism efficiency — methodology pipeline",
            ha="center", va="center",
            fontsize=15, fontweight="bold", color="#264653")
    legend_handles = [
        mpatches.Patch(facecolor="white", edgecolor=col, linewidth=1.6,
                       label=stage)
        for stage, col in STAGE_COLORS.items()
    ]
    ax.legend(handles=legend_handles, loc="lower center",
              ncol=len(STAGE_COLORS), bbox_to_anchor=(0.5, -0.02),
              frameon=False, fontsize=9)

    fig.savefig(save_to, dpi=300, bbox_inches="tight", format=fmt)
    plt.close(fig)


def textual_summary(g: nx.DiGraph) -> str:
    lines = ["Methodology phases:"]
    for p in PHASES:
        lines.append(
            f"  {p['label']:<32} | {p['stage']:<11} | {p['subtitle']}")
    lines.append("\nEdges:")
    for u, v in g.edges():
        lines.append(f"  {u:<10} -> {v}")
    return "\n".join(lines)


def main() -> None:
    g = build_graph()
    out_png = FIGURES / "methodology_graph.png"
    out_svg = FIGURES / "methodology_graph.svg"
    draw(g, out_png, fmt="png")
    draw(g, out_svg, fmt="svg")
    print(textual_summary(g))
    print(f"\nWrote {out_png}\nWrote {out_svg}")


if __name__ == "__main__":
    main()
