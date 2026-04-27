"""Render an editable flowchart that summarises the project's methodology.

Design notes:

  * The layout is a strict left-to-right DAG with one column per stage
    (Inputs → Estimation → Diagnostics → Projection → Outputs). All
    arrows therefore travel forward, which removes back-edges and makes
    crossings rare.
  * Box widths and heights are fixed and chosen so that wrapped text
    always fits inside them. Re-flowing the labels is enough to change
    the figure layout — no manual tweaking of positions needed.
  * Arrow endpoints are computed as the intersection between the
    straight line connecting two box centres and the rectangular border
    of the destination box. The arrow head therefore lands on the box
    edge instead of crossing into it.
  * If two boxes lie in the same column, the arrow is bent into a
    Bezier curve that routes around any intermediate column. This is
    triggered by edges that cross more than one column.

Inputs:  ``PHASES``, ``EDGES``, ``COLUMNS`` and ``STAGE_COLORS`` below.
Outputs: ``results/figures/methodology_graph.png`` and ``.svg``.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as MplPath

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
    {"id": "extend",    "label": "2. Structural extension",
     "subtitle": "23 country-level indicators (WB, UNESCO, Factbook)",
     "stage": "Inputs"},
    {"id": "dea",       "label": "3. DEA frontier",
     "subtitle": "CCR · BCC · Super-efficiency · SBM",
     "stage": "Estimation"},
    {"id": "tobit",     "label": "4. Censored regression",
     "subtitle": "Tobit MLE + OPG · Simar-Wilson bootstrap",
     "stage": "Estimation"},
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
    ("data", "extend"),
    ("dea", "tobit"),
    ("dea", "ml"),
    ("extend", "ml"),
    ("dea", "compare"),
    ("tobit", "compare"),
    ("ml", "compare"),
    ("ml", "explain"),
    ("ml", "scenarios"),
    ("scenarios", "predictor"),
    ("explain", "report"),
    ("compare", "report"),
    ("predictor", "report"),
]

# Column index per stage (left → right).
COLUMNS = {
    "Inputs":      0,
    "Estimation":  1,
    "Diagnostics": 2,
    "Projection":  3,
    "Outputs":     4,
}

STAGE_COLORS = {
    "Inputs":      "#2A9D8F",
    "Estimation":  "#264653",
    "Diagnostics": "#E9C46A",
    "Projection":  "#F4A261",
    "Outputs":     "#E76F51",
}

# Drawing constants.
BOX_W = 2.7    # width of each phase box (data units)
BOX_H = 1.2    # height of each phase box
COL_GAP = 1.6  # horizontal gap between columns
ROW_GAP = 0.55 # vertical gap between rows in the same column
WRAP_TITLE = 26
WRAP_SUB = 32


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
def compute_layout() -> dict[str, tuple[float, float]]:
    """Place each phase at the centre of its (column, row-in-column) cell."""
    by_col: dict[int, list[str]] = {}
    for p in PHASES:
        col = COLUMNS[p["stage"]]
        by_col.setdefault(col, []).append(p["id"])

    layout: dict[str, tuple[float, float]] = {}
    for col, ids in by_col.items():
        n = len(ids)
        # Vertically centre the column on y = 0.
        total = n * BOX_H + (n - 1) * ROW_GAP
        top = total / 2 - BOX_H / 2
        for k, _id in enumerate(ids):
            x = col * (BOX_W + COL_GAP)
            y = top - k * (BOX_H + ROW_GAP)
            layout[_id] = (x, y)
    return layout


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _intersect_rect(cx: float, cy: float, half_w: float, half_h: float,
                    px: float, py: float) -> tuple[float, float]:
    """Return the point on the rectangle border (centre cx,cy, half-extents
    half_w, half_h) that lies on the line going from the centre toward
    the external point (px, py)."""
    dx, dy = px - cx, py - cy
    if dx == 0 and dy == 0:
        return cx, cy
    if dx == 0:
        return cx, cy + (half_h if dy > 0 else -half_h)
    if dy == 0:
        return cx + (half_w if dx > 0 else -half_w), cy
    t_x = half_w / abs(dx)
    t_y = half_h / abs(dy)
    t = min(t_x, t_y)
    return cx + t * dx, cy + t * dy


def _draw_arrow(ax, src_xy, dst_xy, color="#5d6168", curved: bool = False):
    src_x, src_y = src_xy
    dst_x, dst_y = dst_xy
    # Source point is on the right edge of the source box, destination on
    # the left edge of the destination box (when source is to the left).
    src_pt = _intersect_rect(src_x, src_y, BOX_W / 2, BOX_H / 2, dst_x, dst_y)
    dst_pt = _intersect_rect(dst_x, dst_y, BOX_W / 2, BOX_H / 2, src_x, src_y)
    if curved:
        rad = 0.18 if dst_y > src_y else -0.18
        connectionstyle = f"arc3,rad={rad}"
    else:
        connectionstyle = "arc3,rad=0.0"
    arrow = FancyArrowPatch(
        src_pt, dst_pt, arrowstyle="-|>", mutation_scale=14,
        linewidth=1.2, color=color,
        connectionstyle=connectionstyle, zorder=1,
    )
    ax.add_patch(arrow)


def _wrap(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(text, width=width)) or text


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def draw(layout: dict[str, tuple[float, float]], save_to: Path,
         fmt: str = "png") -> None:
    cols = max(COLUMNS.values()) + 1
    rows_max = max(
        sum(1 for p in PHASES if p["stage"] == s)
        for s in COLUMNS
    )
    fig_w = cols * (BOX_W + COL_GAP) + 1.2
    fig_h = rows_max * (BOX_H + ROW_GAP) + 2.4
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    xs = [x for x, _ in layout.values()]
    ys = [y for _, y in layout.values()]
    ax.set_xlim(min(xs) - BOX_W, max(xs) + BOX_W)
    ax.set_ylim(min(ys) - BOX_H, max(ys) + BOX_H * 1.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # ------------------------------------------------------------------
    # Edges first (so the boxes paint on top of the arrow shafts).
    # ------------------------------------------------------------------
    phases_by_id = {p["id"]: p for p in PHASES}
    for u, v in EDGES:
        col_diff = COLUMNS[phases_by_id[v]["stage"]] - COLUMNS[phases_by_id[u]["stage"]]
        curved = col_diff >= 2 or col_diff == 0
        _draw_arrow(ax, layout[u], layout[v], curved=curved)

    # ------------------------------------------------------------------
    # Boxes with wrapped text.
    # ------------------------------------------------------------------
    for p in PHASES:
        x, y = layout[p["id"]]
        color = STAGE_COLORS[p["stage"]]
        box = FancyBboxPatch(
            (x - BOX_W / 2, y - BOX_H / 2), BOX_W, BOX_H,
            boxstyle="round,pad=0.03,rounding_size=0.15",
            linewidth=1.6, edgecolor=color, facecolor="white",
            zorder=3,
        )
        ax.add_patch(box)
        ax.text(x, y + BOX_H * 0.18,
                _wrap(p["label"], WRAP_TITLE),
                ha="center", va="center",
                fontsize=10.0, fontweight="bold", color="#1d1d1f",
                zorder=4)
        ax.text(x, y - BOX_H * 0.20,
                _wrap(p["subtitle"], WRAP_SUB),
                ha="center", va="center",
                fontsize=8.0, color="#4f545b", style="italic",
                zorder=4)

    # ------------------------------------------------------------------
    # Title and stage legend.
    # ------------------------------------------------------------------
    title_y = max(ys) + BOX_H * 1.2
    ax.text((min(xs) + max(xs)) / 2, title_y,
            "European rural-tourism efficiency — methodology pipeline",
            ha="center", va="center",
            fontsize=15, fontweight="bold", color="#264653")

    legend_handles = [
        mpatches.Patch(facecolor="white", edgecolor=col, linewidth=1.6,
                       label=stage)
        for stage, col in STAGE_COLORS.items()
    ]
    ax.legend(handles=legend_handles, loc="lower center",
              ncol=len(STAGE_COLORS), bbox_to_anchor=(0.5, -0.04),
              frameon=False, fontsize=9)

    fig.savefig(save_to, dpi=300, bbox_inches="tight", format=fmt)
    plt.close(fig)


def textual_summary() -> str:
    lines = ["Methodology phases:"]
    for p in PHASES:
        lines.append(f"  {p['label']:<32} | {p['stage']:<11} | {p['subtitle']}")
    lines.append("\nEdges:")
    for u, v in EDGES:
        lines.append(f"  {u:<10} -> {v}")
    return "\n".join(lines)


def main() -> None:
    layout = compute_layout()
    draw(layout, FIGURES / "methodology_graph.png", fmt="png")
    draw(layout, FIGURES / "methodology_graph.svg", fmt="svg")
    print(textual_summary())
    print(f"\nWrote {FIGURES / 'methodology_graph.png'}")
    print(f"Wrote {FIGURES / 'methodology_graph.svg'}")


if __name__ == "__main__":
    main()
