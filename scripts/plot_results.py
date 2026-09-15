"""Chart the grounding effect from the numbers the README tables are rendered from: each model on the same
prompt with and without the Stockfish tool (the silent arm, whose prompt is byte-identical to the baseline's).

Usage:
    uv run python scripts/plot_results.py            # -> charts/golden-v2-grounding.svg
    uv run python scripts/plot_results.py out.png    # any matplotlib-supported path
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("svg")
matplotlib.rcParams["svg.hashsalt"] = "chess-coach"  # deterministic element ids: regenerating gives identical bytes
import matplotlib.pyplot as plt  # noqa: E402

from render_results import ARM, REPO, TABLES, TOTAL, column  # noqa: E402

SILENT = "Tool offered, unmentioned (`silent`)"
RUNS = {"Haiku 4.5": (TABLES["v3"].columns["Haiku 4.5"], TABLES["haiku-grounded"].columns[SILENT]),
        "Sonnet 4.6": (TABLES["v3"].columns["Sonnet 4.6"], TABLES["sonnet-grounded"].columns[SILENT])}
METRICS = {"ground_truth_accuracy": f"Verdict + refutation accuracy ({TOTAL} rows)",
           "substantiation": "Substantiation (correct blunder calls naming the certified refutation)",
           "blunder_recall": f"Blunder recall ({ARM['blunder']} blunder rows)",
           "best_recall": f"Best-move recall ({ARM['best']} best rows)",
           "legal_move_accuracy": f"Legal BEST_MOVE ({TOTAL} rows)"}
# palette tokens: one hue for the series the story is about, gray for its context, ink for text
TOOL, CONTEXT, LINK, GRID, INK, MUTED = "#2a78d6", "#898781", "#c3c2b7", "#e1e0d9", "#52514e", "#898781"


def main(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6))
    rows, y = {}, 0.0
    for key, label in METRICS.items():
        ax.text(0, y - 0.85, label, fontsize=9, color=INK, va="center")
        for model, (none, tool) in RUNS.items():
            before, after = 100 * column(none).m[key], 100 * column(tool).m[key]
            ax.plot([before, after], [y, y], color=LINK, lw=2, solid_capstyle="round", zorder=1)
            ax.plot(before, y, "o", ms=9, color=CONTEXT, mec="white", mew=2, zorder=2)
            ax.plot(after, y, "o", ms=9, color=TOOL, mec="white", mew=2, zorder=3)
            ax.text(min(before, after) - 1.5, y, f"{min(before, after):.1f}", ha="right", va="center", fontsize=8, color=INK)
            ax.text(max(before, after) + 1.5, y, f"{max(before, after):.1f}", ha="left", va="center", fontsize=8, color=INK)
            rows[y] = model
            y += 1
        y += 1.3
    ax.set_yticks(list(rows), list(rows.values()), fontsize=8.5, color=MUTED)
    ax.set_ylim(y - 1.3, -1.6)
    ax.set_xlim(0, 112)
    ax.set_xticks(range(0, 101, 25), [f"{x}%" for x in range(0, 101, 25)], fontsize=8, color=MUTED)
    ax.xaxis.grid(True, color=GRID, lw=1)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    ax.plot([], [], "o", ms=9, color=CONTEXT, label="no tools")
    ax.plot([], [], "o", ms=9, color=TOOL, label="Stockfish tool offered, unmentioned in the prompt")
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1), ncol=2, frameon=False, fontsize=8.5, borderaxespad=0)
    ax.set_title("Grounding: the same prompt with and without the engine\ngolden v2 · instrument v3 · thinking off",
                 loc="left", fontsize=11, color=INK, pad=14)
    fig.tight_layout()
    fig.savefig(output, metadata={"Date": None} if output.suffix == ".svg" else None)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "charts" / "golden-v2-grounding.svg")
