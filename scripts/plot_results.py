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

from render_results import ARM, HAIKU_SILENT, HAIKU_V2, REPO, SONNET_SILENT, SONNET_V2, TOTAL, column  # noqa: E402

RUNS = {"Haiku 4.5": (HAIKU_V2, HAIKU_SILENT), "Sonnet 4.6": (SONNET_V2, SONNET_SILENT)}  # no tools, then the tool
METRICS = {"ground_truth_accuracy": f"Verdict + refutation accuracy ({TOTAL} rows)",
           "substantiation": "Substantiation (correct blunder calls naming the certified refutation)",
           "blunder_recall": f"Blunder recall ({ARM['blunder']} blunder rows)",
           "best_recall": f"Best-move recall ({ARM['best']} best rows)",
           "legal_move_accuracy": f"Legal BEST_MOVE ({TOTAL} rows)"}
# palette tokens: one hue for the series the story is about, gray for its context, ink for text
TOOL, GRAY, LINK, GRID, INK = "#2a78d6", "#898781", "#c3c2b7", "#e1e0d9", "#52514e"
GAP = 1.3  # rows of air between metric groups
STEP = len(RUNS) + GAP  # one row per model, then the gap


def main(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for i, (key, label) in enumerate(METRICS.items()):
        ax.text(0, i * STEP - 0.85, label, fontsize=9, color=INK, va="center")
        for j, (none, tool) in enumerate(RUNS.values()):
            y = i * STEP + j
            before, after = 100 * column(none).m[key], 100 * column(tool).m[key]
            lo, hi = sorted((before, after))
            ax.plot([before, after], [y, y], color=LINK, lw=2, solid_capstyle="round", zorder=1)
            ax.plot(before, y, "o", ms=9, color=GRAY, mec="white", mew=2, zorder=2)
            ax.plot(after, y, "o", ms=9, color=TOOL, mec="white", mew=2, zorder=3)
            ax.text(lo - 1.5, y, f"{lo:.1f}", ha="right", va="center", fontsize=8, color=INK)
            ax.text(hi + 1.5, y, f"{hi:.1f}", ha="left", va="center", fontsize=8, color=INK)
    ax.set_yticks([i * STEP + j for i in range(len(METRICS)) for j in range(len(RUNS))], list(RUNS) * len(METRICS),
                  fontsize=8.5, color=GRAY)
    ax.set_ylim(len(METRICS) * STEP - GAP, -1.6)
    ax.set_xlim(0, 112)
    ax.set_xticks(range(0, 101, 25), [f"{x}%" for x in range(0, 101, 25)], fontsize=8, color=GRAY)
    ax.xaxis.grid(True, color=GRID, lw=1)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    ax.plot([], [], "o", ms=9, color=GRAY, label="no tools")
    ax.plot([], [], "o", ms=9, color=TOOL, label="Stockfish tool offered, unmentioned in the prompt")
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1), ncol=2, frameon=False, fontsize=8.5, borderaxespad=0)
    ax.set_title("Grounding: the same prompt with and without the engine\ngolden v2 · instrument v3 · thinking off",
                 loc="left", fontsize=11, color=INK, pad=14)
    fig.tight_layout()
    fig.savefig(output, metadata={"Date": None} if output.suffix == ".svg" else None)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "charts" / "golden-v2-grounding.svg")
