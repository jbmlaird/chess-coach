"""Chart the latest no-tool baselines from the same numbers the README tables are rendered from.

Usage:
    uv run python scripts/plot_results.py            # -> charts/golden-v2-baselines.svg
    uv run python scripts/plot_results.py out.png    # any matplotlib-supported path
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("svg")
matplotlib.rcParams["svg.hashsalt"] = "chess-coach"  # deterministic element ids: regenerating gives identical bytes
import matplotlib.pyplot as plt  # noqa: E402

from render_results import ARM, TABLES, TOTAL, column  # noqa: E402

REPO = Path(__file__).parent.parent
TABLE = TABLES["v3"]
METRICS = {"blunder_recall": f"Blunder recall\n({ARM['blunder']} blunder rows)",
           "best_recall": f"Best-move recall\n({ARM['best']} best rows)",
           "substantiation": "Substantiation\n(of correct blunder calls)",
           "legal_move_accuracy": f"Legal BEST_MOVE\n({TOTAL} rows)"}
# fixed categorical order, never cycled; eight validated hues, so eight series at most
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def main(output: Path) -> None:
    runs = {title: column(log, TABLE.golden).m for title, log in TABLE.columns.items()}
    fig, ax = plt.subplots(figsize=(8, 4.2))
    width = 0.8 / len(runs)
    for i, (name, m) in enumerate(runs.items()):
        xs = [j + i * width for j in range(len(METRICS))]
        bars = ax.bar(xs, [100 * m[k] for k in METRICS], width - 0.04, label=name, color=COLOURS[i])
        ax.bar_label(bars, fmt="%.1f%%", padding=2, fontsize=8, color="#52514e")
    ax.set_xticks([j + width * (len(runs) - 1) / 2 for j in range(len(METRICS))], METRICS.values(), fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_ylabel("%")
    ax.yaxis.grid(True, color="#e6e6e3")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(TABLE.heading.lstrip("# "), loc="left")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, metadata={"Date": None} if output.suffix == ".svg" else None)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "charts" / "golden-v2-baselines.svg")
