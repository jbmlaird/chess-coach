"""Chart the golden v2 baselines from the same calculate_metrics.py lines the README tables quote.

Usage:
    uv run python scripts/plot_results.py            # -> charts/golden-v2-baselines.svg
    uv run python scripts/plot_results.py out.png    # any matplotlib-supported path
"""

import re
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("svg")
matplotlib.rcParams["svg.hashsalt"] = "chess-coach"  # deterministic element ids: regenerating gives identical bytes
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).parent.parent
METRICS = {"blunder_recall": "Blunder recall\n(200 blunder rows)",
           "best_recall": "Best-move recall\n(50 best rows)",
           "substantiation": "Substantiation\n(of correct blunder calls)",
           "legal_move_accuracy": "Legal BEST_MOVE\n(250 rows)"}
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # fixed categorical order, never cycled


def metrics(log_dir: Path) -> dict[str, float]:
    log, = log_dir.glob("*.eval")  # one committed run per directory
    out = subprocess.run([sys.executable, "scripts/calculate_metrics.py", "--log_file", str(log)],
                         cwd=REPO, capture_output=True, text=True, check=True).stdout
    return {name: float(value) for name, value in re.findall(r"^(\w+): ([0-9.]+)$", out, re.M)}


def main(output: Path) -> None:
    runs = {d.name: metrics(d) for d in sorted((REPO / "logs" / "golden-v2").iterdir())}
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
    ax.set_title("Golden v2 baselines: no tools, thinking off", loc="left")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, metadata={"Date": None} if output.suffix == ".svg" else None)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "charts" / "golden-v2-baselines.svg")
