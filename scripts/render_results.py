"""Render the README's results tables from the committed logs, so no cell is ever typed by hand.

Usage:
    uv run python scripts/render_results.py          # every table, in README order
    uv run python scripts/render_results.py v3       # one table
tests/test_render_results.py asserts README.md contains each rendered table verbatim.
"""

import statistics
import sys
from dataclasses import dataclass
from functools import cache, cached_property
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))
import calculate_metrics  # noqa: E402
import grade_logs  # noqa: E402
from calculate_metrics import ARM_SIZES as ARM, TOTAL  # noqa: E402
from engine import NOISE_FLOOR_PP  # noqa: E402

LOGS = REPO / "logs"


class Column:
    """One run: the metrics script's numbers, and the grader's only if a row asks."""

    def __init__(self, log: str):
        self.log = LOGS / log
        # the golden set a run used is the first directory of its log path (logs/golden-vN/...)
        self.golden = REPO / "golden" / log.split("/")[0].removeprefix("golden-") / "golden_engine.csv"

    @cached_property
    def m(self) -> dict:
        return calculate_metrics.metrics(self.log)[1]

    @cached_property
    def g(self) -> dict:
        return grade_logs.grade(self.log, self.golden)


column = cache(Column)  # one Column per log: a run shared by several tables (and the chart) is read once


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def count(n: int) -> tuple:
    return (f"{n}/{TOTAL} ({100 * n / TOTAL:.1f}%)" if n else "0", n)


def damage(values: list[float], share: bool) -> tuple:
    if not values:
        return "n=0", None
    mean = statistics.fmean(values)
    good = sum(v <= NOISE_FLOOR_PP for v in values)
    return (f"{mean:.1f}pp ({100 * good / len(values):.0f}%, n={len(values)})" if share
            else f"{mean:.1f}pp (n={len(values)})"), mean


def rate(key: str):
    """Cell for a 0-1 metric: the percentage, keyed on the raw value."""
    return lambda c: (pct(c.m[key]), c.m[key])


def unplayable(c: Column) -> tuple:
    return f"{c.m['refutation_unplayable']}/{ARM['blunder']}", c.m["refutation_unplayable"]


def legal(c: Column) -> tuple:
    rate = c.m["legal_move_accuracy"]
    answered = TOTAL - c.m["legal_move_blunder_parse_error"] - c.m["legal_move_best_parse_error"]
    text = pct(rate) + (f" ({100 * rate * TOTAL / answered:.1f}% of answered)" if answered < TOTAL else "")
    return text, rate


# (row label, cell function -> (text, sort key), which direction is better for bolding)
ROWS = (
    ("Verdict + refutation accuracy (overall)", rate("ground_truth_accuracy"), "high"),
    ("- blunder arm / best arm",
     lambda c: (f"{pct(c.m['ground_truth_blunder_accuracy'])} / {pct(c.m['ground_truth_best_accuracy'])}", None), None),
    ("Blunder class - recall (caught real blunders)", rate("blunder_recall"), "high"),
    ("Blunder class - precision (calls that were right)", rate("blunder_precision"), "high"),
    ("Best class - recall (endorsed real best moves)", rate("best_recall"), "high"),
    ("Best class - precision (endorsements right)", rate("best_precision"), "high"),
    ('Best class - mean damage of suggested "improvements" (excludes correct endorsements)',
     lambda c: damage(c.g["damages"][("best", False)], share=False), "low"),
    ("Substantiation (correct blunder calls backing the certified refutation)", rate("substantiation"), "high"),
    ("Unplayable refutations (illegal, invalid or ambiguous)", unplayable, "low"),
    ("Legal `BEST_MOVE` suggestions", legal, "high"),
    ("Invalid `BEST_MOVE` answers (of which a lowercase piece letter)",
     lambda c: (f"{c.m['invalid_best_moves']}/{TOTAL} ({c.m['lowercase_piece']})", c.m["invalid_best_moves"]), "low"),
    ("Suggested improvement damage (blunder arm, excludes rows where the model repeated the blunder)",
     lambda c: damage(c.g["damages"][("blunder", False)], share=True), "low"),
    ("Unfinished samples (stop reason not `stop`)", lambda c: count(sum(c.m["unfinished"].values())), "low"),
    ("Empty outputs (whole budget spent thinking)", lambda c: count(sum(c.m["empty_output"].values())), "low"),
    ("Format failures (non-empty parse errors)",
     lambda c: count(c.m["ground_truth_blunder_abstained"] + c.m["ground_truth_blunder_missing_refutation"]
                     + c.m["ground_truth_best_parse_error"] - sum(c.m["empty_output"].values())), "low"),
    ("Measured cost (full run)", lambda c: (f"~${c.m['cost']:.2f}", c.m["cost"]), "low"),
)
INVALID_ROWS = (  # no "better" direction: nothing is bolded, so the sort keys are ignored
    ("Legal `BEST_MOVE` suggestions", rate("legal_move_accuracy"), None),
    ("Invalid `BEST_MOVE` answers", lambda c: (f"{c.m['invalid_best_moves']}/{TOTAL}", None), None),
    ("- of which a lowercase piece letter", lambda c: (str(c.m["lowercase_piece"]), None), None),
    ("Unplayable refutations", unplayable, None),
)


@dataclass(frozen=True)
class Table:
    heading: str
    columns: dict  # column title -> log path under logs/
    rows: tuple = ROWS


HAIKU_SAN = "golden-v1/haiku-4-5/2026-08-18T22-27-12-00-00_Positions_3ZCkX5hNDqYXoMaqmhWacC.eval"
SONNET_SAN = "golden-v1/sonnet-4-6/2026-08-19T10-03-45-00-00_Positions_BjUGVfCp5AQazJMgFZcWGN.eval"
HAIKU_UCI = "golden-v1/haiku-4-5/2026-08-20T16-46-50-00-00_Positions_ZUC6qqipkz3TMfHuWhSjdq.eval"
SONNET_UCI = "golden-v1/sonnet-4-6/2026-08-20T16-47-36-00-00_Positions_BREFiymF2Vhthv64Yw7XFD.eval"
HAIKU_THINKING = "golden-v1/haiku-4-5-thinking/2026-08-21T11-21-11-00-00_Positions_Bbu78gFi6mq5BeU3A5VXfL.eval"
SONNET_THINKING = "golden-v1/sonnet-4-6-thinking/2026-08-21T11-27-27-00-00_Positions_eim5RxFdeHyg87yjkwiFps.eval"
OPUS = "golden-v1/opus-5/2026-08-21T10-18-31-00-00_Positions_8XbGRba8CoJjq3Ywj48RJd.eval"
OPUS_NO_THINKING = "golden-v1/opus-5-no-thinking/2026-08-21T14-49-55-00-00_Positions_h9P5PVHV529gGusNguaTvZ.eval"
HAIKU_V2 = "golden-v2/haiku-4-5/2026-09-08T12-50-15-00-00_Positions_HFFMrFK37edcScmKRHx8Ss.eval"
SONNET_V2 = "golden-v2/sonnet-4-6/2026-09-08T12-51-04-00-00_Positions_BT4gKBq5Zio6DpXCrg8En4.eval"

TABLES = {
    "baseline": Table("### Instrument v0 · golden v1 · no tools · SAN prompt (2026-08-18/19)",
                      {"Haiku 4.5": HAIKU_SAN, "Sonnet 4.6": SONNET_SAN}),
    "v2": Table("### Instrument v2 · golden v1 · no tools (2026-08-20/21)",
                {"Haiku 4.5": HAIKU_UCI, "Haiku 4.5 (thinking, 42k `max_tokens`)": HAIKU_THINKING,
                 "Sonnet 4.6": SONNET_UCI, "Sonnet 4.6 (thinking, 42k `max_tokens`)": SONNET_THINKING,
                 "Opus 5 (thinking disabled)": OPUS_NO_THINKING, "Opus 5 (adaptive thinking, 32k `max_tokens`)": OPUS}),
    "v3": Table("### Instrument v3 · golden v2 · no tools · thinking off (2026-09-08)",
                {"Haiku 4.5": HAIKU_V2, "Sonnet 4.6": SONNET_V2}),
    "invalid": Table("### Invalid best moves: Haiku's lowercase piece letters",
                     {"Haiku 4.5, SAN prompt, golden v1 (no thinking)": HAIKU_SAN,
                      "Haiku 4.5, UCI prompt, golden v1 (no thinking)": HAIKU_UCI,
                      "Haiku 4.5, UCI prompt, golden v1 (thinking)": HAIKU_THINKING,
                      "Haiku 4.5, UCI prompt, golden v2 (no thinking)": HAIKU_V2,
                      "Sonnet 4.6, UCI prompt, golden v1 (no thinking)": SONNET_UCI,
                      "Sonnet 4.6, UCI prompt, golden v2 (no thinking)": SONNET_V2}, INVALID_ROWS),
}


def render(table: Table) -> str:
    """The markdown table: best cell per row in bold (ties all bold), columns padded to width."""
    cols = {title: column(log) for title, log in table.columns.items()}
    grid = [[""] + list(cols)]
    for label, cell, better in table.rows:
        cells = [cell(c) for c in cols.values()]
        keys = [k for _, k in cells if k is not None]
        best = (max if better == "high" else min)(keys) if better and keys else None
        grid.append([label] + [f"**{text}**" if key is not None and key == best else text for text, key in cells])
    widths = [max(map(len, col)) for col in zip(*grid)]
    lines = ["| " + " | ".join(cell.ljust(w) for cell, w in zip(row, widths)) + " |" for row in grid]
    return "\n".join([lines[0], "|" + "|".join("-" * (w + 2) for w in widths) + "|", *lines[1:]])


if __name__ == "__main__":
    for key in sys.argv[1:] or TABLES:
        print(f"{TABLES[key].heading}\n\n{render(TABLES[key])}\n")
