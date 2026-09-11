"""Every results table in the README is the renderer's output, line for line: a cell can only change by
re-rendering from the committed logs. The two damage rows need the engine and a warm grader cache (a cold
cache means minutes of engine time), so without both (CI, or a fresh clone) every other row is checked."""

import re
from dataclasses import replace

import pytest

import grade_logs
import render_results
from engine import STOCKFISH_PATH

README = (render_results.REPO / "README.md").read_text()
DAMAGE_ROWS = STOCKFISH_PATH is not None and grade_logs.CACHE_PATH.exists()


def body(table: str) -> list[str]:
    """The table's lines minus padding and the separator, both of which depend on which rows are rendered."""
    lines = [re.sub(" +", " ", line) for line in table.splitlines() if not line.startswith("|-")]
    return lines if DAMAGE_ROWS else [line for line in lines if "damage" not in line]


@pytest.mark.parametrize("key", render_results.TABLES)
def test_readme_table_is_the_rendered_one(key):
    table = render_results.TABLES[key]
    if not DAMAGE_ROWS:
        table = replace(table, rows=tuple(r for r in table.rows if "damage" not in r[0]))
    section = README[README.index(table.heading):]  # the first table after the heading is the one
    assert body(re.search(r"(?m)^\|.*(?:\n\|.*)*", section).group()) == body(render_results.render(table))
