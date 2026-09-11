"""Every results table in the README is the renderer's output, verbatim: a cell can only change
by re-rendering from the committed logs."""

from pathlib import Path

import pytest

import render_results
from engine import STOCKFISH_PATH

README = (Path(__file__).parent.parent / "README.md").read_text()


@pytest.mark.parametrize("key", render_results.TABLES)
def test_readme_table_is_the_rendered_one(key):
    table = render_results.TABLES[key]
    if STOCKFISH_PATH is None and table.rows is render_results.ROWS:
        pytest.skip("damage rows need the engine")
    rendered = render_results.render(table)
    assert table.heading in README
    assert rendered in README, f"README table under {table.heading!r} differs from:\n{rendered}"
    assert README.index(table.heading) < README.index(rendered)
