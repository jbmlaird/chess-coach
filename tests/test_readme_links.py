"""Every deep link from the README into the published log viewer names a committed log and a golden sample:
a re-run changes the log's file name, and a link that 404s is worse than no link."""

import csv
import re
from urllib.parse import unquote

import render_results

README = (render_results.REPO / "README.md").read_text()
LINKS = re.findall(r"https://jbmlaird\.github\.io/chess-coach/#/logs/([^/]+)/samples/sample/([A-Za-z0-9]+)/1", README)
IDS = {row["PuzzleId"] for version in ("v1", "v2")
       for row in csv.DictReader((render_results.REPO / "golden" / version / "golden_candidates.csv").open())}


def test_viewer_links_point_at_committed_logs_and_golden_samples():
    assert LINKS, "the README should link into the published viewer"
    for log, sample in LINKS:
        assert (render_results.LOGS / unquote(log)).is_file(), unquote(log)
        assert sample in IDS, sample
