"""Every link from the README into the published log viewer names something committed under logs/ (a run
directory, a log file, or a golden sample of one): a re-run renames the log, and a link that 404s is worse than none."""

import csv
import re
from urllib.parse import unquote

import render_results

README = (render_results.REPO / "README.md").read_text()
LINKS = re.findall(r"https://jbmlaird\.github\.io/chess-coach/#/logs/([^/)\s]+)(?:/samples/sample/([A-Za-z0-9]+)/1)?", README)
IDS = {row["PuzzleId"] for version in ("v1", "v2")
       for row in csv.DictReader((render_results.REPO / "golden" / version / "golden_candidates.csv").open())}


def test_viewer_links_point_at_committed_logs_and_golden_samples():
    assert len(LINKS) > 10, "the README should link every column and the examples into the published viewer"
    for target, sample in LINKS:
        path = render_results.LOGS / unquote(target)
        assert path.exists(), unquote(target)
        assert not sample or (path.is_file() and sample in IDS), (unquote(target), sample)
