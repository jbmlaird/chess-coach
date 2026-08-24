"""Golden tests: calculate_metrics.py must reproduce the README's published
numbers from the committed logs. Guards the number factory against silent
drift from inspect_ai upgrades or interpretation-rule edits."""

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent

HAIKU_V2_LOG = "logs/haiku-4-5/2026-08-20T16-46-50-00-00_Positions_ZUC6qqipkz3TMfHuWhSjdq.eval"
OPUS_LOG = "logs/opus-5/2026-08-21T10-18-31-00-00_Positions_8XbGRba8CoJjq3Ywj48RJd.eval"


def run_metrics(log_file: str) -> str:
    result = subprocess.run(
        [sys.executable, "scripts/calculate_metrics.py", "--log_file", log_file],
        cwd=REPO, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


@pytest.mark.parametrize(
    "log_file,expected_lines",
    [
        pytest.param(HAIKU_V2_LOG, [
            "ground_truth_accuracy: 0.144",
            "blunder_recall: 0.885",
            "unplayable refutations: 88/200",
            "substantiation: 0.11864",
            "cost: $1.02",
        ], id="haiku-v2-reproduces-readme-column"),
        pytest.param(OPUS_LOG, [
            "ground_truth_accuracy: 0.348",
            "blunder_recall: 0.38",
            "ground_truth_blunder_abstained: 30",
            "ground_truth_best_parse_error: 5",
            "substantiation: 0.73684",
            "cost: $109.52",
        ], id="opus-reproduces-readme-column-incl-abstentions"),
    ],
)
def test_published_numbers_reproduce(log_file, expected_lines):
    stdout = run_metrics(log_file)
    for line in expected_lines:
        assert line in stdout, f"missing {line!r} in output:\n{stdout}"