"""Golden tests: calculate_metrics.py must reproduce the README's published
numbers from the committed logs. Guards the number factory against silent
drift from inspect_ai upgrades or interpretation-rule edits."""

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent

HAIKU_V2_LOG = "logs/golden-v1/haiku-4-5/2026-08-20T16-46-50-00-00_Positions_ZUC6qqipkz3TMfHuWhSjdq.eval"
OPUS_LOG = "logs/golden-v1/opus-5/2026-08-21T10-18-31-00-00_Positions_8XbGRba8CoJjq3Ywj48RJd.eval"
SONNET_V2_LOG = "logs/golden-v1/sonnet-4-6/2026-08-20T16-47-36-00-00_Positions_BREFiymF2Vhthv64Yw7XFD.eval"
HAIKU_GOLDEN_V2_LOG = "logs/golden-v2/haiku-4-5/2026-09-08T12-50-15-00-00_Positions_HFFMrFK37edcScmKRHx8Ss.eval"
SONNET_GOLDEN_V2_LOG = "logs/golden-v2/sonnet-4-6/2026-09-08T12-51-04-00-00_Positions_BT4gKBq5Zio6DpXCrg8En4.eval"


def run_metrics(*cli: str) -> str:
    result = subprocess.run(
        [sys.executable, "scripts/calculate_metrics.py", *cli],
        cwd=REPO, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


@pytest.mark.parametrize(
    "cli,expected_lines",
    [
        pytest.param(["--log_file", HAIKU_V2_LOG], [
            "ground_truth_accuracy: 0.144",
            "blunder_recall: 0.885",
            "unplayable refutations: 88/200",
            "substantiation: 0.11864",
            "cost: $1.02",
        ], id="haiku-v2-reproduces-readme-column"),
        pytest.param(["--log_file", OPUS_LOG], [
            "ground_truth_accuracy: 0.348",
            "blunder_recall: 0.38",
            "ground_truth_blunder_abstained: 30",
            "ground_truth_best_parse_error: 5",
            "substantiation: 0.73684",
            "cost: $109.52",
        ], id="opus-reproduces-readme-column-incl-abstentions"),
        pytest.param(["--log_file", HAIKU_GOLDEN_V2_LOG, "--compare_to", HAIKU_V2_LOG], [
            "ground_truth_accuracy: 0.128",
            "blunder_recall: 0.905",
            "unplayable refutations: 95/200",
            "invalid best moves: 32/250 (lowercase piece letter: 32)",
            "substantiation: 0.08839",
            "cost: $1.02",
            "legal_move on shared rows: 116/245 -> 137/245 (lost 41, gained 62, paired se 4.1pp)",
            "ground_truth on shared rows: 36/245 -> 32/245 (lost 20, gained 16, paired se 2.4pp)",
        ], id="haiku-golden-v2-reproduces-readme-column-and-replication"),
        pytest.param(["--log_file", SONNET_GOLDEN_V2_LOG, "--compare_to", SONNET_V2_LOG], [
            "ground_truth_accuracy: 0.22",
            "blunder_recall: 0.84",
            "unplayable refutations: 39/200",
            "invalid best moves: 1/250 (lowercase piece letter: 1)",
            "substantiation: 0.22619",
            "cost: $4.70",
            "legal_move on shared rows: 177/245 -> 178/245 (lost 31, gained 32, paired se 3.2pp)",
            "ground_truth on shared rows: 53/245 -> 54/245 (lost 23, gained 24, paired se 2.8pp)",
        ], id="sonnet-golden-v2-reproduces-readme-column-and-replication"),
    ],
)
def test_published_numbers_reproduce(cli, expected_lines):
    stdout = run_metrics(*cli)
    for line in expected_lines:
        assert line in stdout, f"missing {line!r} in output:\n{stdout}"