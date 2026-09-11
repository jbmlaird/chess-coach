"""Pins the CLI lines the README quotes in prose and nowhere else: the paired-replication figures, and the
abstention / parse-error split that the rendered "Format failures" row collapses into one number."""

import subprocess
import sys

import pytest

from render_results import HAIKU_UCI, HAIKU_V2, LOGS, OPUS, REPO, SONNET_UCI, SONNET_V2


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
        pytest.param(["--log_file", str(LOGS / OPUS)], [
            "ground_truth_blunder_abstained: 30",
            "ground_truth_best_parse_error: 5",
            "cost: $109.52",
        ], id="opus-abstentions-behind-the-format-failures-row"),
        pytest.param(["--log_file", str(LOGS / HAIKU_V2), "--compare_to", str(LOGS / HAIKU_UCI)], [
            "legal_move on shared rows: 116/245 -> 137/245 (lost 41, gained 62, paired se 4.1pp)",
            "ground_truth on shared rows: 36/245 -> 32/245 (lost 20, gained 16, paired se 2.4pp)",
        ], id="haiku-replication-paragraph"),
        pytest.param(["--log_file", str(LOGS / SONNET_V2), "--compare_to", str(LOGS / SONNET_UCI)], [
            "legal_move on shared rows: 177/245 -> 178/245 (lost 31, gained 32, paired se 3.2pp)",
            "ground_truth on shared rows: 53/245 -> 54/245 (lost 23, gained 24, paired se 2.8pp)",
        ], id="sonnet-replication-paragraph"),
    ],
)
def test_published_numbers_reproduce(cli, expected_lines):
    stdout = run_metrics(*cli)
    for line in expected_lines:
        assert line in stdout, f"missing {line!r} in output:\n{stdout}"
