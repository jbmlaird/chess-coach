"""Pins the CLI lines the README quotes in prose and nowhere else: the paired-replication figures, and the
abstention / parse-error split that the rendered "Format failures" row collapses into one number."""

import subprocess
import sys

import pytest

from render_results import HAIKU_OPTIONAL, HAIKU_UCI, HAIKU_V2, OPUS, REPO, SONNET_OPTIONAL, SONNET_UCI, SONNET_V2, column


def run_metrics(log: str, compare_to: str | None = None) -> str:
    """Runs the CLI on a committed run, named by file under logs/ or by its directory (resolved here, not at collection)."""
    cli = ["--log_file", str(column(log).log)] + (["--compare_to", str(column(compare_to).log)] if compare_to else [])
    result = subprocess.run([sys.executable, "scripts/calculate_metrics.py", *cli], cwd=REPO, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    return result.stdout


@pytest.mark.parametrize(
    "log,compare_to,expected_lines",
    [
        pytest.param(OPUS, None, [
            "ground_truth_blunder_abstained: 30",
            "ground_truth_best_parse_error: 5",
            "cost: $109.52",
        ], id="opus-abstentions-behind-the-format-failures-row"),
        pytest.param(HAIKU_V2, HAIKU_UCI, [
            "legal_move on shared rows: 116/245 -> 137/245 (lost 41, gained 62, paired se 4.1pp)",
            "ground_truth on shared rows: 36/245 -> 32/245 (lost 20, gained 16, paired se 2.4pp)",
        ], id="haiku-replication-paragraph"),
        pytest.param(SONNET_V2, SONNET_UCI, [
            "legal_move on shared rows: 177/245 -> 178/245 (lost 31, gained 32, paired se 3.2pp)",
            "ground_truth on shared rows: 53/245 -> 54/245 (lost 23, gained 24, paired se 2.8pp)",
        ], id="sonnet-replication-paragraph"),
        # the tables carry every other grounded figure the prose quotes; only the stop-reason split lives here
        pytest.param(HAIKU_OPTIONAL, None, ["unfinished samples (by stop_reason): {'tool_calls': 37}"],
                     id="haiku-optional-limit-hits"),
        pytest.param(SONNET_OPTIONAL, None, ["unfinished samples (by stop_reason): {'tool_calls': 4}"],
                     id="sonnet-optional-limit-hits"),
    ],
)
def test_published_numbers_reproduce(log, compare_to, expected_lines):
    stdout = run_metrics(log, compare_to)
    for line in expected_lines:
        assert line in stdout, f"missing {line!r} in output:\n{stdout}"
