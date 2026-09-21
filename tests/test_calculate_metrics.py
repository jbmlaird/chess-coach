"""Pins the CLI lines the README quotes in prose and nowhere else: the paired-replication figures, and the
abstention / parse-error split that the rendered "Format failures" row collapses into one number."""

import subprocess
import sys

import pytest

import calculate_metrics

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
        pytest.param(HAIKU_V2, None, [
            "sampling: provider defaults; max_tokens 32000; served by claude-haiku-4-5-20251001",
        ], id="haiku-run-configuration"),
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


def test_repeats_report_the_spread_over_a_run_and_its_siblings(tmp_path, capsys):
    """Two identical mock runs, the second filed under repeats/1/, give a zero spread."""
    from inspect_ai import eval as run_eval
    from move_review import positions
    run_eval(positions(), model="mockllm/model", log_dir=str(tmp_path / "mock"), display="none")
    run_eval(positions(), model="mockllm/model", log_dir=str(tmp_path / "mock" / "repeats" / "1"), display="none")
    calculate_metrics.repeats(next((tmp_path / "mock").glob("*.eval")))
    assert "ground_truth_accuracy over 2 runs: mean 0.0% sd 0.0pp se 0.0pp" in capsys.readouterr().out


def test_a_self_hosted_model_bills_by_the_hour():
    assert calculate_metrics.cost("vllm/Qwen/Qwen2.5-7B-Instruct", None, hours=0.5) == 0.32


def test_the_haiku_repeats_spread_reproduces():
    """The README's measured-noise sentence quotes these three printed lines."""
    out = subprocess.run([sys.executable, "scripts/calculate_metrics.py", "--log_file", str(column(HAIKU_V2).log), "--repeats"],
                         cwd=REPO, capture_output=True, text=True, check=True).stdout
    for line in ("ground_truth_accuracy over 6 runs: mean 13.0% sd 1.2pp se 0.5pp",
                 "legal_move_accuracy over 6 runs: mean 50.8% sd 2.8pp se 1.1pp",
                 "substantiation over 6 runs: mean 10.6% sd 1.5pp se 0.6pp"):
        assert line in out, out
