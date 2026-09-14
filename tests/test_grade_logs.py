"""A tool-arm log is graded only if the model's tool was the reference engine."""

import pytest

import grade_logs
import render_results


def test_a_tool_arm_log_with_a_different_engine_is_refused(monkeypatch):
    col = render_results.column(render_results.HAIKU_V2)
    real = grade_logs.read_eval_log(col.log, header_only=True)
    real.eval.task_args = {"tool_use": "required"}
    real.eval.metadata = {"tool_engine": {"name": "Stockfish 17", "limit": "Limit(nodes=1000000)", "config": {"Threads": 1, "Hash": 128}}}
    monkeypatch.setattr(grade_logs, "read_eval_log", lambda *a, **k: real)
    with pytest.raises(SystemExit, match="Stockfish 17"):
        grade_logs.grade(col.log, col.golden)
