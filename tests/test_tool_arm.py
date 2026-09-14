"""The tool arms end to end through the real MCP server, driven by a scripted model: a relay is
scored correct, a tool error reaches the model and the sample still scores, and an endless caller
hits the turn limit. Engine facts come from the certified reference, not from these tests."""

import json

import inspect_ai.tool._mcp._local as mcp_local
import pytest
from inspect_ai import eval as run_eval
from inspect_ai.model import ModelOutput, get_model

import tool_metrics
from engine import STOCKFISH_PATH
from move_review import positions
from test_stockfish_mcp import F16CC_FEN

pytestmark = pytest.mark.skipif(STOCKFISH_PATH is None, reason="stockfish binary not installed")
ROWS = {s.input: s.metadata for s in positions().dataset}  # FEN -> the row the scorers read


def row(prompt: str) -> dict:
    """The scripted model can only see the prompt, like a real one: find the row whose FEN it carries."""
    return next(m for fen, m in ROWS.items() if fen in prompt)


def ask(prompt: str, moves) -> ModelOutput:
    return ModelOutput.for_tool_call("mockllm/model", "analyse", {"fen": row(prompt)["FEN"], "moves": moves})


def answer(refutation: str, best_move: str) -> ModelOutput:
    return ModelOutput.from_content("mockllm/model", f"VERDICT: BLUNDER\nREFUTATION: {refutation}\nBEST_MOVE: {best_move}\nEXPLANATION: x")


def relaying(messages, tools, tool_choice, config) -> ModelOutput:
    """Asks the engine about the given position, then the played move, then copies both replies."""
    prompt, replies = messages[0].text, [json.loads(m.text) for m in messages if m.role == "tool"]
    if len(replies) == 2:
        return answer(replies[1]["best_move"], replies[0]["best_move"])
    return ask(prompt, [row(prompt)["PlayedMove"]] if replies else [])


def blundering_into_the_tool(messages, tools, tool_choice, config) -> ModelOutput:
    """Sends the engine an illegal move, then answers on its own."""
    if messages[-1].role == "tool":
        return answer("a1a1", row(messages[0].text)["PlayedMove"])
    return ask(messages[0].text, ["e2e5"])


def endless(messages, tools, tool_choice, config) -> ModelOutput:
    # a mated position (pinned in test_stockfish_mcp.py) so each of the turns costs no engine search
    return ModelOutput.for_tool_call("mockllm/model", "analyse", {"fen": F16CC_FEN, "moves": ["c7d8", "g5g7"]})


def run(arm: str, behaviour, ids, tmp_path):
    model = get_model("mockllm/model", custom_outputs=behaviour)
    log, = run_eval(positions(tool_use=arm), model=model, sample_id=ids, log_dir=str(tmp_path),
                    max_connections=2, display="none")
    assert log.status == "success", log.error
    return log


def test_a_relayed_refutation_scores_correct_and_is_seen_as_a_relay(tmp_path, monkeypatch):
    spawns, spawn = [], mcp_local._stdio_client_forwarding_stderr
    monkeypatch.setattr(mcp_local, "_stdio_client_forwarding_stderr", lambda *a: spawns.append(a) or spawn(*a))
    log = run("required", relaying, ["CgxAk", "qenum"], tmp_path)
    by_id = {s.id: s for s in log.samples}
    assert by_id["CgxAk"].scores["ground_truth"].metadata["outcome"] == "CORRECT_REFUTATION"
    assert by_id["qenum"].scores["ground_truth"].metadata["outcome"] == "WRONG_VERDICT"  # a best move called a blunder
    stats = tool_metrics.sample_stats(by_id["CgxAk"])
    assert stats["calls"] == 2 and stats["errors"] == 0 and stats["answered_after_tool"]
    assert stats["queried_given"] and stats["relay"] and stats["beyond_engine"] == 0
    assert log.eval.task_args == {"tool_use": "required"} and log.eval.metadata["tool_engine"]["name"] == "Stockfish 18"
    assert len(spawns) == 2  # one server per sample, held open for both calls: not one per tool call


def test_a_tool_error_reaches_the_model_and_the_sample_still_scores(tmp_path):
    log = run("optional", blundering_into_the_tool, ["CgxAk"], tmp_path)
    sample, = log.samples
    tool_reply, = [m for m in sample.messages if m.role == "tool"]
    assert tool_reply.error is not None and "not legal" in tool_reply.error.message
    assert sample.scores["legal_move"].metadata["outcome"] == "LEGAL"
    assert tool_metrics.sample_stats(sample)["errors"] == 1


def test_an_endless_caller_is_stopped_by_the_turn_limit(tmp_path):
    log = run("silent", endless, ["CgxAk"], tmp_path)
    sample, = log.samples
    assert sample.limit.type == "turn" and sample.scores["ground_truth"].value == "N"
    assert not tool_metrics.sample_stats(sample)["answered_after_tool"]
