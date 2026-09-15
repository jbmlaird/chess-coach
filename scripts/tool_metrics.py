"""Tool-arm metrics from a log's transcripts: what the model asked the engine, and whether its answer
came from the engine's reply. A script, not a scorer: the two scorers stay the ruler.

Usage:
    uv run python scripts/tool_metrics.py --log_file <.eval>     # any sample count; pilots project cost
"""

import argparse
import json
import sys
from pathlib import Path

import chess
from inspect_ai.log import read_eval_log

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))
from calculate_metrics import ARM_SIZES, cost  # noqa: E402
from move_parser import Outcome, parse_move_field  # noqa: E402


def sample_stats(sample) -> dict:
    calls = [call for m in sample.messages if m.role == "assistant" for call in m.tool_calls or []]
    replies = {m.tool_call_id: m for m in sample.messages if m.role == "tool"}
    best_moves, engine_moves = {}, set()  # board epd -> the engine's best moves there; every move it returned
    for call in calls:
        reply = replies.get(call.id)
        if reply is not None and reply.error is None:
            board = chess.Board(call.arguments["fen"])
            for move in call.arguments.get("moves") or []:
                board.push_uci(move)
            verdict = json.loads(reply.text)
            # the same position queried with different move clocks can get a different answer: keep every answer
            best_moves.setdefault(board.epd(), set()).add(verdict["best_move"])
            engine_moves.update([verdict["best_move"], *verdict["principal_variation"]])
    fen, completion = sample.metadata["FEN"], sample.output.completion
    board = chess.Board(fen)
    given = best_moves.get(board.epd(), set())
    board.push_uci(sample.metadata["PlayedMove"])
    after = best_moves.get(board.epd(), set())
    best = parse_move_field(fen, completion, "BEST_MOVE")
    refutation = parse_move_field(board.fen(), completion, "REFUTATION")
    as_reply = parse_move_field(board.fen(), completion, "BEST_MOVE")  # the student's move read as the opponent's
    legal = [p for p in (best, refutation) if p.outcome == Outcome.LEGAL]
    return {
        "calls": len(calls),
        "errors": sum(r.error is not None for r in replies.values()),
        "answered_after_tool": bool(best_moves) and not sample.output.message.tool_calls,
        "queried_given": bool(given),
        "queried_after": bool(after),
        "best_legal": best.outcome == Outcome.LEGAL,
        "relay": best.outcome == Outcome.LEGAL and best.uci in given,
        "legal_fields": len(legal),
        "beyond_engine": sum(p.uci not in engine_moves for p in legal),
        "frame_error": as_reply.outcome == Outcome.LEGAL and as_reply.uci in after,
    }


def sample_cost(sample, model: str) -> float:
    usage = sample.model_usage.get(model)
    return cost(model, usage) if usage is not None else 0.0


def tool_metrics(path: Path) -> dict:
    """{arm: per-arm aggregates} for every golden arm."""
    log = read_eval_log(path, resolve_attachments=True, exclude_fields={"events", "store"})
    arms = {}
    for arm in ARM_SIZES:
        samples = [s for s in log.samples if s.metadata["Arm"] == arm]
        stats = [sample_stats(s) for s in samples]
        n, calls = len(samples), sum(t["calls"] for t in stats)
        arms[arm] = {
            "n": n,
            "calls_mean": calls / n if n else 0.0,
            "tool_calls": calls,
            "tool_errors": sum(t["errors"] for t in stats),
            "samples_with_error": sum(t["errors"] > 0 for t in stats),
            "answered_after_tool": sum(t["answered_after_tool"] for t in stats),
            "relay": sum(t["relay"] for t in stats),
            "relay_n": sum(t["queried_given"] and t["best_legal"] for t in stats),
            "beyond_engine": sum(t["beyond_engine"] for t in stats),
            "legal_fields": sum(t["legal_fields"] for t in stats),
            "frame_errors": sum(t["frame_error"] for t in stats),
            "frame_n": sum(t["queried_after"] for t in stats),
            "cost": [sample_cost(s, log.eval.model) for s in samples],
            "unfinished": sum(s.output.stop_reason != "stop" or s.limit is not None for s in samples),
            "unbilled": sum(s.error is not None or s.model_usage.get(log.eval.model) is None for s in samples),
        }
    return arms


def projection(arms: dict) -> str:
    """The full run's cost from a pilot, stratified by arm. A turn-limited sample is fully billed (its discarded
    generation is in its usage); a sample without usage is not, and censored pilots bias low."""
    if any(t["unbilled"] for t in arms.values()):
        sys.exit("pilot has samples without usage - no projection (censored pilots bias low)")
    projected = sum(ARM_SIZES[arm] * sum(t["cost"]) / t["n"] for arm, t in arms.items() if t["n"])
    worst = sum(ARM_SIZES.values()) * max(c for t in arms.values() for c in t["cost"])
    return f"projected full run ${projected:.2f} (stratified by arm); worst case ${worst:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log_file", type=Path, required=True)
    path = parser.parse_args().log_file
    e = read_eval_log(path, header_only=True).eval
    print(f"{e.model} | task_version {e.task_version} | task_args {e.task_args} | "
          f"tool_engine {'recorded' if (e.metadata or {}).get('tool_engine') else 'none'} | "
          f"max_connections {e.model_generate_config.max_connections}")
    arms = tool_metrics(path)
    for arm, t in arms.items():
        print(
            f"[{arm} n={t['n']}] tool calls/sample {t['calls_mean']:.2f}; tool errors {t['tool_errors']}/{t['tool_calls']} calls, "
            f"samples with an error {t['samples_with_error']}/{t['n']}; answered after a tool call {t['answered_after_tool']}/{t['n']}; "
            f"relay fidelity {t['relay']}/{t['relay_n']}; beyond-engine answers {t['beyond_engine']}/{t['legal_fields']} legal fields; "
            f"frame errors {t['frame_errors']}/{t['frame_n']}; unfinished {t['unfinished']}")
    if sum(t["n"] for t in arms.values()) < sum(ARM_SIZES.values()):  # a pilot
        print(projection(arms))


if __name__ == "__main__":
    main()
