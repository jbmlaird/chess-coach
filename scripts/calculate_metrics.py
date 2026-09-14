"""Derive the README's results-table metrics from an Inspect eval log"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import math
import rich
from inspect_ai.log import read_eval_log
from inspect_ai.scorer import CORRECT, INCORRECT, NOANSWER

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))
from move_parser import Outcome  # noqa: E402
from move_review import (  # noqa: E402
    CORRECT_REFUTATION,
    CORRECT_VERDICT,
    WRONG_REFUTATION,
    WRONG_VERDICT,
)

GOLDEN_META = json.loads((REPO / "golden" / "v2" / "golden_candidates.meta.json").read_text())
ARM_SIZES = {"blunder": GOLDEN_META["blunder_arm"], "best": GOLDEN_META["best_arm"]}
TOTAL = sum(ARM_SIZES.values())

PRICES = {
    "anthropic/claude-haiku-4-5": (1.00, 5.00),
    "anthropic/claude-sonnet-4-6": (3.00, 15.00),
    "anthropic/claude-opus-5": (5.00, 25.00),
    "mockllm/model": (0.00, 0.00),  # the pilot protocol's free first step
}

CLAIMED_BLUNDER_OUTCOMES = {
    CORRECT_REFUTATION,
    WRONG_REFUTATION,
    Outcome.ILLEGAL.name,
    Outcome.INVALID.name,
    Outcome.AMBIGUOUS.name,
}
KNOWN_OUTCOMES = CLAIMED_BLUNDER_OUTCOMES | {WRONG_VERDICT, CORRECT_VERDICT, Outcome.PARSE_ERROR.name}
# a lower-cased r/n/q/k prefix ("rd1", "qe4e8") fails python-chess's SAN regex and scores INVALID;
# a lower-cased b reads as the b-pawn file and lands in ILLEGAL instead, so it is not counted here
LOWERCASE_PIECE = re.compile(r"^[rnqk][a-h1-8x]")


SCORERS = ("legal_move", "ground_truth")


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def load(path: Path):
    return read_eval_log(path, exclude_fields={"messages", "events", "store", "attachments"})


def metrics(path: Path, verbose: bool = False) -> tuple:
    """(eval_log, metrics dict) for one committed run; print_metrics() renders the dict."""
    eval_log = load(path)

    stored = {score.name: score.metrics for score in eval_log.results.scores}
    assert set(stored) == set(SCORERS), f"unexpected scorers: {set(stored)}"
    legal_move_blunder_accuracy = stored["legal_move"]["blunder"].value
    legal_move_best_accuracy = stored["legal_move"]["best"].value
    legal_move_accuracy = stored["legal_move"]["all"].value
    ground_truth_blunder_accuracy = stored["ground_truth"]["blunder"].value
    ground_truth_best_accuracy = stored["ground_truth"]["best"].value
    ground_truth_accuracy = stored["ground_truth"]["all"].value

    ground_truth_outcomes = {"blunder": Counter(), "best": Counter()}
    legal_move_outcomes = {"blunder": Counter(), "best": Counter()}
    missing_refutation = {"blunder": 0, "best": 0}

    unfinished = Counter()
    empty_output = {"blunder": 0, "best": 0}
    lowercase_piece = 0

    for sample in eval_log.samples:
        arm = sample.metadata['GroundTruth']
        ground_truth = sample.scores["ground_truth"]
        legal_move = sample.scores["legal_move"]
        ground_truth_outcomes[arm][ground_truth.metadata["outcome"]] += 1
        legal_move_outcomes[arm][legal_move.metadata["outcome"]] += 1
        if legal_move.metadata["outcome"] == Outcome.INVALID.name and LOWERCASE_PIECE.match(legal_move.answer):
            lowercase_piece += 1
        if ground_truth.metadata["outcome"] == Outcome.PARSE_ERROR.name and ground_truth.value == INCORRECT:
            missing_refutation[arm] += 1
        if sample.output.stop_reason != "stop":
            unfinished[sample.output.stop_reason] += 1
        if not sample.output.completion:
            empty_output[arm] += 1
            assert ground_truth.value == NOANSWER, \
                f"{sample.id}: empty output must score NOANSWER, got {ground_truth.value}"

    for arm, counts in ground_truth_outcomes.items():
        unknown = set(counts) - KNOWN_OUTCOMES
        if unknown:
            raise ValueError(f"unknown ground_truth outcomes {unknown} on {arm} arm - "
                             f"update this script's interpretation before trusting its numbers")

    if verbose:
        print(ground_truth_outcomes)
        print(legal_move_outcomes)

    blunder_legal_moves = legal_move_outcomes["blunder"]
    blunder_legal = blunder_legal_moves[Outcome.LEGAL.name]
    blunder_illegal = blunder_legal_moves[Outcome.ILLEGAL.name]
    blunder_invalid = blunder_legal_moves[Outcome.INVALID.name] + blunder_legal_moves[Outcome.AMBIGUOUS.name]
    blunder_legal_move_parse_error = blunder_legal_moves[Outcome.PARSE_ERROR.name]
    assert blunder_legal + blunder_illegal + blunder_invalid + blunder_legal_move_parse_error == sum(
        blunder_legal_moves.values()) == ARM_SIZES["blunder"], \
        f"legal_move blunder-arm buckets don't reconcile: {dict(blunder_legal_moves)}"

    best_legal_moves = legal_move_outcomes["best"]
    best_legal = best_legal_moves[Outcome.LEGAL.name]
    best_illegal = best_legal_moves[Outcome.ILLEGAL.name]
    best_invalid = best_legal_moves[Outcome.INVALID.name] + best_legal_moves[Outcome.AMBIGUOUS.name]
    best_legal_move_parse_error = best_legal_moves[Outcome.PARSE_ERROR.name]
    assert best_legal + best_illegal + best_invalid + best_legal_move_parse_error == sum(
        best_legal_moves.values()) == ARM_SIZES["best"], \
        f"legal_move best-arm buckets don't reconcile: {dict(best_legal_moves)}"

    if verbose:
        seen = set()
        for sample in eval_log.samples:
            identifier = (sample.metadata['GroundTruth'],
                          sample.scores["legal_move"].metadata["outcome"],
                          sample.scores['ground_truth'].metadata["outcome"])
            if identifier not in seen:
                rich.print(sample.id)
                rich.print(sample.scores)
                rich.print(sample.metadata)
                seen.add(identifier)

    blunder_ground_truths = ground_truth_outcomes["blunder"]

    claimed_blunder = (sum(blunder_ground_truths[outcome] for outcome in CLAIMED_BLUNDER_OUTCOMES)
                       + missing_refutation["blunder"])
    missed_blunder = blunder_ground_truths[WRONG_VERDICT]
    abstained_blunder = blunder_ground_truths[Outcome.PARSE_ERROR.name] - missing_refutation["blunder"]
    assert claimed_blunder + missed_blunder + abstained_blunder == sum(
        blunder_ground_truths.values()) == ARM_SIZES["blunder"], \
        f"blunder-arm verdict buckets don't reconcile: {dict(blunder_ground_truths)}"

    refutation_correct = blunder_ground_truths[CORRECT_REFUTATION]
    refutation_wrong = blunder_ground_truths[WRONG_REFUTATION]
    refutation_unplayable = (blunder_ground_truths[Outcome.ILLEGAL.name]
                             + blunder_ground_truths[Outcome.INVALID.name]
                             + blunder_ground_truths[Outcome.AMBIGUOUS.name])
    refutation_missing = missing_refutation["blunder"]
    assert refutation_correct + refutation_wrong + refutation_unplayable + refutation_missing == claimed_blunder, \
        f"refutation ladder doesn't sum to claimed blunders: {dict(blunder_ground_truths)}"

    best_ground_truths = ground_truth_outcomes["best"]
    endorsed_best = best_ground_truths[CORRECT_VERDICT]
    claimed_blunder_on_best = best_ground_truths[WRONG_VERDICT]
    parse_error_best = best_ground_truths[Outcome.PARSE_ERROR.name]
    assert endorsed_best + claimed_blunder_on_best + parse_error_best == sum(
        best_ground_truths.values()) == ARM_SIZES["best"], \
        f"best-arm verdict buckets don't reconcile: {dict(best_ground_truths)}"

    blunder_recall = claimed_blunder / sum(blunder_ground_truths.values())
    blunder_precision = ratio(claimed_blunder, claimed_blunder + claimed_blunder_on_best)
    best_recall = endorsed_best / sum(best_ground_truths.values())
    best_precision = ratio(endorsed_best, endorsed_best + missed_blunder)

    substantiation = ratio(refutation_correct, claimed_blunder)

    assert math.isclose(best_legal / sum(best_legal_moves.values()), legal_move_best_accuracy), \
        "derived best-arm legality disagrees with the log's stored metric"
    assert math.isclose(endorsed_best / sum(best_legal_moves.values()), ground_truth_best_accuracy), \
        "derived best-arm accuracy disagrees with the log's stored metric"
    assert math.isclose(blunder_legal / sum(blunder_ground_truths.values()), legal_move_blunder_accuracy), \
        "derived blunder-arm legality disagrees with the log's stored metric"
    assert math.isclose(refutation_correct / sum(blunder_ground_truths.values()), ground_truth_blunder_accuracy), \
        "derived blunder-arm accuracy disagrees with the log's stored metric"

    (model, usage), = eval_log.stats.model_usage.items()
    return eval_log, {
        "legal_move_blunder_accuracy": legal_move_blunder_accuracy,
        "legal_move_best_accuracy": legal_move_best_accuracy,
        "legal_move_accuracy": legal_move_accuracy,
        "legal_move_blunder_parse_error": blunder_legal_move_parse_error,
        "legal_move_best_parse_error": best_legal_move_parse_error,
        "invalid_best_moves": sum(counts[Outcome.INVALID.name] for counts in legal_move_outcomes.values()),
        "lowercase_piece": lowercase_piece,
        "ground_truth_blunder_accuracy": ground_truth_blunder_accuracy,
        "ground_truth_best_accuracy": ground_truth_best_accuracy,
        "ground_truth_accuracy": ground_truth_accuracy,
        "ground_truth_blunder_abstained": abstained_blunder,
        "ground_truth_blunder_missing_refutation": refutation_missing,
        "ground_truth_best_parse_error": parse_error_best,
        "blunder_recall": blunder_recall,
        "blunder_precision": blunder_precision,
        "best_recall": best_recall,
        "best_precision": best_precision,
        "substantiation": substantiation,
        "refutation_unplayable": refutation_unplayable,
        "unfinished": dict(unfinished),
        "empty_output": empty_output,
        "cost": cost(model, usage),
        "model": model,
        "tool_use": eval_log.eval.task_args.get("tool_use", "none"),
    }


def cost(model: str, usage) -> float:
    if model not in PRICES:
        raise ValueError(f"no pricing for {model} - add it to PRICES")
    input_rate, output_rate = PRICES[model]
    # cache writes bill at 1.25x the input rate, cache reads at 0.10x
    billable_input = (usage.input_tokens
                      + 1.25 * (usage.input_tokens_cache_write or 0)
                      + 0.10 * (usage.input_tokens_cache_read or 0))
    return billable_input / 1_000_000 * input_rate + usage.output_tokens / 1_000_000 * output_rate


def print_metrics(m: dict) -> None:
    for name in ("legal_move_blunder_accuracy", "legal_move_best_accuracy", "legal_move_accuracy",
                 "legal_move_blunder_parse_error", "legal_move_best_parse_error"):
        print(f"{name}: {m[name]}")
    print(f"invalid best moves: {m['invalid_best_moves']}/{TOTAL} (lowercase piece letter: {m['lowercase_piece']})")
    for name in ("ground_truth_blunder_accuracy", "ground_truth_best_accuracy", "ground_truth_accuracy",
                 "ground_truth_blunder_abstained", "ground_truth_blunder_missing_refutation",
                 "ground_truth_best_parse_error", "blunder_recall", "blunder_precision", "best_recall",
                 "best_precision", "substantiation"):
        print(f"{name}: {m[name]}")
    print(f"unplayable refutations: {m['refutation_unplayable']}/{ARM_SIZES['blunder']}")
    print(f"unfinished samples (by stop_reason): {m['unfinished'] or 0}")
    print(f"empty output samples: {sum(m['empty_output'].values())}/{TOTAL} "
          f"(blunder arm {m['empty_output']['blunder']}, best arm {m['empty_output']['best']})")
    print(f"cost: ${m['cost']:.2f} for model: {m['model']}")

def compare(eval_log, other_path: Path) -> None:
    """Paired comparison on the sample ids two runs share (a replication, or a dataset revision)."""
    other = {s.id: s for s in load(other_path).samples}
    shared = [s for s in eval_log.samples if s.id in other]
    n = len(shared)
    print(f"shared rows with {other_path.name}: {n}")
    for scorer in SCORERS:
        was = [other[s.id].scores[scorer].value == CORRECT for s in shared]
        now = [s.scores[scorer].value == CORRECT for s in shared]
        lost = sum(a and not b for a, b in zip(was, now))
        gained = sum(b and not a for a, b in zip(was, now))
        # standard error of a paired difference in proportions: only the discordant rows carry variance
        se = math.sqrt(lost + gained - (gained - lost) ** 2 / n) / n
        print(f"{scorer} on shared rows: {sum(was)}/{n} -> {sum(now)}/{n} "
              f"(lost {lost}, gained {gained}, paired se {100 * se:.1f}pp)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log_file", type=Path, required=True)
    parser.add_argument("--verbose", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--compare_to", type=Path, help="an earlier run of the same instrument to pair with on shared sample ids")
    return parser.parse_args()


def main():
    parsed_args = parse_args()
    eval_log, m = metrics(parsed_args.log_file, parsed_args.verbose)
    print_metrics(m)
    if parsed_args.compare_to:
        compare(eval_log, parsed_args.compare_to)


if __name__ == "__main__":
    main()