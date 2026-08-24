"""Grade every model-suggested move in an eval log against the frozen engine
reference, in win-percentage points.

For each sample: parse the model's BEST_MOVE, resolve the eval of the position
after that move (sidecar shortcut -> cache -> live Engine.grader() analysis),
and compute the damage relative to best play:

    damage_pp = win_percent(before) - (100 - win_percent(after_move))

Live analyses are cached (graded_moves_cache.csv, gitignored, regenerable):
write-through, keyed (PuzzleId, ParentMove, MoveUci), raw score facts only.
The cache records the engine provenance it was built with and the script
refuses to mix instruments.

Usage:
    uv run python scripts/grade_logs.py --log_file logs/opus-5/<file>.eval
"""

import argparse
import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

import chess
from inspect_ai.log import read_eval_log

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))
from engine import Engine, move_damage_pp, win_percent  # noqa: E402
from move_parser import Outcome, parse_move_field  # noqa: E402
from move_review import WRONG_REFUTATION  # noqa: E402

CACHE_PATH = REPO / "graded_moves_cache.csv"
# 5pp is the measured engine-noise floor (repeat-best sidecar damage spans
# -7.6..+4.8pp) and sits below Lichess's 10pp "inaccuracy" band - a graded
# move under this bar is indistinguishable from best play by this instrument.
GOOD_SUGGESTION_THRESHOLD_PP = 5.0
# +230cp retained. Empirically never reached: the max observed retention on a
# wrong refutation is ~50% - the line reporting this is an honest null.
STILL_WINNING_THRESHOLD_PCT = 70.0


def load_reference(golden: Path) -> tuple[dict[str, dict], set[str]]:
    meta = json.loads(golden.with_suffix(".meta.json").read_text())
    candidates_meta = json.loads((REPO / "golden_candidates.meta.json").read_text())
    if meta["input_sha256"] != candidates_meta["csv_sha256"]:
        sys.exit(f"{golden_meta.name} certifies input {meta['input_sha256'][:12]}... but "
                 f"golden_candidates.meta.json froze {candidates_meta['csv_sha256'][:12]}... "
                 f"- the reference does not describe this dataset")

    with open(golden, newline="") as f:
        reference = {row["PuzzleId"]: row for row in csv.DictReader(f)}
    assert len(reference) == meta["rows"] == 250, \
        f"expected 250 reference rows, got {len(reference)} (meta says {meta['rows']})"

    already_lost = {row["puzzle_id"] for row in meta["audit"]["already_lost_before"]["rows"]}
    return reference, already_lost


def load_cache(engine_provenance: dict) -> dict[tuple[str, str, str], tuple[int, int | None]]:
    if not CACHE_PATH.exists():
        CACHE_PATH.write_text(f"# engine: {json.dumps(engine_provenance)}\n"
                              "PuzzleId,ParentMove,MoveUci,ScoreCentipawns,MateIn\n")
        return {}
    text = CACHE_PATH.read_text()
    if not text.endswith("\n"):
        text = text[:text.rfind("\n") + 1]
        CACHE_PATH.write_text(text)
    lines = text.splitlines()
    recorded = json.loads(lines[0].removeprefix("# engine: "))
    if recorded != engine_provenance:
        sys.exit(f"{CACHE_PATH.name} was built by a different engine configuration:\n"
                 f"  cache:  {recorded}\n  live:   {engine_provenance}\n"
                 f"Delete the cache to regrade under the current engine, or restore the old binary.")
    cache: dict[tuple[str, str, str], tuple[int, int | None]] = {}
    for row in csv.DictReader(lines[1:]):
        cache[(row["PuzzleId"], row["ParentMove"], row["MoveUci"])] = (
            int(row["ScoreCentipawns"]), int(row["MateIn"]) if row["MateIn"] else None)
    return cache


def append_cache(key: tuple[str, str, str], score_cp: int, mate_in: int | None) -> None:
    with open(CACHE_PATH, "a", newline="") as f:
        csv.writer(f, lineterminator="\n").writerow(
            [*key, score_cp, "" if mate_in is None else mate_in])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log_file", type=Path, required=True)
    parser.add_argument("--golden", type=Path, default=REPO / "golden_engine.csv")
    return parser.parse_args()


def resolve_after_eval(engine: Engine, cache: dict, stats: dict,
                       puzzle_id: str, fen: str, parent_move: str,
                       move_uci: str) -> tuple[int, int | None]:
    """(score_centipawns, mate_in) of the position after playing parent_move
    (if any) then move_uci from fen: cache first, live analysis on miss."""
    key = (puzzle_id, parent_move, move_uci)
    if key in cache:
        stats["cache_hits"] += 1
        return cache[key]
    board = chess.Board(fen)
    if parent_move:
        board.push_uci(parent_move)
    board.push_uci(move_uci)
    verdict = engine.analyse(board)
    cache[key] = (verdict.score_centipawns, verdict.mate_in)
    append_cache(key, verdict.score_centipawns, verdict.mate_in)
    stats["live_analyses"] += 1
    return cache[key]


def resolve_log_file(path: Path) -> Path:
    """Accept a .eval file or a directory containing exactly one."""
    if path.is_dir():
        candidates = sorted(path.glob("*.eval"))
        if len(candidates) != 1:
            sys.exit(f"{path} contains {len(candidates)} .eval files - pass the file explicitly")
        return candidates[0]
    return path


def main() -> None:
    args = parse_args()
    reference, already_lost = load_reference(args.golden)

    log_file = resolve_log_file(args.log_file)
    log = read_eval_log(log_file, exclude_fields={"messages", "events", "store", "attachments"})
    model = log.eval.model
    samples = log.samples or []
    assert len(samples) == 250, f"expected 250 samples, got {len(samples)}"

    stats = {"cache_hits": 0, "sidecar_hits": 0, "live_analyses": 0}
    # damage split by (arm, endorsed): endorsed = the model repeated the played
    # move, so on blunder rows the charge is the blunder's own damage (a
    # detection failure); proposed = the model's replacement, graded on merit
    damages: dict[tuple[str, bool], list[float]] = {
        ("blunder", True): [], ("blunder", False): [],
        ("best", True): [], ("best", False): [],
    }
    damages_excl_already_lost: list[float] = []
    ungraded: Counter[str] = Counter()  # parser outcome name -> count
    refutation_deltas: list[float] = []
    still_winning = 0

    with Engine.grader() as engine:
        cache = load_cache(engine.provenance)
        for sample in samples:
            puzzle_id = str(sample.id)
            ref = reference[puzzle_id]
            arm = sample.metadata["Arm"]
            fen = sample.metadata["FEN"]
            before_cp = int(ref["BeforeScoreCentipawns"])

            parsed = parse_move_field(fen, sample.output.completion, "BEST_MOVE")
            if parsed.outcome != Outcome.LEGAL:
                ungraded[parsed.outcome.name] += 1
            else:
                endorsed = parsed.uci == sample.metadata["PlayedMove"]
                if endorsed:
                    stats["sidecar_hits"] += 1
                    after_cp = int(ref["AfterPlayedScoreCentipawns"])
                else:
                    after_cp, _ = resolve_after_eval(
                        engine, cache, stats, puzzle_id, fen, "", parsed.uci)
                damage = move_damage_pp(before_cp, after_cp)
                damages[(arm, endorsed)].append(damage)
                if arm == "blunder" and puzzle_id not in already_lost:
                    damages_excl_already_lost.append(damage)

            # refutation quality: only rows where the model claimed BLUNDER
            # with a legal-but-wrong reply
            ground_truth_score = sample.scores["ground_truth"]
            if ground_truth_score.metadata["outcome"] == WRONG_REFUTATION:
                after_played_board = chess.Board(fen)
                after_played_board.push_uci(sample.metadata["PlayedMove"])
                model_ref = parse_move_field(
                    after_played_board.fen(), sample.output.completion, "REFUTATION")
                assert model_ref.outcome == Outcome.LEGAL, \
                    f"{sample.id}: WRONG_REFUTATION implies a legal refutation"
                ref_cp, _ = resolve_after_eval(
                    engine, cache, stats, puzzle_id, fen,
                    sample.metadata["PlayedMove"], model_ref.uci)
                model_refuter_pct = 100 - win_percent(ref_cp)
                certified_refuter_pct = 100 - win_percent(int(ref["AfterRefutationScoreCentipawns"]))
                refutation_deltas.append(certified_refuter_pct - model_refuter_pct)
                if model_refuter_pct >= STILL_WINNING_THRESHOLD_PCT:
                    still_winning += 1

    arm_sizes = Counter(row["Arm"] for row in reference.values())
    report(model, damages, damages_excl_already_lost, ungraded,
           refutation_deltas, still_winning, stats, dict(arm_sizes))


def describe(damage_list: list[float]) -> str:
    if not damage_list:
        return "n=0"
    good = sum(1 for d in damage_list if d <= GOOD_SUGGESTION_THRESHOLD_PP)
    return (f"n={len(damage_list)} mean={statistics.fmean(damage_list):.1f}pp "
            f"median={statistics.median(damage_list):.1f}pp "
            f"good(<={GOOD_SUGGESTION_THRESHOLD_PP:g}pp)={good}/{len(damage_list)} "
            f"({100 * good / len(damage_list):.0f}%)")


def report(model: str, damages: dict[tuple[str, bool], list[float]], damages_excl: list[float],
           ungraded: Counter, refutation_deltas: list[float],
           still_winning: int, stats: dict, arm_sizes: dict[str, int]) -> None:
    blunder_all = damages[("blunder", True)] + damages[("blunder", False)]
    best_all = damages[("best", True)] + damages[("best", False)]

    print(f"\n=== {model} vs golden v1 ===")
    print("damage = win% the suggested move gives away vs best play; 0 = engine-perfect,")
    print(f"<= {GOOD_SUGGESTION_THRESHOLD_PP:g}pp = within engine noise of best, ~50pp = threw an even game\n")

    total = sum(arm_sizes.values())
    print(f"{len(blunder_all) + len(best_all)} of {total} answers contained a legal BEST_MOVE and were graded")
    if ungraded:
        print(f"  not graded: {dict(ungraded)}")

    print(f"\nBlunder arm - {arm_sizes['blunder']} puzzles where the played move is a certified blunder:")
    endorsed = damages[("blunder", True)]
    proposed = damages[("blunder", False)]
    print(f"  {len(endorsed)} suggestions REPEATED the blunder (model judged it fine):")
    print(f"    charged the blunder's own cost -> {describe(endorsed)}")
    print(f"  {len(proposed)} suggestions PROPOSED a different move (the actual coaching):")
    print(f"    {describe(proposed)}")
    print(f"  all {len(blunder_all)} combined (mixes the two - avoid as a headline): {describe(blunder_all)}")
    print(f"    same, excluding the {len(blunder_all) - len(damages_excl)} already-lost rows: {describe(damages_excl)}")

    print(f"\nBest arm - {arm_sizes['best']} puzzles where the played move is the engine-best move:")
    correct = damages[("best", True)]
    false_pos = damages[("best", False)]
    print(f"  {len(correct)} suggestions correctly repeated it (should grade ~0): {describe(correct)}")
    print(f"  {len(false_pos)} wrongly 'improved' a perfect move: {describe(false_pos)}")

    print(f"\nRefutations - rows where the model called the blunder but named the WRONG punishing reply:")
    if refutation_deltas:
        print(f"  n={len(refutation_deltas)}; on average the wrong reply keeps "
              f"{statistics.fmean(refutation_deltas):.1f}pp LESS winning advantage than the certified reply")
        print(f"  wrong but still clearly winning (>= {STILL_WINNING_THRESHOLD_PCT:g}% win): "
              f"{still_winning}/{len(refutation_deltas)}")
    else:
        print("  n=0")

    print(f"\nengine work: {stats['live_analyses']} fresh analyses, "
          f"{stats['cache_hits']} cache hits, {stats['sidecar_hits']} sidecar lookups")


if __name__ == "__main__":
    main()
