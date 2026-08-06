"""Extract post-cutoff puzzles from the Lichess puzzle archive.

Frontier labs' LLMs have a knowledge cutoff of May 2026, this compares a dump of Lichess puzzles from May 2026 to the
most recent dump in August. It does a diff between the two CSVs to find puzzle IDs in the latest dump that aren't in the
previous dump and then fetches the game creation date. There could be puzzles in the latest dump from games that are
much earlier than August 2nd (when the latest dump was). Fetching creation date ensures that the games haven't been
trained on by the LLMs.

Without --baseline every row is dated via the API (~20k batches, many hours);
with it, only puzzles absent from the older dump are dated (~540 batches).

Usage:
    uv run python scripts/extract_post_cutoff_puzzles.py \
        --baseline ~/.cache/chess-coach-evals/lichess_db_puzzle.csv
    uv run python scripts/extract_post_cutoff_puzzles.py --limit-batches 5  # smoke test
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import requests

BATCH_ENDPOINT = (
    "https://lichess.org/api/games/export/_ids"
    "?moves=false&clocks=false&evals=false&opening=false"
)
USER_AGENT = "chess-coach-eval dataset builder (github.com/jbmlaird/chess-coach)"
GAME_ID_RE = re.compile(r"lichess\.org/([A-Za-z0-9]{8})")

BATCH_SIZE = 300
TOTAL_ROWS = 6_100_960  # rows in the 2026-08 dump; feeds the ETA display only


class BadResponse(requests.RequestException):
    """A 2xx-ish response we can't use (redirect, empty body, non-JSON)."""


def parse_game_id(game_url: str) -> str | None:
    match = GAME_ID_RE.search(game_url)
    return match.group(1) if match else None


def fetch_game_dates(session: requests.Session, game_ids: list[str]) -> dict[str, date]:
    unique_ids = list(dict.fromkeys(game_ids))
    dates: dict[str, date] = {}
    # allow_redirects=False: a redirected POST silently becomes a GET and the
    # body (our ID list) is dropped; better to see the 3xx and retry.
    with session.post(BATCH_ENDPOINT, data=",".join(unique_ids), stream=True,
                      timeout=(10, 30), allow_redirects=False) as response:
        response.raise_for_status()
        if response.status_code != 200:  # raise_for_status passes 3xx through
            raise BadResponse(f"unexpected HTTP {response.status_code}")
        try:
            for line in response.iter_lines():
                if line:
                    game = json.loads(line)
                    played = datetime.fromtimestamp(game["createdAt"] / 1000, tz=timezone.utc)
                    dates[game["id"]] = played.date()
        except (json.JSONDecodeError, KeyError) as error:
            raise BadResponse(f"unparseable response body: {error}") from error
    if unique_ids and not dates:
        raise BadResponse("empty response for a non-empty batch")
    return dates


def fetch_game_dates_with_retries(
        session: requests.Session, game_ids: list[str],
        max_attempts: int = 8, max_rate_limits: int = 10,
) -> dict[str, date]:
    attempts = 0
    rate_limits = 0
    while True:
        try:
            return fetch_game_dates(session, game_ids)
        except requests.RequestException as error:
            is_rate_limit = (
                isinstance(error, requests.HTTPError)
                and error.response is not None
                and error.response.status_code == 429
            )
            if is_rate_limit:
                # 429s don't consume the transient-error budget, but are capped
                # so a hard rate-limit can't loop forever.
                rate_limits += 1
                if rate_limits >= max_rate_limits:
                    raise RuntimeError(f"Rate limited {rate_limits} times; giving up.")
                # Lichess asks clients to stop for a full minute after a 429.
                print("Rate limited (429); sleeping 60s...", flush=True)
                time.sleep(60)
            else:
                attempts += 1
                if attempts >= max_attempts:
                    raise RuntimeError(
                        f"Batch failed after {max_attempts} attempts ({error}); "
                        f"checkpoint holds the last completed batch."
                    )
                backoff = min(5 * attempts, 60)
                print(f"{error} (attempt {attempts}/{max_attempts}); sleeping {backoff}s...", flush=True)
                time.sleep(backoff)


def run_config(args: argparse.Namespace) -> dict:
    """The settings that determine the row-to-output mapping; a resume under a
    different config would silently corrupt the dataset, so these are stamped
    into the checkpoint and must match on resume. (--delay and --limit-batches
    deliberately excluded: they change pacing, not output.) File sizes catch a
    re-downloaded dump at the same path."""
    return {
        "input": str(args.input.resolve()),
        "input_size": args.input.stat().st_size,
        "baseline": str(args.baseline.resolve()) if args.baseline else None,
        "baseline_size": args.baseline.stat().st_size if args.baseline else None,
        "cutoff": args.cutoff.isoformat(),
        "output": str(args.output.resolve()),
    }


def load_checkpoint(path: Path, config: dict) -> dict:
    if not path.exists():
        return {"rows_processed": 0, "config": config}
    try:
        checkpoint = json.loads(path.read_text())
    except json.JSONDecodeError:
        sys.exit(f"Checkpoint {path} is corrupt. Delete it and the output file to start fresh.")
    if checkpoint.get("config") != config:
        sys.exit(
            f"Checkpoint {path} was written by a run with different settings.\n"
            f"  checkpoint: {checkpoint.get('config')}\n"
            f"  current:    {config}\n"
            f"Delete the checkpoint (and the output file) to start fresh, or rerun with matching flags."
        )
    return checkpoint


def save_checkpoint(path: Path, rows_processed: int, config: dict) -> None:
    """Atomic: a crash mid-write must never leave a truncated checkpoint."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"rows_processed": rows_processed, "config": config}))
    os.replace(tmp, path)


def load_written_ids(path: Path) -> set[str]:
    """PuzzleIds already in the output. Rows are written before the checkpoint
    advances, so a crash in that window re-reads the same rows on resume; this
    set makes the re-write a no-op instead of a duplicate."""
    with open(path, newline="") as f:
        return {row["PuzzleId"] for row in csv.DictReader(f)}


def load_baseline_ids(path: Path) -> set[str]:
    with open(path, newline="") as f:
        return {row["PuzzleId"] for row in csv.DictReader(f)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path.home() / "Downloads" / "lichess_db_puzzle.csv")
    parser.add_argument("--output", type=Path, default=Path("post_cutoff_puzzles.csv"))
    parser.add_argument("--checkpoint", type=Path, default=Path("post_cutoff_checkpoint.json"))
    parser.add_argument("--baseline", type=Path, default=None,
                        help="older puzzle dump; only puzzles absent from it are dated. "
                             "Omitting this dates ALL rows: ~20k API batches, many hours")
    parser.add_argument("--cutoff", type=date.fromisoformat, default=date(2026, 6, 1),
                        help="keep puzzles whose game was played on/after this date (default: 2026-06-01)")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="seconds to sleep between API calls (default: 0.5)")
    parser.add_argument("--limit-batches", type=int, default=None,
                        help="stop after this many batches (for smoke testing)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = run_config(args)
    checkpoint = load_checkpoint(args.checkpoint, config)
    rows_done = checkpoint["rows_processed"]
    resume_start = rows_done

    written_ids: set[str] = set()
    if resume_start > 0:
        if not args.output.exists():
            sys.exit(f"Checkpoint says {resume_start} rows done but {args.output} is missing; "
                     f"delete {args.checkpoint} to start fresh.")
        written_ids = load_written_ids(args.output)
    elif args.output.exists() and args.output.stat().st_size > 0:
        sys.exit(f"{args.output} already exists and there is no checkpoint; "
                 f"delete it (or pass a different --output) to start fresh.")
    kept = len(written_ids)

    baseline_ids: set[str] = set()
    if args.baseline:
        baseline_ids = load_baseline_ids(args.baseline)
        print(f"Diff mode: {len(baseline_ids)} baseline PuzzleIds loaded; only new puzzles will be dated")

    session = requests.Session()
    session.headers.update({"Accept": "application/x-ndjson", "User-Agent": USER_AGENT})

    unresolved = 0  # games the API didn't return (deleted/unavailable)
    with session, open(args.input, newline="") as infile:
        reader = csv.DictReader(infile)
        with open(args.output, "a" if resume_start > 0 else "w", newline="") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames + ["GameDate"],
                                    extrasaction="ignore")
            if resume_start == 0:
                writer.writeheader()
            else:
                print(f"Resuming: skipping {resume_start} already-processed rows ({kept} kept so far)")
                for _ in range(resume_start):
                    next(reader, None)

            batch: list[tuple[dict, str]] = []  # (row, game_id)
            batches_done = 0
            started = time.monotonic()

            def flush() -> None:
                nonlocal kept, batches_done, unresolved
                dates = fetch_game_dates_with_retries(session, [gid for _, gid in batch])
                for row, gid in batch:
                    played = dates.get(gid)
                    if played is None:
                        unresolved += 1
                        continue
                    if played >= args.cutoff and row["PuzzleId"] not in written_ids:
                        row["GameDate"] = played.isoformat()
                        writer.writerow(row)
                        written_ids.add(row["PuzzleId"])
                        kept += 1
                outfile.flush()
                save_checkpoint(args.checkpoint, rows_done, config)
                batch.clear()
                batches_done += 1
                if batches_done % 25 == 0:
                    # throughput in rows/sec, not batches/sec: in diff mode each
                    # batch of 300 candidates spans thousands of skipped rows
                    rows_per_sec = (rows_done - resume_start) / (time.monotonic() - started)
                    remaining = (TOTAL_ROWS - rows_done) / rows_per_sec / 60
                    print(f"{rows_done} rows, {kept} kept, {unresolved} unresolved, "
                          f"~{remaining:.0f}min remaining", flush=True)
                time.sleep(args.delay)

            for row in reader:
                rows_done += 1
                if baseline_ids and row["PuzzleId"] in baseline_ids:
                    continue  # already existed in the older dump; can't be a post-cutoff puzzle
                game_id = parse_game_id(row.get("GameUrl") or "")
                if game_id is not None:
                    batch.append((row, game_id))
                if len(batch) == BATCH_SIZE:
                    flush()
                    if args.limit_batches and batches_done >= args.limit_batches:
                        print(f"Stopping at --limit-batches {args.limit_batches}")
                        break
            if batch:
                flush()

    print(f"Done: {rows_done} rows processed, {kept} puzzles kept, "
          f"{unresolved} unresolved games dropped -> {args.output}")


if __name__ == "__main__":
    main()
