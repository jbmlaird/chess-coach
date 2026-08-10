"""Tag the post-cutoff puzzles with Lichess's own (vendored) tagger.

Runs vendor/lichess_puzzler/cook.py - pinned at the commit recorded in
vendor/lichess_puzzler/VENDOR_INFO.json, imported through the lichess_tagger
adapter - over post_cutoff_puzzles.csv and writes a sidecar CSV of generated
themes keyed by PuzzleId. The golden source file is never modified.

Two things the generated themes deliberately lack:
- cp-derived outcome tags (equality/advantage/crushing): the dump carries no
  engine eval, so the tagger's cp field is stubbed to 0 and those tags are
  dropped from the output as meaningless.
- phase tags (opening/middlegame/endgame): cook.py never emits them, they come
  from lila. The post-cutoff rows already carry phase and length themes from
  Lichess, so nothing is lost.

A .meta.json is written next to the output recording the tagger commit, the
input file identity, and run settings, so the labels stay reproducible.

Usage:
    uv run python scripts/tag_post_cutoff.py
"""

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))
from lichess_tagger import CP_DERIVED, VENDOR, generated_themes  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path("post_cutoff_puzzles.csv"))
    parser.add_argument("--output", type=Path, default=Path("post_cutoff_themes.csv"))
    args = parser.parse_args()
    if args.output.resolve() == args.input.resolve():
        sys.exit(f"--output {args.output} is the same file as --input; refusing to overwrite the source")

    failures: list[tuple[str, str]] = []
    tagged = 0
    with open(args.input, newline="") as infile, open(args.output, "w", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["PuzzleId", "GeneratedThemes"])
        for row in csv.DictReader(infile):
            puzzle_id = row["PuzzleId"]
            try:
                themes = generated_themes(puzzle_id, row["FEN"], row["Moves"])
            except Exception as error:  # report all failures at the end, loudly
                failures.append((puzzle_id, repr(error)))
                continue
            writer.writerow([puzzle_id, " ".join(themes)])
            tagged += 1

    if tagged == 0:
        sys.exit(f"no rows tagged from {args.input} - empty or wrong input file")

    vendor_info = json.loads((VENDOR / "VENDOR_INFO.json").read_text())
    meta = {
        "tagger_source": vendor_info["source"],
        "tagger_commit": vendor_info["commit"],
        "generated": date.today().isoformat(),
        "input": args.input.name,
        "input_size": args.input.stat().st_size,
        "cp_stubbed_to_zero": True,
        "dropped_tags": sorted(CP_DERIVED),
        "rows_tagged": tagged,
        "rows_failed": len(failures),
    }
    args.output.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(f"{tagged} rows tagged -> {args.output}")
    if failures:
        print(f"{len(failures)} rows FAILED (sidecar is incomplete):")
        for puzzle_id, error in failures[:20]:
            print(f"  {puzzle_id}: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
