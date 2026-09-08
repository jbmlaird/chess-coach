"""The golden set's own invariants: every version's sha handshake holds, v2 differs
from v1 by exactly the replacements its meta records, every certified blunder row
clears the engine noise floor, and the tooling that makes a version refuses drift."""

import csv
import hashlib
import json
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import chess
import pytest

import sample_golden
from engine import NOISE_FLOOR_PP, move_damage_pp

REPO = Path(__file__).parent.parent
GOLDEN = REPO / "golden"
VERSIONS = sorted(GOLDEN.glob("v*"))


def rows(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def meta(path: Path) -> dict:
    return json.loads(path.read_text())


REPLACED = meta(GOLDEN / "v2" / "golden_candidates.meta.json")["replaced"]


@pytest.mark.parametrize("version", VERSIONS, ids=lambda p: p.name)
def test_frozen_csv_matches_both_metas(version):
    sha = hashlib.sha256((version / "golden_candidates.csv").read_bytes()).hexdigest()
    assert sha == meta(version / "golden_candidates.meta.json")["csv_sha256"]
    assert sha == meta(version / "golden_engine.meta.json")["input_sha256"]


def test_v2_replaces_exactly_the_recorded_rows():
    v1 = rows(GOLDEN / "v1" / "golden_candidates.csv")
    v2 = rows(GOLDEN / "v2" / "golden_candidates.csv")
    assert len(v1) == len(v2) == 250
    changed = [(a, b) for a, b in zip(v1, v2) if a != b]
    assert sorted(a["PuzzleId"] for a, _ in changed) == sorted(REPLACED["removed"])
    assert sorted(b["PuzzleId"] for _, b in changed) == sorted(REPLACED["added"])
    for a, b in changed:  # same stratum, and reviewed
        assert (a["Arm"], a["Category"], a["Band"]) == (b["Arm"], b["Category"], b["Band"])
        assert b["Review"].startswith("ok")


def test_blunder_rows_clear_the_noise_floor():
    for row in rows(GOLDEN / "v2" / "golden_engine.csv"):
        if row["Arm"] == "blunder":
            damage = move_damage_pp(int(row["BeforeScoreCentipawns"]), int(row["AfterPlayedScoreCentipawns"]))
            assert damage > NOISE_FLOOR_PP, (row["PuzzleId"], damage)


def test_unchanged_rows_certify_identically():
    v1 = {r["PuzzleId"]: r for r in rows(GOLDEN / "v1" / "golden_engine.csv")}
    shared = [r for r in rows(GOLDEN / "v2" / "golden_engine.csv") if r["PuzzleId"] in v1]
    assert len(shared) == 250 - len(REPLACED["removed"])
    assert all(r == v1[r["PuzzleId"]] for r in shared)


def test_certify_refuses_an_unfrozen_csv(tmp_path):
    csv_path = tmp_path / "golden_candidates.csv"
    csv_path.write_bytes((GOLDEN / "v2" / "golden_candidates.csv").read_bytes() + b"\r\n")
    (tmp_path / "golden_candidates.meta.json").write_text(json.dumps({"csv_sha256": "stale"}))
    run = subprocess.run([sys.executable, "scripts/certify_golden.py", "--golden", str(csv_path)],
                         cwd=REPO, capture_output=True, text=True, timeout=60)
    assert run.returncode == 1 and "not frozen" in run.stderr
    assert not (tmp_path / "golden_engine.csv").exists()


def test_replace_redraws_from_the_same_cell_and_records_rejections(monkeypatch):
    def puzzle(pid):
        return {"PuzzleId": pid, "FEN": chess.STARTING_FEN, "Moves": "e2e4 e7e5", "Category": "mate",
                "Band": "<1200", "Rating": "1000", "RatingDeviation": "80", "GeneratedThemes": "mate", "GameUrl": ""}
    keep, old = ({**sample_golden.eval_item(puzzle(p), "blunder"), "Review": "ok"} for p in ("keep", "old"))
    cells = {("mate", "<1200"): [puzzle(p) for p in ("keep", "old", "bad", "good")]}
    alphabetical = SimpleNamespace(sample=lambda pool, k: sorted(pool, key=lambda r: r["PuzzleId"]))
    monkeypatch.setattr(sample_golden.Engine, "grader", staticmethod(nullcontext))
    monkeypatch.setattr(sample_golden, "noise_floor_rejection",
                        lambda item, engine: "too weak" if item["PuzzleId"] == "bad" else None)

    table = [keep, old]
    block = sample_golden.replace(table, ["old"], cells, alphabetical)

    assert block["added"] == ["good"] and block["rejected"] == [{"puzzle_id": "bad", "reason": "too weak"}]
    assert table[0] is keep and table[1]["PuzzleId"] == "good" and table[1]["Review"] == ""
    assert list(table[1]) == list(old)  # same columns
    with pytest.raises(SystemExit):
        sample_golden.replace([keep, old], ["old", "old"], cells, alphabetical)
