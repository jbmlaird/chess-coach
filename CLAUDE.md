# chess-coach eval

LLM chess-coaching eval: frozen 250-puzzle golden set, Inspect AI harness,
Stockfish-graded quality metrics. Every published number must be regenerable
from committed artifacts by a script.

## Chess reasoning: use the engine, never your head

All chess analysis goes through the engine - the Stockfish MCP tool
(`stockfish_mcp.py`, registered in `.mcp.json`) or `engine.py` directly
(`uv run python`, `Engine.grader()`):

- Never predict best moves, evaluate positions, count mate distances, or
  judge move legality by hand. Hand analysis in this repo's history produced
  confident wrong lines; the pinned engine is the authority.
- Any chess claim in a README, docstring, test fixture, or review finding
  must be engine-verified (or python-chess-verified for pure legality) before
  it is written down.
- python-chess is fine for board mechanics (legality, FEN handling, replay).
  Judgment calls (better/worse/winning/hanging) are the engine's alone.

## Frozen artifacts — never edit in place

- `golden/vN/golden_candidates.csv` — sha256-frozen (hash in its own
  `golden_candidates.meta.json`). Versions are never edited or deleted: a log
  records the dataset path it ran on (logs from before `golden/` was versioned
  say `golden_candidates.csv`, which is v1) and regrades with
  `--golden golden/vN/golden_engine.csv`
- `golden/vN/golden_engine.csv` + meta — certified reference evals;
  regenerating is a deliberate re-certification event, not a fix
- Replacing rows means a new `golden/vN/` via `sample_golden.py --replace`
  (blunder rows must clear `NOISE_FLOOR_PP`), never an edit to an existing one
- `logs/**/*.eval` — committed runs backing published tables
- `vendor/lichess_puzzler/` — byte-identical to the pinned upstream commit

If a frozen file must change: change it, re-stamp its meta (new sha, date,
reason), and say so in the PR. Silent drift is the failure mode all the
sha-handshakes exist to catch.

## Paid eval runs — the pilot protocol (non-negotiable)

1. `--model mockllm/model` first: plumbing and scorers, free.
2. 8-sample paid pilot spanning both arms (`--sample-id`), logged to a scratch
   `--log-dir`, never to `logs/`.
3. Read the pilot log before fanning out: rendered prompt (no literal
   `{placeholders}`), completions parse, stop reasons, measured tokens.
4. Project cost = measured pilot tokens × published prices. Never project from
   guessed tokens or an incomplete pilot (censored pilots bias low ~2x).
   State the projection and get approval before the full run.
5. Reasoning models: thinking shares `max_tokens` (default 32k). Watch for
   `stop_reason: max_tokens` with empty output — billed and unusable.

History that motivates this: a placeholder bug burned ~$3 across 750 calls; a
censored pilot turned a "$53" Opus run into $109.52.

## Instrument discipline

- The prompt, parsers, and scorers are the ruler. Any change to them bumps
  `Task(version=N)` in `move_review.py` and starts a new results column —
  never compare across instrument versions without saying so.
- Strict parsing is the contract: format noncompliance scores as failure.
  Never make the move/verdict parsers charitable (prose scanning was tried,
  rejected: chess prose is full of square names that parse as moves).
- Engine analyses for measurement use `Engine.grader()` (Threads=1, fixed
  nodes, fresh `ucinewgame` per call — results are byte-reproducible). The
  model-facing MCP tool runs the same `Engine.grader()` preset (a test pins
  the grader's provenance to every `golden/*/golden_engine.meta.json`);
  tool-arm runs record the tool's `engine.provenance` in their sidecar like
  any other engine run. A weaker tool, if ever wanted, gets a new preset -
  the grader never changes.
- Damage/quality aggregation happens in win% space (Lichess model, ±1000cp
  clamp), never by averaging raw centipawns (±10000 mate sentinel).

## Numbers and the README

- Results tables are never typed. `scripts/render_results.py` renders every
  table from the committed logs (via `calculate_metrics.metrics()` and
  `grade_logs.grade()`); paste its output verbatim under the heading it
  declares. `tests/test_render_results.py` fails on any differing row (the two
  damage rows need the engine and a warm cache, so CI checks the rest). A new
  column is a new entry in `TABLES`, never a hand-typed cell — hand-transcription
  produced transposed cells twice and let a wrong token budget survive three
  tables. Numbers quoted in prose come from a printed script line that a test
  pins (`tests/test_calculate_metrics.py`).
- Table headings carry all three dimensions: instrument version, golden
  version, arm (`Instrument v3 · golden v2 · no tools`). Logs live at
  `logs/golden-v<N>/<model>[-thinking|-no-thinking][-tool-optional|-tool-required]/`,
  one committed run per directory; the golden version a run graded against is
  read from that path.
- Quality stats condition on legal answers only: always print/quote the
  per-arm denominator (`n=63/200`). Never headline the blended blunder-arm
  damage (it mixes detection failures with suggestion quality and inverts
  model rankings).
- Grounded (tool-arm) columns are published only with the tool-aware metrics
  beside them (tool errors, called-before-answer, relay fidelity, frame
  errors): on a tool arm `ground_truth` measures whether the model copied the
  engine, not whether it judged the position.

## Training data (rule stated before any training exists)

- The golden set never appears in training data, in any version: every
  training set is checked against `golden/*/golden_candidates.csv` by puzzle
  ID and by FEN before use, and that check runs in CI once a training set is
  committed.

## Environment

- `uv` for everything: `uv run pytest -q`, `uv run inspect eval move_review.py`,
  `uv run inspect view`, `uv run python scripts/...`
- Stockfish 18 via Homebrew; override with `STOCKFISH_PATH` env var.
- API keys in `.env` (never committed, never printed). Third-party keys need
  the owner's explicit OK before any paid call.
- `graded_moves_cache.csv` is a disposable cache (gitignored) with an engine
  provenance header — delete it freely, never commit it.
