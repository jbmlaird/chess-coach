# chess-coach eval

LLM chess-coaching eval: frozen 250-puzzle golden set, Inspect AI harness,
Stockfish-graded quality metrics. Every published number must be regenerable
from committed artifacts by a script.

## Chess reasoning: use the engine, never your head

All chess analysis goes through the engine - the Stockfish MCP tool once it
exists, `engine.py` until then (`uv run python`, `Engine.grader()`):

- Never predict best moves, evaluate positions, count mate distances, or
  judge move legality by hand. Hand analysis in this repo's history produced
  confident wrong lines; the pinned engine is the authority.
- Any chess claim in a README, docstring, test fixture, or review finding
  must be engine-verified (or python-chess-verified for pure legality) before
  it is written down.
- python-chess is fine for board mechanics (legality, FEN handling, replay).
  Judgment calls (better/worse/winning/hanging) are the engine's alone.

## Frozen artifacts — never edit in place

- `golden_candidates.csv` — sha256-frozen (hash in `golden_candidates.meta.json`)
- `golden_engine.csv` + meta — certified reference evals; regenerating is a
  deliberate re-certification event, not a fix
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
  future model-facing tool gets its own preset; grader settings never change
  to suit the tool. Record `engine.provenance` in every meta sidecar.
- Damage/quality aggregation happens in win% space (Lichess model, ±1000cp
  clamp), never by averaging raw centipawns (±10000 mate sentinel).

## Numbers and the README

- Every table cell must come from a script (`scripts/calculate_metrics.py`,
  `scripts/grade_logs.py`) run against committed logs — hand-transcription has
  produced transposed cells twice; verify cells against script output after
  editing.
- Quality stats condition on legal answers only: always print/quote the
  per-arm denominator (`n=63/200`). Never headline the blended blunder-arm
  damage (it mixes detection failures with suggestion quality and inverts
  model rankings).

## Environment

- `uv` for everything: `uv run pytest -q`, `uv run inspect eval move_review.py`,
  `uv run inspect view`, `uv run python scripts/...`
- Stockfish 18 via Homebrew; override with `STOCKFISH_PATH` env var.
- API keys in `.env` (never committed, never printed). Third-party keys need
  the owner's explicit OK before any paid call.
- `graded_moves_cache.csv` is a disposable cache (gitignored) with an engine
  provenance header — delete it freely, never commit it.
