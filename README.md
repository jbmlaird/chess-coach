# Chess Coach Eval

An evaluation harness that measures how well LLMs coach chess, built so that every published number regenerates
from a committed artifact by a script. A model is shown a position and the move a student played, and must say
whether it was a blunder or the best move, name the punishing reply if blunder, and suggest the best move. Every
answer is scored against Stockfish, never against the model's own chess judgement.

**What it's found so far**:

- Ungrounded models detect blunders but cannot verify them. They call 84-91% of real blunders a blunder, but also
  call two thirds of engine-best moves a blunder. When they're right about a blunder they name the certified
  refutation only 8.8% (Haiku) and 22.6% (Sonnet) of the time. A frontier model with adaptive thinking reaches
  73.7% on the same question at $110 a run instead of $1-5 (golden v1).
- The "better move" they suggest instead is usually worse than the blunder: on blunder positions the suggested
  improvement gives away 37-40 win-probability points on average and only a quarter of suggestions are within engine
  noise of the best move. Between 28% and 45% of suggested moves are not even legal.
- Grounding fixes both. Given Stockfish as a tool the same models reach 87.6% (Haiku) and 100.0% (Sonnet, 250/250)
  verdict + refutation accuracy with the tool silently offered, for $3.59-6.90 a run across the three tool arms, and
  their suggested improvements sit within engine noise of best play 97-100% of the time because they relay the engine's
  move (fidelity 96-100%, no frame errors on any sample that queried the position after the move). What remains is tool
  misuse: the engine rejected at least one of Haiku's calls on 27-43% of blunder rows, Haiku hit the eight-turn limit
  on up to 15% of samples, and the wording that invited unlimited calls was the worst arm for Haiku and the costliest
  for both.
- Evaluation noise is measured, not assumed. Re-running an identical configuration nineteen days apart moved
  Haiku's legality rate by 8.6 points and flipped its verdict on 103 of 245 shared positions; the paired standard
  error is about 3-4 points, so single-run differences under 8 points are noise here.
- A notation contract exposes a model habit: under a UCI prompt, Haiku writes lowercase piece letters (`rd1`) on
  12% of answers, which the strict parser rejects by design
- Long-thinking runs are a bad trade at this budget: Sonnet 4.6 with thinking spent its whole 42k-token budget and
  returned nothing on 45% of positions, for 23 times the cost of the same model without thinking.

![Grounding on golden v2: for Haiku 4.5 and Sonnet 4.6, verdict + refutation accuracy, substantiation, blunder recall, best-move recall and legal best-move rate with no tools against the same prompt with the Stockfish tool silently offered; instrument v3, thinking off](charts/golden-v2-grounding.svg)

The chart pairs each no-tool baseline with the `silent` tool arm, whose prompt is byte-identical. The request differs
only by the tool definition and the eight-turn limit it needs, and the runs are six days apart, so each gap is the
engine plus the run-to-run noise measured under the v3 table: Haiku's blunder recall (90.5 to 92.5) sits inside that
band, every other bar clears it by 16 points or more. It is drawn by [plot_results.py](scripts/plot_results.py) from
the same numbers the tables are rendered from; `uv run python scripts/plot_results.py` regenerates it byte-for-byte,
and CI fails if the committed SVG differs.

**How it's built**

- Dataset: 250 Lichess puzzles played after the models' training cutoffs, stratified over 10 tactical motifs and 4
  rating bands, tagged by Lichess's own vendored tagger, hand-reviewed, then certified by a pinned Stockfish 18 and
  frozen with a sha256 handshake (`golden/v2/`). Five rows that failed the damage-floor rule (four already-lost
  positions and one 3pp blunder) were replaced and the whole set re-certified; both versions are kept so every
  old table still regenerates.
- Harness: [Inspect AI](https://inspect.aisi.org.uk) task in [move_review.py](move_review.py) with two scorers,
  legality and ground truth, and a strict output contract. The prompt, parsers and scorers are the instrument; any
  change bumps `Task(version=N)` and starts a new results column.
- Logs: every run is committed under `logs/` and published as the Inspect log viewer at
  https://jbmlaird.github.io/chess-coach/ by [a workflow](.github/workflows/pages.yml) that rebuilds it from the committed logs on
  every merge; the sample links in this README point into it.
- Grading: [certify_golden.py](scripts/certify_golden.py) freezes the reference evals,
  [grade_logs.py](scripts/grade_logs.py) scores suggested moves in Lichess win-probability space,
  [calculate_metrics.py](scripts/calculate_metrics.py) derives every table cell,
  [render_results.py](scripts/render_results.py) renders the tables below from those numbers, and
  [tests](tests/test_render_results.py) fail if any row of a README table differs from the rendered one (the two damage
  rows need the engine, so CI checks the rest).
- Spend discipline: mock run, then an 8-sample paid pilot whose measured tokens project the full cost, then approval.
  Full runs cost $1 (Haiku) to $5 (Sonnet); the three thinking and frontier runs that cost $100-110 each are in the
  tables as the reason the protocol exists.

**Answered:** yes, and past them. Grounded in the engine with the tool silently offered, Sonnet 4.6 scores 100.0% and
Haiku 4.5 87.6% on verdict + refutation for $3.68-5.75 a run, against 34.8% for the frontier model with adaptive
thinking and no tool at $109.52 (golden v1 and instrument v2, the only set it ran on; the prompt is byte-identical to
v3, so this compares across sets, and its recall and precision are in the second table). Substantiation goes from 73.7%
frontier to 91.9-100% grounded, legal suggestions from 82.8% to 95.6-100%. The next column is a self-hosted open
model, a chess-post-trained Qwen2.5-7B served with vLLM, first without the tool and then with it.

## Background

This is a rebuild of an AI agent repo I have where I'm rewriting the evaluation logic by hand to get a solid
understanding.

### Dataset selection

To try and avoid training memorisation of chess positions, puzzles are selected after the tested model's cutoff dates.
Anthropic's can be found [here](https://support.claude.com/en/articles/8114494-how-up-to-date-is-claude-s-training-data)
(with Opus 5 being May 2026) and OpenAI's [here](https://developers.openai.com/api/docs/models) (Astra being April
2026).

Of course, canonical positions that are common (such as endgames) likely will have been seen before, but this attempts
to avoid the number of seen-before positions, mainly in the middlegame where there are more pieces and more variation in
board setup.

To build this dataset I used the current DB dump found on https://database.lichess.org/#puzzles that was last updated on
2nd August. Since there is only a PuzzleDate in this dump but no `PlayedDate` (puzzles from games months ago can be
added to the archive) I compared the puzzles in this dump with puzzles in the dump from May taking only the new puzzles.
I then took those new puzzles and got the `PlayedDate` from the Lichess API, see how
in [extract_post_cutoff_puzzles.py](/scripts/extract_post_cutoff_puzzles.py).

Newly added puzzles are not yet motif-tagged by Lichess (phase and length themes are present, tactic themes aren't) so
[motif_detector.py](/motif_detector.py) was originally created to tag puzzles for manual review to be part of the
golden dataset. Lichess' own puzzle tagger
is [cook.py](https://github.com/ornicar/lichess-puzzler/blob/8d9faff694ba3a8598abc5465347209af3f90a82/tagger/cook.py)
which I've vendored to tag the outstanding puzzles, rather than handroll myself. It's copied verbatim into
[vendor/lichess_puzzler](/vendor/lichess_puzzler).

[tag_post_cutoff.py](/scripts/tag_post_cutoff.py) runs it over every post-cutoff puzzle and writes the labels to a
sidecar [post_cutoff_themes.csv](/post_cutoff_themes.csv).

A golden dataset of 250 puzzles was sampled and written to `golden/v1/golden_candidates.csv`. This has a mixture of
themes (such
as `fork`, `skewer`, `pin`, `attraction`, `deflection` etc.) at a variety of rating levels to have a varied dataset. 200
of the puzzles play moves that are blunders, 50 puzzles where the best move was played. This dataset was automatically
tagged by the vendored `lichess_puzzler` and verified myself by hand. 250 rows had correct tags. Of those 250, 7 felt
as if they were missing tags:

* [lD0lt](https://lichess.org/training/lD0lt) & [0F0X2](https://lichess.org/training/0F0X2) - only absolute pins (pins
  against the King) are detected, relative pins are invisible
* [oNdUC](https://lichess.org/training/oNdUC) - pawns are never counted as fork *targets* (a pawn may deliver a fork,
  but forked pawns don't count). This is meaningful in the endgame
* [jU61o](https://lichess.org/training/jU61o) - Rxc6 decoys the rook *off* its back-rank duty into a fork; the template
  requires the attracted square itself to be re-attacked and captured on. The fallback tag `capturingDefender` is also
  blocked: Kf8 "defends" e7 geometrically, but Kxe7 is illegal (Nd5 covers e7)
* [cNPtu](https://lichess.org/training/cNPtu) - deflection missing because the rooks are traded. When the deflecting
  move is a capture, deflection requires it to be a net sacrifice - equal trades are excluded
* [pJzBI](https://lichess.org/training/pJzBI) - `exposedKing`. Two blocks: its check-scan window drops the first and
  last solver moves, which on a 4-ply puzzle is all of them (~65% of the corpus can never receive this tag), and its
  shelter test is a 5-square occupancy box around the King - the blunder moved the c-pawn *within* that box, opening
  the rank the rook attacks down while still counting as shelter
* [BaYNY](https://lichess.org/training/BaYNY) - this tactic has no codified theme, but it uses the player's King to box
  in the other King

## Engine certification of the golden set

Before grading anything against Stockfish, I certified the dataset itself:
[certify_golden.py](scripts/certify_golden.py) runs a pinned engine (Stockfish 18, 1M nodes/position) over every golden
row and freezes the reference evals in [golden_engine.csv](golden/v1/golden_engine.csv). The engine verified the labels
almost
completely: on all 50 best-arm rows its best move _is_ the played move, and on all 200 blunder rows its best reply _is_
the certified refutation

...with one exception: on 4 blunder rows the player was already in a forced mate before the "blunder", so every legal
move lost identically. This traces to how Lichess generates puzzles - the "setup move must have thrown the game away"
gate only applies to advantage puzzles; mate puzzles certify the mating line with no requirement that the position was
savable. A model answering `best` on those rows is engine-correct but scored wrong (at most ~2pp effect on blunder-arm
accuracy, and since every model over-calls `blunder`, these rows mildly reward that bias). These 4 puzzles should be
replaced when selecting a new dataset.

### Golden v2

Those four rows were replaced, plus a fifth, under the rule the v1 audit made obvious: a blunder row's certified damage
must exceed the 5pp engine noise floor (the same bar [grade_logs.py](scripts/grade_logs.py) uses for "within noise of
best"). The four already-lost rows have 0pp by definition, and `Wj3vh` (3.0pp: a lost position made lost faster) fails
it too. Each was redrawn from its own category x band cell by [sample_golden.py](scripts/sample_golden.py) `--replace`,
which engine-checks every draw: 4 of the 9 candidates it tried were rejected for the same two reasons (three already
mated, one at 1.9pp), so the rule is not cosmetic. Removed: f16cc, 2NC6V, Nx8fl, FHpic, Wj3vh. Added: WmRuu, DDrC0,
xmHxa, ZaVDg, CgxAk (all `mate`, bands <1200 and 1200-1599). Both sets live under `golden/v1/` and `golden/v2/` with
their own certification, so every earlier table still regenerates against v1, and [test_golden.py](tests/test_golden.py)
pins that v2 differs from v1 by exactly these rows, that the 245 shared rows certify byte-identically, and that every
blunder row clears the floor.

## Results

### Instrument v0 · golden v1 · no tools · SAN prompt (2026-08-18/19)

|                                                                                                   | Haiku 4.5         | Sonnet 4.6         |
|---------------------------------------------------------------------------------------------------|-------------------|--------------------|
| Verdict + refutation accuracy (overall)                                                           | **18.4%**         | 16.8%              |
| - blunder arm / best arm                                                                          | 11.0% / 48.0%     | 13.0% / 32.0%      |
| Blunder class - recall (caught real blunders)                                                     | **93.0%**         | 86.5%              |
| Blunder class - precision (calls that were right)                                                 | **87.7%**         | 83.6%              |
| Best class - recall (endorsed real best moves)                                                    | **48.0%**         | 32.0%              |
| Best class - precision (endorsements right)                                                       | **63.2%**         | 37.2%              |
| Best class - mean damage vs best play of suggested "improvements" (excludes correct endorsements) | **62.0pp (n=14)** | 64.3pp (n=24)      |
| Substantiation (correct blunder calls backing the certified refutation)                           | 11.8%             | **15.0%**          |
| Unplayable refutations (illegal, invalid or ambiguous)                                            | 84/200            | **46/200**         |
| Legal `BEST_MOVE` suggestions                                                                     | 58.0%             | **68.8%**          |
| Invalid `BEST_MOVE` answers (of which a lowercase piece letter)                                   | **0/250 (0)**     | **0/250 (0)**      |
| Suggested improvement damage vs best play (blunder arm, legal alternatives to the blunder)        | 42.6pp (n=93)     | **38.3pp (n=105)** |
| - of which within engine noise of best play (damage of 5pp or less)                               | 16/93 (17%)       | **21/105 (20%)**   |
| Unfinished samples (stop reason not `stop`)                                                       | **0**             | **0**              |
| Empty outputs (no text in the final generation)                                                   | **0**             | **0**              |
| Format failures (non-empty parse errors)                                                          | **0**             | **0**              |
| Measured cost (full run)                                                                          | **~$1.00**        | ~$4.04             |

Logs for the golden v1 tables live in `logs/golden-v1/`, viewable with `uv run inspect view` from the root. The accuracy
rows are pulled from the
log metadata, the damage rows come from [grade_logs.py](scripts/grade_logs.py) (defined below the next table), and the
rest of the metrics are calculated via the [calculate_metrics](scripts/calculate_metrics.py) script.

Both models love to call things a blunder (93%/86.5% recall blunder-class), with the best-arm class showing that it errs
on the side of calling best moves also a blunder (48%/32% recall best-class). The blunder-class recall would make it
seem like the model is performing great without the best-class stats.

When a model correctly calls a blunder, it can only name the punishing reply 11.8%/15.0% of the time.

After the first run against Haiku & Sonnet, I noticed that the "best move" suggested by the LLM are legal but bad. For
example, [`0F0X2`](https://jbmlaird.github.io/chess-coach/#/logs/golden-v1%2Fhaiku-4-5%2F2026-08-18T22-27-12-00-00_Positions_3ZCkX5hNDqYXoMaqmhWacC.eval/samples/sample/0F0X2/1) suggested best move `Qd4` which immediately hangs the queen. In [`nBP6h`](https://jbmlaird.github.io/chess-coach/#/logs/golden-v1%2Fhaiku-4-5%2F2026-08-18T22-27-12-00-00_Positions_3ZCkX5hNDqYXoMaqmhWacC.eval/samples/sample/nBP6h/1), the refutation line doesn't
show a move but instead shows `REFUTATION: The move wastes a chance; Black should have played Bxa1 to win White's
bishop. After Bb2, White continues but Black has missed the decisive material advantage.` not honouring the prompt.
[`fMkW1`](https://jbmlaird.github.io/chess-coach/#/logs/golden-v1%2Fhaiku-4-5%2F2026-08-18T22-27-12-00-00_Positions_3ZCkX5hNDqYXoMaqmhWacC.eval/samples/sample/fMkW1/1) suggested move `Rc6+` which was ambiguous as either rook could move to c6 and give check. To address this,
all moves need to be provided in UCI so this ambiguity can be removed.

Sample [3hzja](https://lichess.org/training/3hzja) (best arm) shows why we measure both legality and ground truth.
The played move is Kxc7 - king takes rook. Haiku got the best verdict right but restated the move as `Rxc7`,
misidentifying the capturing piece as the rook. Sonnet did the opposite by suggesting a perfectly legal move (`Ke5`)
attached to the wrong verdict. Board-state tracking and chess judgment fail independently; one scorer would hide half
the picture.

For both of the above cases, I decided
to [use UCI notation instead of SAN](https://github.com/jbmlaird/chess-coach/commit/78e80ae9f87d0ff912ad3ede3b72410de5089d76)
aligning with the standard that UCI was created for chess engines since knowledge of the piece isn't required.

### Instrument v2 · golden v1 · no tools (2026-08-20/21)

The two 2026-08-20 columns without thinking record `task_version 0`: they ran with this byte-identical prompt before the
version stamp landed. "Instrument v2" names the prompt, parsers and scorers, not the field in the log.

|                                                                                                   | Haiku 4.5     | Haiku 4.5 (thinking, 42k `max_tokens`) | Sonnet 4.6     | Sonnet 4.6 (thinking, 42k `max_tokens`) | Opus 5 (thinking disabled) | Opus 5 (adaptive thinking, 32k `max_tokens`) |
|---------------------------------------------------------------------------------------------------|---------------|----------------------------------------|----------------|-----------------------------------------|----------------------------|----------------------------------------------|
| Verdict + refutation accuracy (overall)                                                           | 14.4%         | 20.4%                                  | 21.6%          | 16.4%                                   | 31.2%                      | **34.8%**                                    |
| - blunder arm / best arm                                                                          | 10.5% / 30.0% | 13.5% / 48.0%                          | 13.5% / 54.0%  | 10.5% / 40.0%                           | 21.5% / 70.0%              | 28.0% / 62.0%                                |
| Blunder class - recall (caught real blunders)                                                     | **88.5%**     | 73.0%                                  | **88.5%**      | 37.5%                                   | 34.5%                      | 38.0%                                        |
| Blunder class - precision (calls that were right)                                                 | 83.5%         | 84.9%                                  | 88.5%          | 89.3%                                   | **90.8%**                  | 84.4%                                        |
| Best class - recall (endorsed real best moves)                                                    | 30.0%         | 48.0%                                  | 54.0%          | 40.0%                                   | **70.0%**                  | 62.0%                                        |
| Best class - precision (endorsements right)                                                       | 39.5%         | 30.8%                                  | **54.0%**      | 38.5%                                   | 28.5%                      | 24.8%                                        |
| Best class - mean damage vs best play of suggested "improvements" (excludes correct endorsements) | 67.4pp (n=17) | 61.6pp (n=16)                          | 72.7pp (n=13)  | 71.7pp (n=8)                            | **59.9pp (n=5)**           | 64.2pp (n=12)                                |
| Substantiation (correct blunder calls backing the certified refutation)                           | 11.9%         | 18.5%                                  | 15.3%          | 28.0%                                   | 62.3%                      | **73.7%**                                    |
| Unplayable refutations (illegal, invalid or ambiguous)                                            | 88/200        | 46/200                                 | 39/200         | 11/200                                  | **2/200**                  | 5/200                                        |
| Legal `BEST_MOVE` suggestions                                                                     | 47.2%         | 70.0%                                  | 72.0%          | 44.0% (80.9% of answered)               | 72.4% (95.8% of answered)  | **82.8% (96.3% of answered)**                |
| Invalid `BEST_MOVE` answers (of which a lowercase piece letter)                                   | 30/250 (30)   | **0/250 (0)**                          | **0/250 (0)**  | 1/250 (1)                               | 1/250 (0)                  | **0/250 (0)**                                |
| Suggested improvement damage vs best play (blunder arm, legal alternatives to the blunder)        | 38.6pp (n=63) | 41.6pp (n=81)                          | 38.5pp (n=117) | 31.1pp (n=50)                           | **13.1pp (n=56)**          | 20.3pp (n=70)                                |
| - of which within engine noise of best play (damage of 5pp or less)                               | 13/63 (21%)   | 14/81 (17%)                            | 23/117 (20%)   | 17/50 (34%)                             | **37/56 (66%)**            | 39/70 (56%)                                  |
| Unfinished samples (stop reason not `stop`)                                                       | **0**         | **0**                                  | **0**          | 115/250 (46.0%)                         | 29/250 (11.6%)             | 37/250 (14.8%)                               |
| Empty outputs (no text in the final generation)                                                   | **0**         | **0**                                  | **0**          | 113/250 (45.2%)                         | **0**                      | 35/250 (14.0%)                               |
| Format failures (non-empty parse errors)                                                          | **0**         | **0**                                  | **0**          | 1/250 (0.4%)                            | 59/250 (23.6%)             | **0**                                        |
| Measured cost (full run)                                                                          | **~$1.02**    | ~$9.73                                 | ~$4.70         | ~$110.28                                | ~$99.75                    | ~$109.52                                     |

Damage (produced by [grade_logs.py](scripts/grade_logs.py)) is defined as the Lichess win% a move gives away versus the
best move; 0pp is engine-perfect, <=5pp is
within [Stockfish's noise floor](https://chess.stackexchange.com/questions/38860/for-fixed-depth-search-how-much-is-the-efficiency-different-between-odd-and-eve),
~47.5pp is an even game thrown into a forced mate, larger values would be the difference from the win% the best move
would have given them. Damage cells read `mean (n graded)`; the row beneath the blunder-arm one counts the graded
suggestions inside that floor. A legal
`BEST_MOVE` cell's "of answered" figure divides by the rows whose output had a `BEST_MOVE` line the parser could read,
which is a different count from the format-failure row (that one counts missing verdict lines). Every table
above ran on golden v1
and regrades with `--golden golden/v1/golden_engine.csv`.

After switching to UCI, notation caused nearly every metric to drop for Haiku (substantiation was flat: 11.8% to 11.9%).
Sonnet saw an increase to most of its metrics, most noticeably best-class precision (+17pp) & recall (+22pp), and the
number of illegal moves suggested dropped.

The substantiation barely shifted between version runs despite the notation change - naming the certified refutation is
a verification problem, not a syntax problem, so grounding the model with Stockfish should move it. Ambiguous SAN
outputs fell from 3 to 1 (Haiku) and 2 to 0 (Sonnet).

I introduced Opus 5 at adaptive thinking to compare how a frontier model works ungrounded. Opus cries blunder a lot less
often with its Blunder recall at 38% and precision about the same as the previous models and increased its
substantiation massively, while reducing the number of unplayable refutations. It's unclear whether this is due to using
adaptive thinking or a more powerful model is the main reason for this shift. The default max token size with Inspect is
32k tokens (shared by thinking and answer), and the output of 37 samples were incomplete due to reaching that cap.

A great example of exhausting the 42k `max_tokens` output cap (32k plus the 10k Inspect adds for medium reasoning
effort): run `uv run inspect view` and open the link, take a look
at
[`0F0X2`](https://jbmlaird.github.io/chess-coach/#/logs/golden-v1%2Fsonnet-4-6-thinking%2F2026-08-21T11-27-27-00-00_Positions_eim5RxFdeHyg87yjkwiFps.eval/samples/sample/0F0X2/1) in the `logs/golden-v1/sonnet-4-6-thinking` folder: `Let me step back`,
`This is getting tangled, so let me step back`,
`This is getting complicated, so let me step back`, `This line is getting tangled, so let me step back`,
`Let me reconsider the position more practically`, `Stepping back from this deep line, I want to reconsider`,
`Given the complexity, I'll step back and just evaluate the overall line`,
`Actually, I'm overcomplicating this line. Let me step back` etc - it continuously re-evaluates the same lines then
needs to restart.

This isn't exclusive to thinking models, either. Opus 5 with no thinking also had max token exhaustion but the thinking
into a hole was done in the prose returned rather than in a separate reasoning section.

Compared with Opus, Sonnet 4.6 with no thinking performed the best with all samples finishing, and reasonable metrics at
5% of the cost of Opus. Grounded in the engine on golden v2 (the two tool tables below; Opus ran only on golden v1 with
instrument v2, whose prompt is byte-identical, so this compares across sets) both cheaper models pass the frontier
model's headline ungrounded numbers: overall 34.8% against 87.6-100.0%, best-move recall 62.0% against 86.0-100.0%,
substantiation 73.7% against 91.9-100%, at 3-6% of its cost. Haiku does not beat it on unplayable refutations (6-10/200
against 5/200) or format failures (9-38/250 against 0, its turn-limit hits).

### Instrument v3 · golden v2 · no tools · thinking off (2026-09-08)

The instrument is `Task(version=3)` with `tool_use=none`: the prompt is byte-identical to v2, the version bump only
introduces the task parameter that the tool arms will use. These are the no-tool baselines on golden v2 that every
grounded column will be read against. Logs live under `logs/golden-v2/`. Damage cells regrade with the default
reference (golden v2); the golden v1 damage cells in the two tables above still take
`--golden golden/v1/golden_engine.csv`.

|                                                                                                   | Haiku 4.5     | Sonnet 4.6         |
|---------------------------------------------------------------------------------------------------|---------------|--------------------|
| Verdict + refutation accuracy (overall)                                                           | 12.8%         | **22.0%**          |
| - blunder arm / best arm                                                                          | 8.0% / 32.0%  | 19.0% / 34.0%      |
| Blunder class - recall (caught real blunders)                                                     | **90.5%**     | 84.0%              |
| Blunder class - precision (calls that were right)                                                 | **84.2%**     | 83.6%              |
| Best class - recall (endorsed real best moves)                                                    | 32.0%         | **34.0%**          |
| Best class - precision (endorsements right)                                                       | **45.7%**     | 34.7%              |
| Best class - mean damage vs best play of suggested "improvements" (excludes correct endorsements) | 70.8pp (n=20) | **69.5pp (n=20)**  |
| Substantiation (correct blunder calls backing the certified refutation)                           | 8.8%          | **22.6%**          |
| Unplayable refutations (illegal, invalid or ambiguous)                                            | 95/200        | **39/200**         |
| Legal `BEST_MOVE` suggestions                                                                     | 55.2%         | **72.4%**          |
| Invalid `BEST_MOVE` answers (of which a lowercase piece letter)                                   | 32/250 (32)   | **1/250 (1)**      |
| Suggested improvement damage vs best play (blunder arm, legal alternatives to the blunder)        | 40.2pp (n=84) | **36.7pp (n=112)** |
| - of which within engine noise of best play (damage of 5pp or less)                               | 20/84 (24%)   | **27/112 (24%)**   |
| Unfinished samples (stop reason not `stop`)                                                       | **0**         | **0**              |
| Empty outputs (no text in the final generation)                                                   | **0**         | **0**              |
| Format failures (non-empty parse errors)                                                          | **0**         | **0**              |
| Measured cost (full run)                                                                          | **~$1.02**    | ~$4.70             |

Golden v2 shares 245 rows with v1, so this table is also a replication of the v2 Haiku and Sonnet columns: same prompt,
same settings, nineteen days apart. `calculate_metrics.py --compare_to <earlier log>` pairs the two runs on the
shared rows. Haiku's legal `BEST_MOVE` rate moved from 116/245 (47.3%) to 137/245 (55.9%) and its verdict + refutation
accuracy from 36/245 (14.7%) to 32/245 (13.1%); Sonnet moved from 177/245 (72.2%) to 178/245 (72.7%) and from 53/245
(21.6%) to 54/245 (22.0%). The aggregates hide how much the individual answers churn: Haiku's legality verdict changed
on 103 of the 245 rows (41 lost, 62 gained, the +21 behind its rise) and Sonnet's on 63 (31 lost, 32 gained,
cancelling). Two runs give one difference, not a distribution, so the honest statement is the paired standard error the
script prints from those discordant rows: about 4pp (Haiku) and 3pp (Sonnet) on legality, 2.5-3pp on verdict +
refutation. A single-run difference inside roughly twice that, 8pp on legality or 5pp on accuracy, is within sampling
noise for either model at n=250, and row-level comparisons between two runs are close to meaningless.

### Instrument v3 · golden v2 · Stockfish tool · Haiku 4.5 · thinking off (2026-09-14)

`tool_use` is a new parameter of the same task with 4 values: `none` is the baseline above without using Stockfish,
`silent` provides the tool without reference in the prompt, `optional` and `required` add one paragraph ("You may call
it as often as you like before answering." and "Call it at least once before you answer."). The tool is
`analyse(fen, moves)`, served over MCP by [stockfish_mcp.py](stockfish_mcp.py), which runs the same `Engine.grader()`
that certified the golden set; every log records the tool's engine and [grade_logs.py](scripts/grade_logs.py) refuses to
grade a mismatch. Each sample gets its own server for its whole life, may take at most eight turns (8 attempts/messages
by the LLM) and runs ten samples wide. `required` is enforced by prompt only so every arm is the same request shape and
any provider can run it.

On a tool arm the two scorers measure whether the model copied the engine rather than whether it judged the position,
so the standard rows are published only with the six tool rows beneath them, each cell reading `blunder arm . best
arm`. [tool_metrics.py](scripts/tool_metrics.py) reads them from the transcripts. Two rows condition on opportunity:
relay fidelity counts legal `BEST_MOVE` answers among samples that queried the student's position, and frame errors
count samples that queried the position after the played move. A sample that hits the turn limit is scored on its last
completed generation and counts as unfinished; the ninth generation is billed and discarded, and the cost row includes
it. Such a sample also shows as a format failure or, when that last generation was a bare tool call, as an empty
output.

|                                                                                                               | No tools      | Tool offered, unmentioned (`silent`) | Tool described (`optional`) | Tool required (`required`) |
|---------------------------------------------------------------------------------------------------------------|---------------|--------------------------------------|-----------------------------|----------------------------|
| Verdict + refutation accuracy (overall)                                                                       | 12.8%         | 87.6%                                | 78.8%                       | **88.8%**                  |
| - blunder arm / best arm                                                                                      | 8.0% / 32.0%  | 85.0% / 98.0%                        | 77.0% / 86.0%               | 87.5% / 94.0%              |
| Blunder class - recall (caught real blunders)                                                                 | 90.5%         | 92.5%                                | 82.5%                       | **93.5%**                  |
| Blunder class - precision (calls that were right)                                                             | 84.2%         | **100.0%**                           | 99.4%                       | 99.5%                      |
| Best class - recall (endorsed real best moves)                                                                | 32.0%         | **98.0%**                            | 86.0%                       | 94.0%                      |
| Best class - precision (endorsements right)                                                                   | 45.7%         | 87.5%                                | 93.5%                       | **97.9%**                  |
| Best class - mean damage vs best play of suggested "improvements" (excludes correct endorsements)             | 70.8pp (n=20) | n=0                                  | **45.6pp (n=1)**            | n=0                        |
| Substantiation (correct blunder calls backing the certified refutation)                                       | 8.8%          | 91.9%                                | 93.3%                       | **93.6%**                  |
| Unplayable refutations (illegal, invalid or ambiguous)                                                        | 95/200        | 10/200                               | **6/200**                   | 8/200                      |
| Legal `BEST_MOVE` suggestions                                                                                 | 55.2%         | **95.6% (99.2% of answered)**        | 84.4% (99.1% of answered)   | 93.2% (98.3% of answered)  |
| Invalid `BEST_MOVE` answers (of which a lowercase piece letter)                                               | 32/250 (32)   | **0/250 (0)**                        | 1/250 (0)                   | 1/250 (0)                  |
| Suggested improvement damage vs best play (blunder arm, legal alternatives to the blunder)                    | 40.2pp (n=84) | 1.1pp (n=183)                        | **0.1pp (n=164)**           | 0.6pp (n=185)              |
| - of which within engine noise of best play (damage of 5pp or less)                                           | 20/84 (24%)   | 179/183 (98%)                        | **162/164 (99%)**           | 180/185 (97%)              |
| Unfinished samples (stop reason not `stop`)                                                                   | **0**         | 9/250 (3.6%)                         | 37/250 (14.8%)              | 13/250 (5.2%)              |
| Empty outputs (no text in the final generation)                                                               | **0**         | **0**                                | **0**                       | **0**                      |
| Format failures (non-empty parse errors)                                                                      | **0**         | 9/250 (3.6%)                         | 38/250 (15.2%)              | 14/250 (5.6%)              |
| Measured cost (full run)                                                                                      | **~$1.02**    | ~$3.68                               | ~$4.90                      | ~$3.59                     |
| Tool errors (illegal moves or bad FENs sent to the engine), samples with one                                  | -             | 85/200 . 16/50                       | 77/200 . 22/50              | **53/200 . 9/50**          |
| Tool calls per sample                                                                                         | -             | 4.9 . 3.6                            | 5.5 . 4.2                   | 3.9 . 2.8                  |
| Called the tool before answering                                                                              | -             | **192/200 . 49/50**                  | 169/200 . 44/50             | 189/200 . 48/50            |
| Relay fidelity (legal `BEST_MOVE` = engine best move for the queried position)                                | -             | 177/184 . 46/46                      | 163/167 . 42/43             | **181/184 . 46/46**        |
| Beyond-engine answers (legal moves the engine never returned)                                                 | -             | 10/365 . 2/49                        | 4/326 . 0/44                | 2/365 . 1/47               |
| Frame errors (opponent's reply reported as the student's move, of samples that queried the position after it) | -             | **0/193 . 0/48**                     | **0/190 . 0/41**            | **0/192 . 0/48**           |

Haiku gains the most and loses the most to its own tool use. Verdict + refutation accuracy goes from 12.8% to 87.6%
with the tool silently offered and 88.8% when required; legal suggestions from 55.2% to 84-96%; unplayable
refutations from 95/200 to ten or fewer; and the suggested improvement, which gave away 40 points against best play
without the tool, now sits within engine noise 97-99% of the time because it is the engine's move (relay fidelity
96-98% on blunder rows). What it still gets wrong is mostly mechanical. The engine rejected at least one call on 53-85
of 200 blunder rows (27-43%): a corrupted FEN (a rank with the wrong number of squares, the side not to move left in
check, pawns on the back rank, a missing king) or an illegal move. It also sent positions the engine accepted but that
are unreachable from the student's, which no row counts, and 9-37 samples ran into the turn limit, usually with the
engine's answers already in the transcript ([`cjOIr`](https://jbmlaird.github.io/chess-coach/#/logs/golden-v2%2Fhaiku-4-5-tool-optional%2F2026-09-14T17-09-31-00-00_Positions_82SghKgWJi5rMoZ6xcWLVt.eval/samples/sample/cjOIr/1) under `optional` spends its
nine turns proposing moves for a king that is in check). The rest is judgement: a few verdicts per arm contradict engine output the
model had already received. The `optional` wording did worst where it decides the column: the most calls per sample
(5.5), the most limit hits (37), the lowest accuracy (78.8%) and the highest cost ($4.90); the answers it did give were
as good as the other arms'. An invitation to call "as often as you like" was taken literally.

### Instrument v3 · golden v2 · Stockfish tool · Sonnet 4.6 · thinking off (2026-09-14)

|                                                                                                               | No tools          | Tool offered, unmentioned (`silent`) | Tool described (`optional`) | Tool required (`required`) |
|---------------------------------------------------------------------------------------------------------------|-------------------|--------------------------------------|-----------------------------|----------------------------|
| Verdict + refutation accuracy (overall)                                                                       | 22.0%             | **100.0%**                           | 98.4%                       | **100.0%**                 |
| - blunder arm / best arm                                                                                      | 19.0% / 34.0%     | 100.0% / 100.0%                      | 98.0% / 100.0%              | 100.0% / 100.0%            |
| Blunder class - recall (caught real blunders)                                                                 | 84.0%             | **100.0%**                           | 98.0%                       | **100.0%**                 |
| Blunder class - precision (calls that were right)                                                             | 83.6%             | **100.0%**                           | **100.0%**                  | **100.0%**                 |
| Best class - recall (endorsed real best moves)                                                                | 34.0%             | **100.0%**                           | **100.0%**                  | **100.0%**                 |
| Best class - precision (endorsements right)                                                                   | 34.7%             | **100.0%**                           | **100.0%**                  | **100.0%**                 |
| Best class - mean damage vs best play of suggested "improvements" (excludes correct endorsements)             | **69.5pp (n=20)** | n=0                                  | n=0                         | n=0                        |
| Substantiation (correct blunder calls backing the certified refutation)                                       | 22.6%             | **100.0%**                           | **100.0%**                  | **100.0%**                 |
| Unplayable refutations (illegal, invalid or ambiguous)                                                        | 39/200            | **0/200**                            | **0/200**                   | **0/200**                  |
| Legal `BEST_MOVE` suggestions                                                                                 | 72.4%             | **100.0%**                           | 98.4% (100.0% of answered)  | **100.0%**                 |
| Invalid `BEST_MOVE` answers (of which a lowercase piece letter)                                               | 1/250 (1)         | **0/250 (0)**                        | **0/250 (0)**               | **0/250 (0)**              |
| Suggested improvement damage vs best play (blunder arm, legal alternatives to the blunder)                    | 36.7pp (n=112)    | 0.1pp (n=200)                        | -0.1pp (n=196)              | **-0.1pp (n=200)**         |
| - of which within engine noise of best play (damage of 5pp or less)                                           | 27/112 (24%)      | 198/200 (99%)                        | 195/196 (99%)               | **199/200 (100%)**         |
| Unfinished samples (stop reason not `stop`)                                                                   | **0**             | **0**                                | 4/250 (1.6%)                | **0**                      |
| Empty outputs (no text in the final generation)                                                               | **0**             | **0**                                | 3/250 (1.2%)                | **0**                      |
| Format failures (non-empty parse errors)                                                                      | **0**             | **0**                                | 1/250 (0.4%)                | **0**                      |
| Measured cost (full run)                                                                                      | **~$4.70**        | ~$5.75                               | ~$6.90                      | ~$5.76                     |
| Tool errors (illegal moves or bad FENs sent to the engine), samples with one                                  | -                 | 19/200 . 13/50                       | 33/200 . 11/50              | **16/200 . 6/50**          |
| Tool calls per sample                                                                                         | -                 | 3.8 . 2.7                            | 4.6 . 3.1                   | 3.4 . 2.3                  |
| Called the tool before answering                                                                              | -                 | **200/200 . 50/50**                  | 196/200 . 50/50             | **200/200 . 50/50**        |
| Relay fidelity (legal `BEST_MOVE` = engine best move for the queried position)                                | -                 | 198/199 . 50/50                      | **196/196 . 50/50**         | **200/200 . 50/50**        |
| Beyond-engine answers (legal moves the engine never returned)                                                 | -                 | 1/400 . 0/50                         | 0/392 . 0/50                | 0/400 . 0/50               |
| Frame errors (opponent's reply reported as the student's move, of samples that queried the position after it) | -                 | **0/199 . 0/39**                     | **0/200 . 0/39**            | **0/200 . 0/45**           |

Sonnet with the tool is a faithful relay: 250/250 on both scorers with the tool silently offered and again when
required, and the four misses under `optional` are all turn-limit hits. Relay fidelity is 198/199, 196/196 and 200/200
on blunder rows and 50/50 on every best arm; the one exception ([`UQlwg`](https://jbmlaird.github.io/chess-coach/#/logs/golden-v2%2Fsonnet-4-6-tool-silent%2F2026-09-14T17-01-53-00-00_Positions_cTe9eraBrUSYDmQHAPfYfb.eval/samples/sample/UQlwg/1), silent arm) misread which side the engine's score was for and
answered a move well outside the noise floor. One answer in the three runs went beyond the engine (1/400 blunder-arm
legal fields, silent arm), and suggested improvements sit within engine noise on 99-100% of rows. The engine rejected a
call on 16-33 of 200 blunder rows (8-16.5%) and, except for three of the four `optional` limit hits, the model
recovered within its turns. Cost rose from $4.70 to $5.75-6.90 a run; costs are as billed, and Sonnet's requests were
mostly cache reads while Haiku's were almost entirely uncached input, so the Haiku figures are closer to list price. No
frame errors on the samples that queried the position after the move (the row's denominator): the row measures the
opponent's reply reported as the student's move, not misreadings of the score's perspective, which did occur in a few
transcripts. Whether the tool description's perspective sentence prevented more of them is untested, since no arm ran
without it.

Together the six runs cost $30.58, the sum of the six cost cells.

### Invalid best moves: Haiku's lowercase piece letters

The prompt asks for UCI (from-square then to-square, `e7e5`). The parser is more tolerant than the prompt: it accepts
UCI or correctly cased SAN, because python-chess parses both. What it cannot parse is a lower-cased rook, knight, queen
or king letter: plain lowercase SAN (`rd1`, `kf8`, `kc3`) or a piece letter bolted onto a UCI move (`rd3e2`, `qe4e8`,
`qf5f1`). Those score `INVALID` and count as incorrect. A lower-cased bishop is read as the b-pawn file instead and
scores `ILLEGAL`, so the counts below slightly undercount the habit. Since the prompt states the notation and the format
is the contract, they stay incorrect. The `invalid best moves` line of
[calculate_metrics.py](scripts/calculate_metrics.py) counts them.

|                                     | Haiku 4.5, SAN prompt, golden v1 (no thinking) | Haiku 4.5, UCI prompt, golden v1 (no thinking) | Haiku 4.5, UCI prompt, golden v1 (thinking) | Haiku 4.5, UCI prompt, golden v2 (no thinking) | Sonnet 4.6, UCI prompt, golden v1 (no thinking) | Sonnet 4.6, UCI prompt, golden v2 (no thinking) |
|-------------------------------------|------------------------------------------------|------------------------------------------------|---------------------------------------------|------------------------------------------------|-------------------------------------------------|-------------------------------------------------|
| Legal `BEST_MOVE` suggestions       | 58.0%                                          | 47.2%                                          | 70.0%                                       | 55.2%                                          | 72.0%                                           | 72.4%                                           |
| Invalid `BEST_MOVE` answers         | 0/250                                          | 30/250                                         | 0/250                                       | 32/250                                         | 0/250                                           | 1/250                                           |
| - of which a lowercase piece letter | 0                                              | 30                                             | 0                                           | 32                                             | 0                                               | 1                                               |
| Unplayable refutations              | 84/200                                         | 88/200                                         | 46/200                                      | 95/200                                         | 39/200                                          | 39/200                                          |

The class is a stable property of Haiku 4.5 without thinking under the UCI instruction, not run-to-run noise: 30 and
32 in the two UCI runs, none under the SAN prompt, none with thinking on, and one for Sonnet across two runs. It is also
the size of the legality drop that followed the switch to UCI: legal answers fell from 145 to 118 while, per
`calculate_metrics.py --verbose`, ILLEGAL stayed flat (102 to 101), AMBIGUOUS fell from 3 to 1 and INVALID rose from 0
to 30. That is a statement about totals, not about which rows moved, and it says nothing about whether those moves would
have been legal with the piece letter cased. The grounded arms should make the class vanish: a model that copies
`best_move` out of the tool result gets a clean UCI string for free.

## motif_detector.py: an independent cross-check

The shipped labels come from the vendored cook.py unmodified. motif_detector.py is kept as an independent cross-check
on the vendored tagger's output; its `hanging_piece` deliberately differs:

- cook.py skips hung pawns, I include them. A one piece hang is still a hang, knowingly including gambits.
- En passant hangs exist therefore included
- cook.py refuses to tag when the setup move gives check and only a pawn (or nothing) is captured. Since I keep pawn
  victims, I keep these too.

Cross-checked on the same 4,000-row sample: the two agree on 3,887/4,000 verdicts (97.2%), and all 113 disagreements
are the documented pawn/en-passant class - zero unexplained, zero where Lichess fires and I don't.
