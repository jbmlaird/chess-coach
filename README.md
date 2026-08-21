# Chess Coach Eval

This is an evaluation framework evaluating LLMs and their chess accuracy. It's a rebuild of an AI agent repo I have
where I'm rewriting the evaluation logic by hand to get a solid understanding.

### Dataset selection

To try and avoid training memorisation of chess positions, puzzles are selected after the tested model's cutoff dates.
Anthropic's can be found [here](https://support.claude.com/en/articles/8114494-how-up-to-date-is-claude-s-training-data)
(with Opus 5 being May 2026) and OpenAI's [here](https://developers.openai.com/api/docs/models) (Feb 2026).

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

A golden dataset of 250 puzzles was sampled and written to `golden_candidates.csv`. This has a mixture of themes (such
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

## Results

### Baseline results (2026-08-18/19, golden v1, 250 rows, no tools)

|                                                                         | Haiku 4.5 | Sonnet 4.6 |
|-------------------------------------------------------------------------|-----------|------------|
| Verdict + refutation accuracy (overall)                                 | 18.4%     | 16.8%      |
| - blunder arm / best arm                                                | 11% / 48% | 13% / 32%  |
| Blunder class - recall (caught real blunders)                           | 93.0%     | 86.5%      |
| Blunder class - precision (calls that were right)                       | 87.7%     | 83.6%      |
| Best class - recall (endorsed real best moves)                          | 48.0%     | 32.0%      |
| Best class - precision (endorsements right)                             | 63.2%     | 37.2%      |
| Substantiation (correct blunder calls backing the certified refutation) | 11.8%     | 15.0%      |
| Unplayable refutations (illegal, invalid or ambiguous)                                   | 84/200    | 46/200     |
| Legal `BEST_MOVE` suggestions                                           | 58%       | 69%        |
| Measured cost (full run)                                                | ~$1.00    | ~$4.04     |

Cutting edge frontier models are currently excluded until the harness is built out more. Logs are viewable in `/logs`
viewable with `uv run inspect view` from the root. The accuracy rows are pulled from the log metadata, the rest of the
metrics are calculated via the [calculate_metrics](scripts/calculate_metrics.py) script.

Both models love to call things a blunder (93%/86.5% recall blunder-class), with the best-arm class showing that it errs
on the side of calling best moves also a blunder (48%/32% recall best-class). The blunder-class recall would make it
seem like the model is performing great without the best-class stats.

When a model correctly calls a blunder, it can only name the punishing reply 11.8%/15.0% of the time.

After the first run against Haiku & Sonnet, I noticed that the "best move" suggested by the LLM are legal but bad. For
example, `0F0X2` suggested best move `Qd4` which immediately hangs the queen. In `nBP6h`, the refutation line doesn't
show a move but instead shows `REFUTATION: The move wastes a chance; Black should have played Bxa1 to win White's
bishop. After Bb2, White continues but Black has missed the decisive material advantage.` not honouring the prompt.
`fMkW1` suggested move `Rc6+` which was ambiguous as either rook could move to c6 and give check. To address this,
all moves need to be provided in UCI so this ambiguity can be removed.

Sample [3hzja](https://lichess.org/training/3hzja) (best arm) shows why we measure both legality and ground truth.
The played move is Kxc7 - king takes rook. Haiku got the best verdict right but restated the move as `Rxc7`,
misidentifying the capturing piece as the rook. Sonnet did the opposite by suggesting a perfectly legal move (`Ke5`)
attached to the wrong verdict. Board-state tracking and chess judgment fail independently; one scorer would hide half
the picture.

For both of the above cases, I decided
to [use UCI notation instead of SAN](https://github.com/jbmlaird/chess-coach/commit/78e80ae9f87d0ff912ad3ede3b72410de5089d76)
aligning with the standard that UCI was created for chess engines since knowledge of the piece isn't required.

### v2 results (2026-08-20, golden v1, 250 rows, no tools)

|                                                                         | Haiku 4.5   | Sonnet 4.6  |
|-------------------------------------------------------------------------|-------------|-------------|
| Verdict + refutation accuracy (overall)                                 | 14.4%       | 21.6%       |
| - blunder arm / best arm                                                | 10.5% / 30% | 13.5% / 54% |
| Blunder class - recall (caught real blunders)                           | 88.5%       | 88.5%       |
| Blunder class - precision (calls that were right)                       | 83.5%       | 88.5%       |
| Best class - recall (endorsed real best moves)                          | 30.0%       | 54.0%       |
| Best class - precision (endorsements right)                             | 39.5%       | 54.0%       |
| Substantiation (correct blunder calls backing the certified refutation) | 11.9%       | 15.3%       |
| Unplayable refutations (illegal, invalid or ambiguous)                                   | 88/200      | 39/200      |
| Legal `BEST_MOVE` suggestions                                           | 47.2%       | 72%         |
| Measured cost (full run)                                                | ~$1.02      | ~$4.70      |

After switching to UCI, notation caused nearly every metric to drop for Haiku (substantiation was flat: 11.8% to 11.9%).
Sonnet saw an increase to most of its metrics, most noticeably best-class precision (+17pp) & recall (+22pp), and the
number of illegal moves suggested dropped.

Worth pointing out that the substantiation barely shifted between version runs despite the notation change - naming
the certified refutation is a verification problem, not a syntax problem, so grounding the model with Stockfish should
move it. Ambiguous SAN outputs fell from 3 to 1 (Haiku) and 2 to 0 (Sonnet).

## motif_detector.py: an independent cross-check

The shipped labels come from the vendored cook.py unmodified. motif_detector.py is kept as an independent cross-check
on the vendored tagger's output; its `hanging_piece` deliberately differs:

- cook.py skips hung pawns, I include them. A one piece hang is still a hang, knowingly including gambits.
- En passant hangs exist therefore included
- cook.py refuses to tag when the setup move gives check and only a pawn (or nothing) is captured. Since I keep pawn
  victims, I keep these too.

Cross-checked on the same 4,000-row sample: the two agree on 3,887/4,000 verdicts (97.2%), and all 113 disagreements
are the documented pawn/en-passant class - zero unexplained, zero where Lichess fires and I don't.