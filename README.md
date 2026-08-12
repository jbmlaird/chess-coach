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

## motif_detector.py: an independent cross-check

The shipped labels come from the vendored cook.py unmodified. motif_detector.py is kept as an independent cross-check
on the vendored tagger's output; its `hanging_piece` deliberately differs:

- cook.py skips hung pawns, I include them. A one piece hang is still a hang, knowingly including gambits.
- En passant hangs exist therefore included
- cook.py refuses to tag when the setup move gives check and only a pawn (or nothing) is captured. Since I keep pawn
  victims, I keep these too.

Cross-checked on the same 4,000-row sample: the two agree on 3,887/4,000 verdicts (97.2%), and all 113 disagreements
are the documented pawn/en-passant class - zero unexplained, zero where Lichess fires and I don't.
