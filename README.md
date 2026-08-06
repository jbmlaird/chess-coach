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


