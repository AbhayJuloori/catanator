# Contributing

Thanks for looking. The fastest way to help is to pick a problem from
[docs/OPEN_PROBLEMS.md](docs/OPEN_PROBLEMS.md) and open an issue with its number.

## How a change gets in

1. **Put new behavior behind a flag that is off by default.** The champion's exact behavior
   must stay reproducible with every flag off. Look at `setup_draft_search` in
   `catan_ranked/value_arena.py` for the pattern.
2. **Add a test** that pins the behavior on at least one exact position. Positions are
   rebuilt from a seed plus an action list and checked with `replay_state_json()` hashes.
3. **Measure it against the champion** with mirrored pairs and SPRT:

   ```bash
   PYTHONPATH=. .venv/bin/python scripts/run_value_arena.py \
     --candidate weights/champion.json --candidate-setup-draft-search --candidate-setup-draft-features \
     --baseline weights/champion.json --baseline-setup-draft-search --baseline-setup-draft-features \
     --max-pairs 4000 --workers 8 --elo1 10 --output results/your_change.json
   ```

   Add your flag to the candidate side. Post the `counts`, `score`, `llr`, and `decision`
   fields in the pull request.
4. **No conclusions from small samples.** A result after 30 pairs is a pilot, not evidence.
   We have been fooled by this more than once.

## Rules of thumb

- Big levers over small ones: search structure, learned value, data scale. Another hand-tuned
  scalar term rarely moves the needle.
- Simulator wins are the bar for merging. Human-game evidence is welcome as motivation.
- Keep Rust engine changes behind the existing tests; any rules change needs a test that
  replays a full game.
