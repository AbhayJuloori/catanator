# Open problems

Each problem below lists what we tried, the measured result, and what a fix would need to
show. All scores are mirrored-pair scores (0.5 = even). "Champion" means the current bot:
`weights/champion.json` plus opening draft search and draft features. "Previous champion"
is the depth-2 alpha-beta bot with the hand-tuned evaluation (`--baseline champion`).

If you want to work on one, open an issue with the problem number so we do not duplicate
effort.

---

## 1. A value function that beats the tuned linear evaluation

**Status.** The planner scores end-of-turn positions with 24 linear terms. CMA-ES tuning of
those weights gave the current champion (about 0.585 vs the previous champion over 2,000
pairs, roughly +60 Elo). A second CMA-ES round (720 candidates, 576,000 mirrored pairs) never moved
the parent mean past 0.577, so the linear form looks saturated.

**What we tried.**
- A residual MLP on top of the linear base: first looked like 0.68 after 39 pairs, then
  replicated at only 0.505 over 400 fresh pairs. Early stops on small samples lie.
- An earlier AlphaZero-style track (policy and value net with MCTS) starved for data:
  200 to 2,000 game corpora on a Python engine at 4 to 8 seconds per game.

**Why it might work now.** The Rust engine is 114x faster, so millions of self-play games
are realistic on free CPU. `scripts/generate_value_selfplay.py` produces training data from
end-of-turn afterstates.

**Ideas.** TD(lambda) on afterstates (the TD-Gammon recipe for dice games), better input
features (per-resource production by number, port access, reachable sites), expert iteration
with search-improved targets.

**Done when.** A learned value passes SPRT [0, 10] Elo against the champion, then holds up on
a fresh 400-pair confirmation run.

---

## 2. Opponent lookahead makes the bot weaker

**Status.** Adding a chance node over the opponent's roll plus the opponent's best reply
should help. It lost on every base:

| Base | With opponent lookahead |
|---|---:|
| Linear (untuned) | 0.314 |
| CMA-tuned (champion weights) | 0.396 vs same weights without lookahead |
| Residual MLP | 0.444 vs previous champion |

Alternative opponent models also lost: opponent maximizes its own value 0.375, opponent only
rolls 0.353.

**Hypothesis.** The weights were tuned for one specific point in the turn (own end of turn).
Scoring two turns deep puts leaves at a different point, where the weights mean something
else.

**Ideas.** Co-tune weights with lookahead switched on; evaluate leaves at a consistent point
for both players; a value function trained on the deeper leaves (ties to problem 1).

---

## 3. Passivity: it ends turns it should not end

**Status.** In a rollout study of 144 positions where the bot chose to end its turn,
building a road instead won more often in 29 of them (20%), mean +1.6 points of win rate.
Human test games show the same pattern: the largest single mistakes flagged by rollout
analysis were "ended turn instead of building a road" (+13 to +16 points of win rate).

**What we tried.** A spending guard that reserved cards less aggressively. Version 1 made it
hoard instead: dev buys fell from 11 to 4.6 per game, discards rose from 6 to 23 per game,
and it scored 0.23. Version 2 was narrowed to a few cases and scored 0.506 over 4,000 pairs,
which is noise.

**Why.** The turn search stops at the end of the bot's own turn, so it cannot see that being
two ore short of a city now is worth more than a road that only adds length. Generic
"cards in hand" is the only term that rewards holding.

**Done when.** A principled fix (not a hand rule) passes SPRT against the champion without
raising discards.

---

## 4. Development card strategy

**Status.** Dev card play is mostly greedy one-turn choice. An endgame liquidation rule
(play cards when close to winning) scored 0.505 over 4,000 pairs: real but tiny or zero.

**Open questions.**
- Knight and robber: block the hex the opponent most needs, not just their best hex.
- Road Building: use it for contested settlement spots, not just road length.
- Monopoly and Year of Plenty: time them to complete a build this turn.

Default-off toggles for each of these exist in `catan_ranked/value_arena.py` and are being
screened now. Results so far are not decisive.

---

## 5. Opening resource coverage

**Status.** Opening draft search was the biggest recent win (0.565, accepted after 825
pairs). Weak spot left: it sometimes opens with no brick or no wood. A coverage rule that
forced wood and brick access scored 0.490 and was rejected; in the exact human game that
motivated it, the original choice actually won 85% of rollouts versus 29% for the "fixed"
one. Ports have no evaluation term at all.

**Ideas.** A port term; learned opening values from self-play (ties to problem 1).

---

## 6. Measuring small gains cheaply

**Status.** Effects worth having now are 1 to 3 points of score. Confirming +1 point needs
thousands of mirrored pairs, and the opening search makes each pair cost about 18 seconds on
4 cores.

**Ideas.** Variance reduction beyond mirroring (control variates on final VP difference,
common random numbers per decision), a faster opening search, a cheap proxy metric that
predicts SPRT outcomes.

---

## Good first issues

- `tests/test_dev_hoarding_roots.py` has one test marked `xfail`. Its expected actions
  predate the rule that only a Knight can be played before rolling. Re-derive the
  expectations and remove the marker.
- Silence the three Rust compiler warnings in `fast_engine`.
- Add a GitHub Actions workflow that builds the engine and runs the test suite.
