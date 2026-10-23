import hashlib
import json
from pathlib import Path

import pytest

pytest.importorskip("catan_ranked_fast")

from catan_ranked.value_arena import choose_greedy_action
from catan_ranked.value_residual_features import TERM_NAMES
from catan_ranked_fast import Game, decode_action_json


FIXTURE = json.loads(Path("tests/fixtures/dev_hoarding_roots.json").read_text())
WEIGHT_PAYLOAD = json.loads(Path("tests/fixtures/replica_a_weights.json").read_text())
WEIGHTS = [WEIGHT_PAYLOAD["weights"][name] for name in TERM_NAMES]


def _state_hash(game):
    return hashlib.sha256(game.replay_state_json().encode()).hexdigest()


def _replay_roots(game_fixture):
    game = Game(game_fixture["seed"])
    roots = {int(sequence): root for sequence, root in game_fixture["roots"].items()}
    for sequence, (action, roll, stolen) in enumerate(game_fixture["actions"], 1):
        if sequence in roots:
            assert _state_hash(game) == roots[sequence]["state_hash_before"]
            yield sequence, game, roots[sequence]
        game.apply(action, roll, stolen)


def _planner_action(game, *, liquidation, spending_guard=False):
    search = game.turn_search(128, max_actions=14, chance_samples=4)
    action_values = search.action_values(search.weighted_linear_values(WEIGHTS))
    action = choose_greedy_action(
        game,
        action_values,
        endgame_dev_card_liquidation=liquidation,
        build_spending_guard=spending_guard,
    )
    return action, action_values


def _is_dev_play(action):
    return json.loads(decode_action_json(action))["kind"].startswith("play_")


@pytest.mark.xfail(strict=True, reason="expected actions predate the Knight-only pre-roll rule; root 272 now rolls instead of playing Year of Plenty. Re-derive expectations (see docs/OPEN_PROBLEMS.md).")
def test_rebuilt_endgame_roots_preserve_replica_a_and_liquidate_close_dev_plays():
    expected_fixed = {
        233: 304,  # knight before roll
        234: 304,  # knight before maritime trades
        250: 328,  # YoP before roll
        251: 327,  # YoP instead of end turn
        255: 329,  # YoP before roll
        256: 327,  # YoP instead of end turn
        272: 318,  # YoP before roll
        273: 312,  # YoP instead of brick -> ore trade
    }
    seen = set()
    for sequence, game, root in _replay_roots(FIXTURE["game1"]):
        baseline, action_values = _planner_action(game, liquidation=False)
        fixed, fixed_values = _planner_action(game, liquidation=True)
        assert baseline == root["recorded_action"]
        assert fixed == expected_fixed[sequence]
        assert _is_dev_play(fixed)
        assert action_values == fixed_values
        seen.add(sequence)
    assert seen == set(expected_fixed)


def test_rebuilt_roots_expose_dev_candidate_coverage_pre_roll_and_main_phase():
    expected_counts = {
        233: (21, 6),   # all YoP actions legal, but the 128-leaf root budget omits them
        234: (21, 21),  # all knight/Monopoly/YoP actions scored after the roll
        250: (15, 15),
        251: (15, 15),
    }
    for sequence, game, _root in _replay_roots(FIXTURE["game1"]):
        if sequence not in expected_counts:
            continue
        legal_devs = sum(_is_dev_play(action) for action in game.legal_actions())
        _action, action_values = _planner_action(game, liquidation=False)
        scored_devs = sum(_is_dev_play(action) for action, _value in action_values)
        assert (legal_devs, scored_devs) == expected_counts[sequence]


def test_game_two_flagged_buy_and_road_roots_have_no_playable_dev_card():
    seen = set()
    for sequence, game, root in _replay_roots(FIXTURE["game2"]):
        baseline, action_values = _planner_action(game, liquidation=False)
        fixed, _ = _planner_action(game, liquidation=True)
        assert baseline == root["recorded_action"]
        assert fixed == baseline  # only 2 VP: the endgame option is intentionally inactive
        assert not any(_is_dev_play(action) for action in game.legal_actions())
        assert not any(_is_dev_play(action) for action, _value in action_values)
        seen.add(sequence)
    assert seen == {72, 73}


def test_build_spending_guard_fixes_exact_game_two_roots_and_is_default_off():
    expected_hashes = {
        29: "da3de05260416186162ebb40f3edee1d64ca5396b25c1fb05421f224a37751fe",
        30: "703f0f589b639d29a27044de5addcd64c90d261a662cb95837cf735c0b77ef6c",
        72: "9855bcbb6af2cacc4c88f6338c17aa9148ec246dfca87ab60321237c43d5980a",
        73: "1959e503bf2d90557b1ce693cf5a60b270d4a8a49812fdfa0dabbd8d3e5faf81",
    }
    recorded = {29: 89, 30: 62, 72: 180, 73: 94}
    # Every replacement is present in the N=800 retained rollouts and
    # improves on the recorded choice: +14.4pp, +17.8pp, +9.3pp, and +9.4pp respectively.
    fixed = {29: 64, 30: 64, 72: 148, 73: 64}
    game = Game(FIXTURE["game2"]["seed"])
    seen = set()
    for sequence, (action, roll, stolen) in enumerate(FIXTURE["game2"]["actions"], 1):
        if sequence in fixed:
            assert _state_hash(game) == expected_hashes[sequence]
            baseline, baseline_values = _planner_action(game, liquidation=True)
            guarded, guarded_values = _planner_action(
                game, liquidation=True, spending_guard=True)
            assert baseline == recorded[sequence]
            assert guarded == fixed[sequence]
            assert guarded_values == baseline_values
            seen.add(sequence)
        game.apply(action, roll, stolen)
    assert seen == set(fixed)
