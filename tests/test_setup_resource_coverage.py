import hashlib
import json
from pathlib import Path

import pytest

pytest.importorskip("catan_ranked_fast")

from catan_ranked.value_arena import (  # noqa: E402
    _production_by_resource,
    choose_greedy_action,
)
from catan_ranked.value_residual_features import TERM_NAMES  # noqa: E402
from catan_ranked_fast import Game  # noqa: E402


FIXTURE = json.loads(Path("tests/fixtures/setup_coverage_root.json").read_text())
WEIGHTS = json.loads(Path(
    "weights/champion.json").read_text())["weights"]
WEIGHT_VECTOR = [WEIGHTS[name] for name in TERM_NAMES]


def _state_hash(game):
    return hashlib.sha256(game.replay_state_json().encode()).hexdigest()


def _planner(game, enabled):
    search = game.turn_search(128, max_actions=14, chance_samples=4)
    values = search.weighted_linear_values(WEIGHT_VECTOR)
    action_values = search.action_values(values)
    return choose_greedy_action(
        game, action_values, setup_resource_coverage=enabled), action_values


def _after_profile(game, action):
    child = game.clone_game()
    child.apply(action)
    observation = child.observation(game.current_player)
    production = _production_by_resource(observation)
    owns_brick_port = observation[1016 + 21] > 0.5
    return production, owns_brick_port


def test_setup_coverage_is_default_off_and_fixes_exact_human_game_board():
    game = Game(FIXTURE["seed"])
    for action, roll, stolen in FIXTURE["prefix"]:
        game.apply(action, roll, stolen)
    assert _state_hash(game) == FIXTURE["first_root_hash"]

    baseline_first, values = _planner(game, enabled=False)
    covered_first, covered_values = _planner(game, enabled=True)
    assert baseline_first == FIXTURE["first_settlement"]
    assert covered_first == baseline_first
    assert covered_values == values

    game.apply(FIXTURE["first_settlement"])
    game.apply(FIXTURE["first_road"])
    assert _state_hash(game) == FIXTURE["second_root_hash"]
    baseline, values = _planner(game, enabled=False)
    covered, covered_values = _planner(game, enabled=True)
    assert baseline == FIXTURE["recorded_second_settlement"]
    assert covered == FIXTURE["coverage_second_settlement"]
    assert covered_values == values

    baseline_production, baseline_brick_port = _after_profile(game, baseline)
    covered_production, _covered_brick_port = _after_profile(game, covered)
    assert baseline_production[1] == pytest.approx(0.0)
    assert not baseline_brick_port
    assert covered_production[1] > 0.0
    assert sum(covered_production) >= sum(baseline_production) - 1.0 / 36.0
