import hashlib
import json
from pathlib import Path

import pytest

pytest.importorskip("catan_ranked_fast")

from catan_ranked.setup_draft import choose_setup_draft_action  # noqa: E402
from catan_ranked.value_arena import choose_greedy_action  # noqa: E402
from catan_ranked.value_residual_features import TERM_NAMES  # noqa: E402
from catan_ranked_fast import Game  # noqa: E402


SEED = 2_159_100_487_904_895_606
PREFIX_TO_FIRST_BOT_ROAD = (178, 122, 136)  # human v8/r68, bot v18
ROOT_HASH = "ecf81748516794afe50f0dda0f4afc61211df1af4015475eb4e8ff5580409a43"
WEIGHTS = json.loads(Path(
    "weights/champion.json").read_text())["weights"]
WEIGHT_VECTOR = [WEIGHTS[name] for name in TERM_NAMES]


def _root():
    game = Game(SEED)
    for action in PREFIX_TO_FIRST_BOT_ROAD:
        game.apply(action)
    return game


def _action_values(game):
    search = game.turn_search(128, max_actions=14, chance_samples=4)
    return search.action_values(search.weighted_linear_values(WEIGHT_VECTOR))


def test_setup_draft_search_avoids_human_game_cutoff_lane_and_is_opt_in():
    game = _root()
    assert hashlib.sha256(game.replay_state_json().encode()).hexdigest() == ROOT_HASH
    values = _action_values(game)

    baseline = choose_greedy_action(
        game, values, build_spending_guard=True, weights=WEIGHT_VECTOR)
    assert baseline == 69  # edge 15 [16,18], recorded and later cut off at v21

    draft = choose_setup_draft_action(
        game, WEIGHT_VECTOR, preferred=baseline, feature_terms=True)
    assert draft.action == 71  # edge 17 [17,18], preserving the v39 lane
    assert draft.action in game.legal_actions()
    assert draft.principal_variation[0] == draft.action

    selected = choose_greedy_action(
        game,
        values,
        build_spending_guard=True,
        setup_draft_search=True,
        setup_draft_features=True,
        weights=WEIGHT_VECTOR,
    )
    assert selected == draft.action


def test_setup_draft_feature_terms_are_separately_optioned():
    game = _root()
    plain = choose_setup_draft_action(
        game, WEIGHT_VECTOR, preferred=69, feature_terms=False)
    strategic = choose_setup_draft_action(
        game, WEIGHT_VECTOR, preferred=69, feature_terms=True)
    assert plain.action == 69
    assert strategic.action == 71
