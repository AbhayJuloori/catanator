import pytest

pytest.importorskip("catan_ranked_fast")

from catan_ranked.value_arena import (
    choose_greedy_action, evaluate_sprt, pentanomial_llr, play_game, play_pair, run_sprt,
)


def test_pentanomial_statistic_reference_values():
    assert pentanomial_llr([0, 0, 10, 0, 0]) == pytest.approx(-1.210146615081, abs=1e-10)
    assert evaluate_sprt([0, 0, 100, 0, 0])["decision"] == "accept_h0"
    assert evaluate_sprt([0, 0, 0, 0, 100])["decision"] == "accept_h1"


def test_identical_greedy_values_are_exact_color_mirror_null():
    pair = play_pair(0, {"kind": "linear"}, {"kind": "linear"},
                     base_seed=2_145_000_000, max_plans=32, chance_samples=2)
    assert pair["category"] == 2
    assert pair["games"][0]["candidate_seat"] == 0
    assert pair["games"][1]["candidate_seat"] == 1
    assert pair["games"][0]["plies"] == pair["games"][1]["plies"]
    for game in pair["games"]:
        assert set(game["candidate_action_mix"]) == {
            "cities", "roads", "settlements", "dev_buys", "discards"}
        assert game["candidate_vp"] >= 0


def test_fixed_pair_mode_does_not_stop_at_sprt_bound():
    report = run_sprt({"kind": "linear"}, {"kind": "linear"},
                      max_pairs=3, stop_on_decision=False, max_plans=8,
                      chance_samples=1, base_seed=2_145_100_000)
    assert report["pairs"] == 3
    assert report["stop_on_decision"] is False
    assert report["games"] == 6
    assert report["mean_final_vp"]["candidate"] == report["mean_final_vp"]["baseline"]
    assert report["action_mix"]["candidate"] == report["action_mix"]["baseline"]
    assert report["decision_timing_ms"]["candidate"]["count"] > 0
    assert report["decision_timing_ms"]["candidate"]["median"] >= 0.0


def test_fast_champion_is_a_player_kind():
    game = play_game(5, 0, {"kind": "champion"}, {"kind": "linear"}, max_plies=1)
    assert game["plies"] == 1


def test_weighted_linear_is_an_arena_player_kind():
    import json
    from pathlib import Path

    from catan_ranked.value_residual_features import TERM_NAMES

    weights = json.loads(Path(
        "weights/tuned_consensus.json").read_text())["weights"]
    pair = play_pair(0, {"kind": "weighted_linear",
                         "weights": [weights[name] for name in TERM_NAMES]},
                     {"kind": "linear"}, base_seed=2_145_200_000,
                     max_plans=16, chance_samples=1)
    assert len(TERM_NAMES) == 24
    assert pair["category"] == 2


def test_endgame_dev_card_liquidation_is_default_off_and_bounded():
    class Position:
        current_player = 1
        actual_vp = [11, 10]

    game = Position()
    close = [(186, 0.805), (304, 0.800), (311, 0.799)]
    assert choose_greedy_action(game, close) == 186
    assert choose_greedy_action(
        game, close, endgame_dev_card_liquidation=True) == 304

    too_far = [(186, 0.811), (304, 0.800)]
    assert choose_greedy_action(
        game, too_far, endgame_dev_card_liquidation=True) == 186

    game.actual_vp[1] = 9
    assert choose_greedy_action(
        game, close, endgame_dev_card_liquidation=True) == 186


def test_pre_roll_knight_only_is_default_on_and_can_be_disabled():
    class Position:
        current_player = 0
        actual_vp = [4, 4]

    game = Position()
    ranked = [(311, 0.9), (305, 0.8), (304, 0.7), (331, 0.6)]
    assert choose_greedy_action(game, ranked) == 304
    assert choose_greedy_action(game, ranked, pre_roll_knight_only=False) == 311

    post_roll = [(311, 0.9), (304, 0.7), (186, 0.6)]
    assert choose_greedy_action(game, post_roll) == 311
