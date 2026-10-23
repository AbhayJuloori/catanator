import json

import numpy as np
import pytest

pytest.importorskip("catan_ranked_fast")

from catan_ranked.value_selfplay import (
    LinearValue,
    NumpyMLPValue,
    generate_games,
    validate_shard,
    write_shard,
)
from catan_ranked_fast import OBSERVATION_DIM


def _small_run():
    return generate_games(
        games=1,
        seed_start=900_000,
        epsilon=0.2,
        top_k=3,
        max_plans=24,
        max_actions=12,
        chance_samples=1,
        exploration_seed=19,
    )


def test_linear_selfplay_is_deterministic_complete_and_aligned():
    first, first_meta = _small_run()
    second, second_meta = _small_run()
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    assert first_meta["games_completed"] == second_meta["games_completed"] == 1
    assert first_meta["games_capped"] == second_meta["games_capped"] == 0
    rows = first["observation"].shape[0]
    assert first["observation"].shape == (rows, OBSERVATION_DIM)
    assert set(first["game_id"]) == {900_000}
    assert set(first["outcome"]) == {0.0, 1.0}
    assert np.all(first["vp_margin"][first["outcome"] == 1.0] > 0)
    assert np.all(first["vp_margin"][first["outcome"] == 0.0] < 0)


def test_shard_round_trip_and_hash_rejection(tmp_path):
    arrays, metadata = _small_run()
    path = tmp_path / "shard.npz"
    written = write_shard(path, arrays, metadata)
    assert validate_shard(path) == written
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="hash"):
        validate_shard(path)


def test_numpy_mlp_checkpoint_contract_and_forward(tmp_path):
    path = tmp_path / "mlp.npz"
    np.savez(
        path,
        w1=np.zeros((OBSERVATION_DIM, 512), np.float32),
        b1=np.zeros(512, np.float32),
        w2=np.zeros((512, 512), np.float32),
        b2=np.zeros(512, np.float32),
        w_win=np.zeros(512, np.float32),
        b_win=np.zeros(1, np.float32),
    )
    model = NumpyMLPValue(path)

    class Search:
        @staticmethod
        def observations():
            return np.zeros((3, OBSERVATION_DIM), np.float32)

    assert model.leaf_values(Search()) == [0.5, 0.5, 0.5]
    metadata = json.loads(json.dumps({"backend": model.name}))
    assert metadata == {"backend": "numpy_mlp"}


def test_mixed_play_only_generation_records_policy_contract(tmp_path):
    path = tmp_path / "mlp.npz"
    np.savez_compressed(
        path,
        w1=np.zeros((OBSERVATION_DIM, 512), np.float32),
        b1=np.zeros(512, np.float32),
        w2=np.zeros((512, 512), np.float32),
        b2=np.zeros(512, np.float32),
        w_win=np.zeros(512, np.float32),
        b_win=np.zeros(1, np.float32),
    )
    learned, linear = NumpyMLPValue(path), LinearValue()
    _rows, metadata = generate_games(
        games=4, seed_start=994_000, max_plans=8, chance_samples=1,
        epsilon=0.25, top_k=8, uniform_epsilon=0.05,
        matchup_cycle=[(learned, learned), (learned, linear),
                       (linear, learned), (linear, linear)],
        play_only_values=True)
    assert metadata["games_completed"] == 4
    assert metadata["uniform_epsilon"] == 0.05
    assert metadata["play_only_values"] is True
    assert metadata["matchups"] == [["numpy_mlp", "numpy_mlp"],
                                    ["numpy_mlp", "linear"],
                                    ["linear", "numpy_mlp"],
                                    ["linear", "linear"]]
