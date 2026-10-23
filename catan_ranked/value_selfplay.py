"""Fast-engine afterstate self-play and versioned training shards."""
from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np


SHARD_SCHEMA = "value_learning.selfplay/1"
OBSERVATION_VERSION = "value-afterstate-observer/1"
OBSERVATION_DIM = 1193
END_TURN = 186


class LinearValue:
    name = "linear"

    @staticmethod
    def leaf_values(search):
        return search.linear_values()

    @staticmethod
    def opponent_leaf_values(search):
        return search.opponent_linear_values()


class WeightedLinearValue:
    name = "weighted_linear"

    def __init__(self, weights):
        if len(weights) != 24 or not np.isfinite(weights).all():
            raise ValueError("weighted linear value needs 24 finite weights")
        self.weights = tuple(float(value) for value in weights)

    def leaf_values(self, search):
        return search.weighted_linear_values(self.weights)

    def opponent_leaf_values(self, search):
        return search.opponent_weighted_linear_values(self.weights)


class NumpyMLPValue:
    """Strict two-hidden-layer ReLU checkpoint used by generation self-play."""

    name = "numpy_mlp"

    def __init__(self, checkpoint):
        payload = np.load(checkpoint)
        required = ("w1", "b1", "w2", "b2", "w_win", "b_win")
        if set(payload.files) != set(required):
            raise ValueError(f"MLP checkpoint must contain exactly {required}")
        self.arrays = {key: np.asarray(payload[key], dtype=np.float32) for key in required}
        expected = {
            "w1": (OBSERVATION_DIM, 512), "b1": (512,),
            "w2": (512, 512), "b2": (512,),
            "w_win": (512,), "b_win": (1,),
        }
        for key, shape in expected.items():
            if self.arrays[key].shape != shape:
                raise ValueError(f"{key} has shape {self.arrays[key].shape}, expected {shape}")

    def _values(self, observations):
        x = np.asarray(observations, dtype=np.float32)
        a = self.arrays
        hidden = np.maximum(x @ a["w1"] + a["b1"], 0.0)
        hidden = np.maximum(hidden @ a["w2"] + a["b2"], 0.0)
        logits = hidden @ a["w_win"] + a["b_win"][0]
        values = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        return values.astype(np.float64).tolist()

    def leaf_values(self, search):
        return self._values(search.observations())

    def opponent_leaf_values(self, search):
        return self._values(search.opponent_observations())


class NumpyResidualValue:
    """Calibrated frozen linear value plus a strict zero-init-compatible residual MLP."""

    name = "numpy_residual"

    def __init__(self, checkpoint):
        from catan_ranked.value_residual_features import RESIDUAL_FEATURE_DIM
        payload = np.load(checkpoint)
        required = ("w1", "b1", "w2", "b2", "w_residual", "b_residual", "calibration")
        if set(payload.files) != set(required):
            raise ValueError(f"residual checkpoint must contain exactly {required}")
        self.arrays = {key: np.asarray(payload[key], dtype=np.float32) for key in required}
        expected = {
            "w1": (RESIDUAL_FEATURE_DIM, 512), "b1": (512,),
            "w2": (512, 512), "b2": (512,),
            "w_residual": (512,), "b_residual": (1,), "calibration": (2,),
        }
        for key, shape in expected.items():
            if self.arrays[key].shape != shape:
                raise ValueError(f"{key} has shape {self.arrays[key].shape}, expected {shape}")

    def _values(self, observations):
        from catan_ranked.value_residual_features import residual_features
        observations = np.asarray(observations, dtype=np.float32)
        x, linear_value = residual_features(observations)
        a = self.arrays
        hidden = np.maximum(x @ a["w1"] + a["b1"], 0.0)
        hidden = np.maximum(hidden @ a["w2"] + a["b2"], 0.0)
        logits = (a["calibration"][0] * linear_value + a["calibration"][1]
                  + hidden @ a["w_residual"] + a["b_residual"][0])
        values = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        return values.astype(np.float64).tolist()

    def leaf_values(self, search):
        return self._values(search.observations())

    def opponent_leaf_values(self, search):
        return self._values(search.opponent_observations())


def _choose_action(search, values, rng, epsilon, top_k, uniform_epsilon=0.0):
    ranked = sorted(search.action_values(values), key=lambda item: (-item[1], item[0]))
    if not ranked:
        raise RuntimeError("turn search returned no backed action")
    if len(ranked) > 1:
        draw = rng.random()
        if draw < uniform_epsilon:
            return rng.choice(ranked)[0]
        if draw < uniform_epsilon + epsilon:
            return rng.choice(ranked[: min(top_k, len(ranked))])[0]
    return ranked[0][0]


def generate_games(*, games, seed_start, epsilon=0.1, top_k=4, max_plans=128,
                   max_actions=14, chance_samples=4, max_plies=8_000,
                   value=None, exploration_seed=0, matchup_cycle=None,
                   play_only_values=False, uniform_epsilon=0.0):
    from catan_ranked_fast import Game

    if (games <= 0 or epsilon < 0.0 or uniform_epsilon < 0.0
            or epsilon + uniform_epsilon > 1.0 or top_k <= 0):
        raise ValueError("games/top_k must be positive and epsilon must be in [0, 1]")
    value = value or LinearValue()
    matchup_cycle = matchup_cycle or [(value, value)]
    if not matchup_cycle or any(len(matchup) != 2 for matchup in matchup_cycle):
        raise ValueError("matchup_cycle must contain evaluator pairs")
    linear = LinearValue()
    rng = random.Random(exploration_seed)
    rows = []
    completed = capped = total_actions = 0
    started = time.process_time()
    for offset in range(games):
        seed = seed_start + offset
        game = Game(seed)
        player_values = matchup_cycle[offset % len(matchup_cycle)]
        game_rows = []
        for action_ply in range(max_plies):
            actor = game.current_player
            search = game.turn_search(max_plans, max_actions=max_actions,
                                      chance_samples=chance_samples)
            selected_value = player_values[actor]
            if play_only_values:
                prompt = int(np.argmax(np.asarray(game.observation(actor))[1179:1184]))
                if prompt != 2:
                    selected_value = linear
            leaf_values = selected_value.leaf_values(search)
            action = _choose_action(
                search, leaf_values, rng, epsilon, top_k, uniform_epsilon)
            game.apply(action)
            total_actions += 1
            terminal = game.winner >= 0
            if action == END_TURN or terminal:
                # Avoid duplicating a terminal END_TURN afterstate.
                if not game_rows or game_rows[-1][1] != action_ply:
                    game_rows.append((np.asarray(game.observation(actor), np.float32),
                                      action_ply, actor))
            if terminal:
                scores = game.actual_vp
                winner = game.winner
                for observation, ply, player in game_rows:
                    rows.append((observation, seed, ply, player,
                                 float(player == winner),
                                 scores[player] - scores[1 - player]))
                completed += 1
                break
        else:
            capped += 1
    cpu_seconds = time.process_time() - started
    arrays = {
        "observation": np.asarray([row[0] for row in rows], dtype=np.float32),
        "game_id": np.asarray([row[1] for row in rows], dtype=np.int64),
        "ply": np.asarray([row[2] for row in rows], dtype=np.int32),
        "player": np.asarray([row[3] for row in rows], dtype=np.uint8),
        "outcome": np.asarray([row[4] for row in rows], dtype=np.float32),
        "vp_margin": np.asarray([row[5] for row in rows], dtype=np.int8),
    }
    if arrays["observation"].size == 0:
        arrays["observation"] = np.empty((0, OBSERVATION_DIM), dtype=np.float32)
    metadata = {
        "schema": SHARD_SCHEMA,
        "observation_version": OBSERVATION_VERSION,
        "value": value.name,
        "matchups": [[left.name, right.name] for left, right in matchup_cycle],
        "play_only_values": play_only_values,
        "games_requested": games,
        "games_completed": completed,
        "games_capped": capped,
        "seed_start": seed_start,
        "epsilon": epsilon,
        "uniform_epsilon": uniform_epsilon,
        "top_k": top_k,
        "max_plans": max_plans,
        "max_actions": max_actions,
        "chance_samples": chance_samples,
        "max_plies": max_plies,
        "exploration_seed": exploration_seed,
        "rows": len(rows),
        "actions": total_actions,
        "cpu_seconds": cpu_seconds,
        "games_per_cpu_hour": completed / cpu_seconds * 3600.0 if cpu_seconds else None,
    }
    return arrays, metadata


def write_shard(path, arrays, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = {**metadata, "sha256": digest}
    meta_path = path.with_suffix(".json")
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def validate_shard(path):
    path = Path(path)
    metadata = json.loads(path.with_suffix(".json").read_text())
    if metadata.get("schema") != SHARD_SCHEMA:
        raise ValueError("wrong self-play shard schema")
    if metadata.get("observation_version") != OBSERVATION_VERSION:
        raise ValueError("wrong self-play observation version")
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata.get("sha256"):
        raise ValueError("self-play shard hash mismatch")
    arrays = np.load(path)
    required = {"observation", "game_id", "ply", "player", "outcome", "vp_margin"}
    if set(arrays.files) != required:
        raise ValueError("self-play shard arrays do not match schema")
    rows = arrays["observation"].shape[0]
    if arrays["observation"].shape != (rows, OBSERVATION_DIM):
        raise ValueError("invalid observation matrix shape")
    if any(arrays[key].shape != (rows,) for key in required - {"observation"}):
        raise ValueError("self-play label arrays are misaligned")
    if rows != metadata.get("rows") or not np.isfinite(arrays["observation"]).all():
        raise ValueError("self-play shard row count or values are invalid")
    return metadata
