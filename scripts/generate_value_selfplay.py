#!/usr/bin/env python3
import argparse
import json

from catan_ranked.value_selfplay import (
    LinearValue, NumpyMLPValue, generate_games, validate_shard, write_shard,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--uniform-epsilon", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--max-plans", type=int, default=128)
    parser.add_argument("--max-actions", type=int, default=14)
    parser.add_argument("--chance-samples", type=int, default=4)
    parser.add_argument("--max-plies", type=int, default=8_000)
    parser.add_argument("--exploration-seed", type=int, default=0)
    parser.add_argument("--mlp")
    parser.add_argument("--mixed-linear", action="store_true")
    parser.add_argument("--play-only-values", action="store_true")
    args = parser.parse_args()
    value = NumpyMLPValue(args.mlp) if args.mlp else LinearValue()
    linear = LinearValue()
    matchups = ([(value, value), (value, linear), (linear, value), (linear, linear)]
                if args.mixed_linear else None)
    arrays, metadata = generate_games(
        games=args.games, seed_start=args.seed_start, epsilon=args.epsilon,
        top_k=args.top_k, max_plans=args.max_plans, max_actions=args.max_actions,
        chance_samples=args.chance_samples, max_plies=args.max_plies,
        value=value, exploration_seed=args.exploration_seed,
        matchup_cycle=matchups, play_only_values=args.play_only_values,
        uniform_epsilon=args.uniform_epsilon,
    )
    result = write_shard(args.output, arrays, metadata)
    validate_shard(args.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
