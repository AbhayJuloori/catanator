#!/usr/bin/env python3
"""Run a fast paired value-planner SPRT."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from catan_ranked.value_arena import run_sprt


def value_spec(checkpoint, play_only=False):
    if checkpoint in {"linear", "champion"}:
        return {"kind": checkpoint}
    if checkpoint.startswith("champion_d"):
        return {"kind": "champion", "depth": int(checkpoint.removeprefix("champion_d"))}
    checkpoint = Path(checkpoint).resolve()
    if checkpoint.suffix == ".json":
        from catan_ranked.value_residual_features import TERM_NAMES
        payload = json.loads(checkpoint.read_text())
        if payload.get("kind"):
            return payload
        weights = payload["weights"]
        return {"kind": "weighted_linear",
                "weights": [weights[name] for name in TERM_NAMES]}
    with np.load(checkpoint) as payload:
        residual = "calibration" in payload.files
    kind = "residual" if residual else "mlp"
    return {"kind": f"play_only_{kind}" if play_only else kind,
            "checkpoint": str(checkpoint)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, help="linear, champion, or checkpoint.npz")
    parser.add_argument("--baseline", required=True, help="linear, champion, or checkpoint.npz")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-pairs", type=int, default=4000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--elo0", type=float, default=0.0)
    parser.add_argument("--elo1", type=float, default=40.0)
    parser.add_argument("--base-seed", type=int, default=2_140_000_000)
    parser.add_argument("--candidate-play-only", action="store_true")
    parser.add_argument("--baseline-play-only", action="store_true")
    parser.add_argument("--candidate-opponent-lookahead", action="store_true")
    parser.add_argument("--baseline-opponent-lookahead", action="store_true")
    parser.add_argument("--candidate-endgame-dev-card-liquidation", action="store_true")
    parser.add_argument("--baseline-endgame-dev-card-liquidation", action="store_true")
    parser.add_argument("--candidate-build-spending-guard", action="store_true")
    parser.add_argument("--baseline-build-spending-guard", action="store_true")
    parser.add_argument("--candidate-setup-resource-coverage", action="store_true")
    parser.add_argument("--baseline-setup-resource-coverage", action="store_true")
    parser.add_argument("--candidate-setup-draft-search", action="store_true")
    parser.add_argument("--baseline-setup-draft-search", action="store_true")
    parser.add_argument("--candidate-setup-draft-features", action="store_true")
    parser.add_argument("--baseline-setup-draft-features", action="store_true")
    parser.add_argument("--candidate-opponent-model", choices=("min", "self_max", "roll_only"),
                        default="min")
    parser.add_argument("--baseline-opponent-model", choices=("min", "self_max", "roll_only"),
                        default="min")
    parser.add_argument("--max-plans", type=int, default=128)
    parser.add_argument("--max-actions", type=int, default=14)
    parser.add_argument("--chance-samples", type=int, default=4)
    parser.add_argument("--opponent-plans", type=int, default=12)
    parser.add_argument("--opponent-max-actions", type=int, default=10)
    parser.add_argument("--fixed-pairs", action="store_true")
    args = parser.parse_args()
    candidate = value_spec(args.candidate, args.candidate_play_only)
    baseline = value_spec(args.baseline, args.baseline_play_only)
    if args.candidate_opponent_lookahead:
        candidate.update(opponent_lookahead=True,
                         opponent_model=args.candidate_opponent_model,
                         opponent_plans=args.opponent_plans,
                         opponent_max_actions=args.opponent_max_actions)
    if args.candidate_endgame_dev_card_liquidation:
        candidate["endgame_dev_card_liquidation"] = True
    if args.candidate_build_spending_guard:
        candidate["build_spending_guard"] = True
    if args.candidate_setup_resource_coverage:
        candidate["setup_resource_coverage"] = True
    if args.candidate_setup_draft_search:
        candidate["setup_draft_search"] = True
    if args.candidate_setup_draft_features:
        candidate["setup_draft_features"] = True
    if args.baseline_opponent_lookahead:
        baseline.update(opponent_lookahead=True,
                        opponent_model=args.baseline_opponent_model,
                         opponent_plans=args.opponent_plans,
                         opponent_max_actions=args.opponent_max_actions)
    if args.baseline_endgame_dev_card_liquidation:
        baseline["endgame_dev_card_liquidation"] = True
    if args.baseline_build_spending_guard:
        baseline["build_spending_guard"] = True
    if args.baseline_setup_resource_coverage:
        baseline["setup_resource_coverage"] = True
    if args.baseline_setup_draft_search:
        baseline["setup_draft_search"] = True
    if args.baseline_setup_draft_features:
        baseline["setup_draft_features"] = True
    report = run_sprt(
        candidate, baseline,
        max_pairs=args.max_pairs, workers=args.workers,
        elo0=args.elo0, elo1=args.elo1, base_seed=args.base_seed,
        stop_on_decision=not args.fixed_pairs,
        max_plans=args.max_plans, max_actions=args.max_actions,
        chance_samples=args.chance_samples,
        opponent_plans=args.opponent_plans,
        opponent_max_actions=args.opponent_max_actions)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
