"""Derived public features for the residual value model."""
from __future__ import annotations

import numpy as np

from catan_ranked.value_selfplay import OBSERVATION_DIM


TILE_VERTICES = np.asarray([
    [0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 2, 1], [2, 9, 10, 11, 12, 3],
    [4, 3, 12, 13, 14, 15], [16, 5, 4, 15, 17, 18], [19, 20, 0, 5, 16, 21],
    [22, 23, 6, 1, 0, 20], [24, 25, 26, 27, 8, 7], [8, 27, 28, 29, 10, 9],
    [10, 29, 30, 31, 32, 11], [12, 11, 32, 33, 34, 13],
    [14, 13, 34, 35, 36, 37], [17, 15, 14, 37, 38, 39],
    [40, 18, 17, 39, 41, 42], [43, 21, 16, 18, 40, 44],
    [45, 46, 19, 21, 43, 47], [48, 49, 22, 20, 19, 46],
    [50, 51, 52, 23, 22, 49], [52, 53, 24, 7, 6, 23],
], dtype=np.int64)


def _topology():
    edges = set()
    for vertices in TILE_VERTICES:
        for index, left in enumerate(vertices):
            edges.add(tuple(sorted((int(left), int(vertices[(index + 1) % 6])))))
    edges = np.asarray(sorted(edges, key=str), dtype=np.int64)
    neighbors = [[] for _ in range(54)]
    for left, right in edges:
        neighbors[left].append(int(right))
        neighbors[right].append(int(left))
    return edges, tuple(tuple(values) for values in neighbors)


EDGE_VERTICES, VERTEX_NEIGHBORS = _topology()
assert EDGE_VERTICES.shape == (72, 2)

TERM_NAMES = (
    "vp_progress", "vp_lead", "production", "opp_production", "prod_diversity",
    "buildable", "longest_road", "has_longest_road", "army", "has_largest_army",
    "hand_resources", "hand_devs", "res_WOOD", "res_BRICK", "res_SHEEP",
    "res_WHEAT", "res_ORE", "scale", "synergy_expansion", "synergy_city",
    "bonus_vp_reach", "bonus_vp_denial", "expansion_tempo", "contested_share",
)
RESIDUAL_FEATURE_DIM = OBSERVATION_DIM + len(TERM_NAMES) + 10

VP_PROGRESS = 1.16465
VP_LEAD = 0.96372
PRODUCTION = 0.94685
OPP_PRODUCTION = 1.23796
PROD_DIVERSITY = 0.70161
BUILDABLE = 0.2981
LONGEST_ROAD = 0.50993
HAS_LONGEST_ROAD = 0.46285
ARMY = 0.27988
HAS_LARGEST_ARMY = 0.258
HAND_RESOURCES = 0.2411
HAND_DEVS = 0.20713
RESOURCE_WEIGHTS = np.asarray([1.09979, 1.17024, 0.99872, 1.75782, 1.80343])
SCALE = 0.45319
SYNERGY_EXPANSION = 0.41919
SYNERGY_CITY = 0.27732
BONUS_VP_REACH = 0.01199
BONUS_VP_DENIAL = -0.03921
EXPANSION_TEMPO = 0.08352
CONTESTED_SHARE = -0.12056
DEV_MAX = np.asarray([14.0, 2.0, 2.0, 2.0, 5.0])


def _bonus(production, road_length, has_road, played_knights, has_army):
    road_fuel = np.minimum(production[:, 0], production[:, 1])
    road = np.minimum(road_length / 5.0, 1.0) * (
        0.5 + 0.5 * np.minimum(road_fuel / 0.5, 1.0))
    road = np.where(has_road, 1.0, road)
    dev_fuel = np.minimum.reduce((production[:, 4], production[:, 2], production[:, 3]))
    army = np.minimum(played_knights / 3.0, 1.0) * (
        0.5 + 0.5 * np.minimum(dev_fuel / 0.35, 1.0))
    army = np.where(has_army, 1.0, army)
    return np.maximum(road, army)


def residual_features(observations):
    """Return ``(obs + 24 terms + own/opp production, frozen_linear_value)``."""
    observations = np.asarray(observations, dtype=np.float32)
    if observations.ndim != 2 or observations.shape[1] != OBSERVATION_DIM:
        raise ValueError(f"expected [N,{OBSERVATION_DIM}] observations")
    if not np.isfinite(observations).all():
        raise ValueError("observations must be finite")
    rows = len(observations)
    tiles = observations[:, :152].reshape(rows, 19, 8)
    nodes = observations[:, 152:800].reshape(rows, 54, 12)
    edges = observations[:, 800:1016].reshape(rows, 72, 3)
    own = observations[:, 1016:1041]
    opp = observations[:, 1041:1066]

    resources = tiles[:, :, :6].argmax(axis=2)
    probability = tiles[:, :, 6].astype(np.float64) * (5.0 / 36.0)
    probability = np.where(tiles[:, :, 7] > 0.5, 0.0, probability)
    occupancy = nodes[:, :, :5].argmax(axis=2)
    production = np.zeros((rows, 2, 5), dtype=np.float64)
    for tile, vertices in enumerate(TILE_VERTICES):
        tile_occupancy = occupancy[:, vertices]
        multipliers = (
            (tile_occupancy == 1).sum(axis=1) + 2 * (tile_occupancy == 2).sum(axis=1),
            (tile_occupancy == 3).sum(axis=1) + 2 * (tile_occupancy == 4).sum(axis=1),
        )
        for resource in range(5):
            selected = resources[:, tile] == resource
            for player in range(2):
                production[:, player, resource] += np.where(
                    selected, probability[:, tile] * multipliers[player], 0.0)

    occupied = occupancy != 0
    edge_owner = edges.argmax(axis=2)
    connected = np.zeros((rows, 2, 54), dtype=bool)
    for edge, (left, right) in enumerate(EDGE_VERTICES):
        for player in range(2):
            selected = edge_owner[:, edge] == player + 1
            connected[:, player, left] |= selected
            connected[:, player, right] |= selected
    neighbor_occupied = np.zeros_like(occupied)
    for vertex, neighbors in enumerate(VERTEX_NEIGHBORS):
        neighbor_occupied[:, vertex] = occupied[:, neighbors].any(axis=1)
    buildable = connected & ~occupied[:, None, :] & ~neighbor_occupied[:, None, :]

    my_vp = own[:, 12].astype(np.float64) * 15.0 + own[:, 9] * DEV_MAX[4]
    opp_vp = opp[:, 12].astype(np.float64) * 15.0
    my_prod, opp_prod = production[:, 0], production[:, 1]
    my_weighted = my_prod @ RESOURCE_WEIGHTS
    opp_weighted = opp_prod @ RESOURCE_WEIGHTS
    diversity = (my_prod > 1e-6).sum(axis=1) / 5.0
    my_buildable = buildable[:, 0].sum(axis=1)
    opp_buildable = buildable[:, 1].sum(axis=1)
    contested = (buildable[:, 0] & buildable[:, 1]).sum(axis=1) / np.maximum(my_buildable, 1)
    contested = np.where(my_buildable == 0, 0.0, contested)
    my_road = own[:, 13].astype(np.float64) * 15.0
    opp_road = opp[:, 13].astype(np.float64) * 15.0
    my_knights = own[:, 11].astype(np.float64) * 14.0
    opp_knights = opp[:, 11].astype(np.float64) * 14.0
    my_has_road, opp_has_road = own[:, 14] > 0.5, opp[:, 14] > 0.5
    my_has_army, opp_has_army = own[:, 15] > 0.5, opp[:, 15] > 0.5
    my_bonus = _bonus(my_prod, my_road, my_has_road, my_knights, my_has_army)
    opp_bonus = _bonus(opp_prod, opp_road, opp_has_road, opp_knights, opp_has_army)
    cards = own[:, :5].astype(np.float64) * 19.0
    turns = np.zeros(rows, dtype=np.float64)
    for resource in range(4):
        deficit = np.maximum(0.0, 1.0 - cards[:, resource])
        rate = my_prod[:, resource]
        turns = np.maximum(turns, np.where(rate > 1e-6, deficit / np.maximum(rate, 1e-6), 25.0))
    expansion_tempo = 1.0 / (1.0 + turns)
    hand_resources = np.minimum(cards.sum(axis=1), 9.0) / 9.0
    hand_devs = (own[:, 5:10].astype(np.float64) * DEV_MAX).sum(axis=1)

    score = (
        VP_PROGRESS * (my_vp / 15.0)
        + VP_LEAD * ((my_vp - opp_vp) / 15.0)
        + PRODUCTION * my_weighted - OPP_PRODUCTION * opp_weighted
        + PROD_DIVERSITY * diversity + BUILDABLE * (my_buildable / 10.0)
        + LONGEST_ROAD * (my_road / 10.0) + HAS_LONGEST_ROAD * my_has_road
        + ARMY * (my_knights / 3.0) + HAS_LARGEST_ARMY * my_has_army
        + HAND_RESOURCES * hand_resources + HAND_DEVS * hand_devs
        + SYNERGY_EXPANSION * np.minimum(my_prod[:, 0], my_prod[:, 1])
        + SYNERGY_CITY * np.minimum(my_prod[:, 4], my_prod[:, 3])
        + BONUS_VP_REACH * my_bonus - BONUS_VP_DENIAL * opp_bonus
        + EXPANSION_TEMPO * expansion_tempo + CONTESTED_SHARE * contested
    )
    linear_value = np.tanh(SCALE * score)
    linear_value = np.where(my_vp >= 15.0 - 1e-6, 1.0, linear_value)
    resource_terms = PRODUCTION * my_prod - OPP_PRODUCTION * opp_prod
    terms = np.column_stack((
        my_vp / 15.0, (my_vp - opp_vp) / 15.0, my_weighted, -opp_weighted,
        diversity, my_buildable / 10.0, my_road / 10.0, my_has_road,
        my_knights / 3.0, my_has_army, hand_resources, hand_devs,
        resource_terms, score, np.minimum(my_prod[:, 0], my_prod[:, 1]),
        np.minimum(my_prod[:, 4], my_prod[:, 3]), my_bonus, -opp_bonus,
        expansion_tempo, contested,
    )).astype(np.float32)
    if terms.shape != (rows, len(TERM_NAMES)):
        raise AssertionError(f"invalid term matrix {terms.shape}")
    features = np.concatenate((observations, terms, production.reshape(rows, 10)), axis=1)
    return features.astype(np.float32, copy=False), linear_value.astype(np.float32)
