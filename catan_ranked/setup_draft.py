"""Bounded adversarial draft search for the fast-engine ranked opening."""
from __future__ import annotations

from dataclasses import dataclass
import math

from catan_ranked.value_residual_features import (
    EDGE_VERTICES,
    TILE_VERTICES,
    VERTEX_NEIGHBORS,
)


_DEV_MAX = (14.0, 2.0, 2.0, 2.0, 5.0)


class DraftBudgetExceeded(RuntimeError):
    """Raised instead of returning a silently partial opening tree."""


@dataclass(frozen=True)
class DraftResult:
    action: int
    value: float
    leaves: int
    principal_variation: tuple[int, ...]
    root_values: tuple[tuple[int, float], ...]


def _observation(game, observer):
    return game.observation(observer)


def _is_initial(game, observer):
    return _observation(game, observer)[-9] > 0.5


def _occupancy(observation):
    nodes = observation[152:800]
    return [
        max(range(5), key=lambda category: nodes[vertex * 12 + category])
        for vertex in range(54)
    ]


def _production(observation, own):
    tiles = observation[:152]
    occupancy = _occupancy(observation)
    production = [0.0] * 5
    settlement, city = ((1, 2) if own else (3, 4))
    for tile, vertices in enumerate(TILE_VERTICES):
        base = tile * 8
        resource = max(range(6), key=lambda category: tiles[base + category])
        if resource >= 5 or tiles[base + 7] > 0.5:
            continue
        probability = float(tiles[base + 6]) * 5.0 / 36.0
        for vertex in vertices:
            building = occupancy[int(vertex)]
            if building == settlement:
                production[resource] += probability
            elif building == city:
                production[resource] += 2.0 * probability
    return production


def _buildable_sites(observation, own):
    edges = observation[800:1016]
    occupancy = _occupancy(observation)
    owner = 1 if own else 2
    connected = set()
    for edge, vertices in enumerate(EDGE_VERTICES):
        if edges[edge * 3 + owner] > 0.5:
            connected.update(map(int, vertices))
    return {
        vertex for vertex in connected
        if occupancy[vertex] == 0
        and all(occupancy[int(neighbor)] == 0
                for neighbor in VERTEX_NEIGHBORS[vertex])
    }


def _one_road_sites(observation, own):
    """Legal settlement targets reachable after adding one road to the setup lane."""
    edges = observation[800:1016]
    occupancy = _occupancy(observation)
    owner = 1 if own else 2
    blocking_buildings = (3, 4) if own else (1, 2)
    connected = set()
    owned_edges = set()
    edge_by_vertices = {}
    for edge, vertices in enumerate(EDGE_VERTICES):
        pair = tuple(sorted(map(int, vertices)))
        edge_by_vertices[pair] = edge
        if edges[edge * 3 + owner] > 0.5:
            owned_edges.add(edge)
            connected.update(map(int, vertices))
    targets = set()
    for source in connected:
        if occupancy[source] in blocking_buildings:
            continue
        for target in map(int, VERTEX_NEIGHBORS[source]):
            edge = edge_by_vertices[tuple(sorted((source, target)))]
            if edge in owned_edges or edges[edge * 3] < 0.5:
                continue
            if occupancy[target] != 0:
                continue
            if all(occupancy[int(neighbor)] == 0
                   for neighbor in VERTEX_NEIGHBORS[target]):
                targets.add(target)
    return targets


def _site_production(observation, vertex):
    tiles = observation[:152]
    score = 0.0
    for tile, vertices in enumerate(TILE_VERTICES):
        base = tile * 8
        resource = max(range(6), key=lambda category: tiles[base + category])
        if vertex not in vertices or resource >= 5:
            continue
        score += float(tiles[base + 6]) * 5.0 / 36.0
    return score


def _owned_ports(observation, own):
    block = 1016 if own else 1041
    return {
        resource for resource in range(5)
        if observation[block + 20 + resource] > 0.5
    }


def _reachable_port_fit(observation, production, sites):
    nodes = observation[152:800]
    fit = 0.0
    total = max(sum(production), 1e-9)
    for vertex in sites:
        base = vertex * 12 + 5
        port = max(range(7), key=lambda category: nodes[base + category])
        if port == 0:
            continue
        fit = max(fit, 0.5 if port == 1 else production[port - 2] / total)
    return fit


def _bonus_reach(observation, own, production):
    block = 1016 if own else 1041
    road_fuel = min(production[0], production[1])
    road = min(float(observation[block + 13]) * 15.0 / 5.0, 1.0)
    road *= 0.5 + 0.5 * min(road_fuel / 0.5, 1.0)
    if observation[block + 14] > 0.5:
        road = 1.0
    dev_fuel = min(production[4], production[2], production[3])
    army = min(float(observation[block + 11]) * 14.0 / 3.0, 1.0)
    army *= 0.5 + 0.5 * min(dev_fuel / 0.35, 1.0)
    if observation[block + 15] > 0.5:
        army = 1.0
    return max(road, army)


def _expansion_tempo(observation, production):
    turns = 0.0
    for resource in range(4):
        cards = float(observation[1016 + resource]) * 19.0
        deficit = max(0.0, 1.0 - cards)
        rate = production[resource]
        turns = max(turns, deficit / rate if rate > 1e-6 else 25.0)
    return 1.0 / (1.0 + turns)


def _position_metrics(game, observer):
    observation = _observation(game, observer)
    production = _production(observation, True)
    opp_production = _production(observation, False)
    sites = _buildable_sites(observation, True)
    opp_sites = _buildable_sites(observation, False)
    lane_sites = _one_road_sites(observation, True)
    owned_ports = _owned_ports(observation, True)
    access = [production[index] > 1e-9 or index in owned_ports for index in range(5)]
    return observation, production, opp_production, sites, opp_sites, access, {
        "total_production": sum(production),
        "coverage": sum(access) / 5.0,
        "expansion": min(production[0], production[1]),
        "ows": min(production[4], production[3], production[2]),
        "denial": sum(value <= 1e-9 for value in opp_production) / 5.0,
        "site_quality": max((_site_production(observation, vertex) for vertex in sites), default=0.0),
        "port_fit": _reachable_port_fit(observation, production, sites),
        "lane_quality": max((_site_production(observation, vertex) for vertex in lane_sites),
                            default=0.0),
        "lane_port_fit": _reachable_port_fit(
            observation, production, lane_sites),
    }


def setup_position_value(game, observer, weights, *, feature_terms):
    """Replica-a's raw state value plus optional opening-only strategic terms."""
    (observation, production, opp_production, sites, opp_sites, access,
     metrics) = _position_metrics(game, observer)
    own = observation[1016:1041]
    opp = observation[1041:1066]
    my_vp = float(own[12]) * 15.0 + float(own[9]) * _DEV_MAX[4]
    opp_vp = float(opp[12]) * 15.0
    my_weighted = sum(value * weights[12 + index]
                      for index, value in enumerate(production))
    opp_weighted = sum(value * weights[12 + index]
                       for index, value in enumerate(opp_production))
    contested = len(sites & opp_sites) / len(sites) if sites else 0.0
    score = (
        weights[0] * my_vp / 15.0
        + weights[1] * (my_vp - opp_vp) / 15.0
        + weights[2] * my_weighted - weights[3] * opp_weighted
        + weights[4] * sum(value > 1e-6 for value in production) / 5.0
        + weights[5] * len(sites) / 10.0
        + weights[6] * (float(own[13]) * 15.0) / 10.0
        + weights[7] * float(own[14] > 0.5)
        + weights[8] * (float(own[11]) * 14.0) / 3.0
        + weights[9] * float(own[15] > 0.5)
        + weights[10] * min(sum(float(own[index]) * 19.0 for index in range(5)), 9) / 9.0
        + weights[11] * sum(float(own[5 + index]) * _DEV_MAX[index]
                            for index in range(5))
        + weights[18] * min(production[0], production[1])
        + weights[19] * min(production[4], production[3])
        + weights[20] * _bonus_reach(observation, True, production)
        - weights[21] * _bonus_reach(observation, False, opp_production)
        + weights[22] * _expansion_tempo(observation, production)
        + weights[23] * contested
    )
    if feature_terms:
        # Opening hypotheses are deliberately modest, continuous value terms rather than
        # mandatory rules.  The post-reply site term is what makes a road lane lose value
        # when the opponent can legally extinguish it during the remaining snake draft.
        score += (
            0.10 * metrics["coverage"]
            + 0.45 * metrics["expansion"]
            + 0.30 * metrics["ows"]
            + 0.06 * metrics["denial"]
            + 0.80 * metrics["site_quality"]
            + 0.08 * metrics["port_fit"]
            + 0.90 * metrics["lane_quality"]
            + 0.10 * metrics["lane_port_fit"]
        )
    return math.tanh(weights[17] * score)


def _settlement_candidates(game, observer, weights, limit, preferred, feature_terms):
    legal = sorted(action for action in map(int, game.legal_actions()) if 126 <= action < 180)
    if len(legal) <= limit:
        return legal
    actor = int(game.current_player)
    rows = []
    for action in legal:
        child = game.clone_game()
        child.apply(action)
        metrics = _position_metrics(child, actor)[-1]
        rows.append((action, setup_position_value(
            child, actor, weights, feature_terms=feature_terms), metrics))
    selected = []
    if preferred in legal:
        selected.append(preferred)
    criteria = (
        lambda row: row[1],
        lambda row: row[2]["total_production"],
        lambda row: row[2]["expansion"],
        lambda row: row[2]["ows"],
        lambda row: row[2]["port_fit"],
        lambda row: row[2]["lane_quality"],
        lambda row: row[2]["coverage"],
    )
    for criterion in criteria:
        action = max(rows, key=lambda row: (criterion(row), row[1], -row[0]))[0]
        if action not in selected:
            selected.append(action)
        if len(selected) >= limit:
            break
    for action, _value, _metrics in sorted(rows, key=lambda row: (-row[1], row[0])):
        if action not in selected:
            selected.append(action)
        if len(selected) >= limit:
            break
    return selected


def choose_setup_draft_action(game, weights, *, preferred=None, settlement_limit=6,
                              max_leaves=30_000, feature_terms=True):
    """Minimax every retained settlement and every road through setup completion."""
    if settlement_limit < 1 or max_leaves < 1:
        raise ValueError("settlement_limit and max_leaves must be positive")
    if not _is_initial(game, int(game.current_player)):
        raise ValueError("setup draft search called after initial placement")
    observer = int(game.current_player)
    leaves = 0

    def candidates(position, root=False):
        legal = list(map(int, position.legal_actions()))
        if legal and all(126 <= action < 180 for action in legal):
            return _settlement_candidates(
                position, observer, weights, settlement_limit,
                preferred if root else None, feature_terms)
        return sorted(legal)

    def visit(position, alpha, beta):
        nonlocal leaves
        if not _is_initial(position, observer):
            leaves += 1
            if leaves > max_leaves:
                raise DraftBudgetExceeded(f"setup draft exceeded {max_leaves} leaves")
            return setup_position_value(
                position, observer, weights, feature_terms=feature_terms), ()
        actions = candidates(position)
        maximizing = int(position.current_player) == observer
        best = -math.inf if maximizing else math.inf
        best_line = ()
        for action in actions:
            child = position.clone_game()
            child.apply(action)
            value, line = visit(child, alpha, beta)
            if ((maximizing and value > best)
                    or (not maximizing and value < best)):
                best, best_line = value, (action,) + line
            if maximizing:
                alpha = max(alpha, best)
            else:
                beta = min(beta, best)
            if beta <= alpha:
                break
        return best, best_line

    best_action, best_value, best_line, alpha = None, -math.inf, (), -math.inf
    root_values = []
    for action in candidates(game, root=True):
        child = game.clone_game()
        child.apply(action)
        value, line = visit(child, alpha, math.inf)
        root_values.append((action, value))
        if value > best_value:
            best_action, best_value, best_line = action, value, (action,) + line
        alpha = max(alpha, best_value)
    if best_action is None:
        raise RuntimeError("setup draft search received no legal action")
    return DraftResult(best_action, best_value, leaves, best_line,
                       tuple(root_values))
