"""Fast-engine paired arena with the turn-search workstream's pentanomial GSPRT."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
import math
import time

from catan_ranked.value_residual_features import (
    EDGE_VERTICES,
    TILE_VERTICES,
    VERTEX_NEIGHBORS,
)
from catan_ranked.value_selfplay import (
    LinearValue, NumpyMLPValue, NumpyResidualValue, WeightedLinearValue,
)
from catan_ranked_fast import Game


PAIR_SCORES = (0.0, 0.25, 0.5, 0.75, 1.0)
_EPSILON = 1e-3
_VALUE_CACHE = {}
_ACTION_CATEGORIES = {
    0: "cities", 1: "roads", 2: "settlements", 3: "dev_buys", 4: "discards",
}
_DEV_PLAY_ACTION_MIN = 304
_DEV_PLAY_ACTION_MAX = 330
_PLAY_KNIGHT = 304
_ROLL = 331
_ENDGAME_DEV_PLAY_MIN_VP = 10
_ENDGAME_DEV_PLAY_MARGIN = 0.01
_SETTLEMENT_OVER_DEV_MARGIN = 0.01
_FREE_ROAD_EXPANSION_MARGIN = 0.03
_PAID_ROAD_EXPANSION_MARGIN = 0.08
_MIN_SITE_QUALITY_GAIN = 2.0 / 36.0
_SETUP_COVERAGE_MARGIN = 0.06
_SETUP_PIP_TOLERANCE = 1.0 / 36.0
_BUY_DEV_CARD = 180
_ROBBER_ACTION_TILES = (
    10, 10, 10, 3, 3, 3, 4, 4, 4, 14, 14, 14, 11, 11, 11, 12, 12, 12,
    13, 13, 13, 2, 2, 2, 9, 9, 9, 0, 0, 0, 5, 5, 5, 15, 15, 15, 1, 1, 1,
    8, 8, 8, 6, 6, 6, 16, 16, 16, 18, 18, 18, 7, 7, 7, 17, 17, 17,
)
_ROBBER_FRONTIER_MARGIN = 0.01
_DEV_COMBO_WEIGHT = 0.012
_ROAD_BUILDING_TEMPO_WEIGHT = 0.018
_LONGEST_ROAD_REALISM_WEIGHT = 0.035
_CONTESTED_SPOT_URGENCY_WEIGHT = 0.035
_MEANINGFUL_ROAD_BUILDING_PROBABILITY = 0.10
_ROBBER_OPPONENT_NEED_WEIGHT = 0.03


def _buildable_sites(observation):
    nodes = observation[152:800]
    edges = observation[800:1016]
    occupancy = [
        max(range(5), key=lambda category: nodes[node * 12 + category])
        for node in range(54)
    ]
    connected = set()
    for edge in range(72):
        if edges[edge * 3 + 1] > 0.5:
            connected.update(int(vertex) for vertex in EDGE_VERTICES[edge])
    sites = set()
    for vertex in connected:
        if occupancy[vertex] != 0:
            continue
        if any(occupancy[int(neighbor)] != 0
               for neighbor in VERTEX_NEIGHBORS[vertex]):
            continue
        sites.add(vertex)
    return sites


def _site_production(observation, vertex):
    tiles = observation[:152]
    score = 0.0
    for tile, vertices in enumerate(TILE_VERTICES):
        base = tile * 8
        resource = max(range(6), key=lambda category: tiles[base + category])
        if vertex not in vertices or resource >= 5:
            continue
        if tiles[base + 7] > 0.5:
            continue
        # The observation stores pips / 5; divide the recovered pips by 36.
        score += float(tiles[base + 6]) * 5.0 / 36.0
    return score


def _production_by_resource(observation, player=0):
    """Return one player's unrobbed production probability by resource."""
    tiles = observation[:152]
    nodes = observation[152:800]
    production = [0.0] * 5
    for tile, vertices in enumerate(TILE_VERTICES):
        base = tile * 8
        resource = max(range(6), key=lambda category: tiles[base + category])
        if resource >= 5 or tiles[base + 7] > 0.5:
            continue
        multiplier = 0
        for vertex in vertices:
            occupancy = max(
                range(5),
                key=lambda category: nodes[int(vertex) * 12 + category],
            )
            settlement = 1 if player == 0 else 3
            city = 2 if player == 0 else 4
            multiplier += 1 if occupancy == settlement else 2 if occupancy == city else 0
        production[resource] += float(tiles[base + 6]) * 5.0 / 36.0 * multiplier
    return production


def _setup_expansion_profile(observation):
    production = _production_by_resource(observation)
    # Own-player block starts at 1016; its six port flags occupy offsets 19..24.
    owns_wood_port = observation[1016 + 20] > 0.5
    owns_brick_port = observation[1016 + 21] > 0.5
    access = (
        production[0] > 1e-12 or owns_wood_port,
        production[1] > 1e-12 or owns_brick_port,
    )
    return production, access


def _setup_coverage_settlement(game, ranked, observation, observer):
    """Complete a comparable second setup with wood-and-brick expansion access.

    Setup settlement search backs up the same general linear value used midgame.  Its
    diversity term only counts nonzero resource types, so swapping missing brick for
    missing ore is invisible, and ports have no value term.  This bounded correction
    considers only already-backed settlements within six hundredths of value and no more than
    one production pip below the planner choice.
    """
    top_action, top_value = ranked[0]
    if not 126 <= top_action < 180:
        return top_action
    nodes = observation[152:800]
    own_settlements = sum(nodes[vertex * 12 + 1] > 0.5 for vertex in range(54))
    if own_settlements != 1:
        return top_action

    candidates = []
    for action, value in ranked:
        if not 126 <= action < 180 or top_value - value > _SETUP_COVERAGE_MARGIN + 1e-12:
            continue
        child = game.clone_game()
        child.apply(action)
        production, access = _setup_expansion_profile(child.observation(observer))
        candidates.append({
            "action": action,
            "value": value,
            "production": production,
            "access": access,
        })
    if not candidates:
        return top_action
    chosen = next(candidate for candidate in candidates if candidate["action"] == top_action)
    chosen_total = sum(chosen["production"])
    comparable = [
        candidate for candidate in candidates
        if sum(candidate["production"]) + _SETUP_PIP_TOLERANCE + 1e-12 >= chosen_total
    ]
    best = max(
        comparable,
        key=lambda candidate: (
            sum(candidate["access"]),
            min(candidate["production"][:2]),
            candidate["value"],
            -candidate["action"],
        ),
    )
    if sum(best["access"]) <= sum(chosen["access"]):
        return top_action
    return best["action"]


def _expansion_road(game, ranked, observation, observer):
    """Prefer a close road that immediately opens a materially better settlement site."""
    top_action, top_value = ranked[0]
    if not 54 <= top_action < 126:
        return top_action
    margin = (_FREE_ROAD_EXPANSION_MARGIN if observation[-2] > 0.5
              else _PAID_ROAD_EXPANSION_MARGIN)
    root_sites = _buildable_sites(observation)
    candidates = []
    for action, value in ranked:
        if not 54 <= action < 126 or top_value - value > margin + 1e-12:
            continue
        child = game.clone_game()
        child.apply(action)
        child_observation = child.observation(observer)
        new_sites = _buildable_sites(child_observation) - root_sites
        quality = max((_site_production(child_observation, vertex) for vertex in new_sites),
                      default=0.0)
        candidates.append((quality, value, -action, action))
    if not candidates:
        return top_action
    best = max(candidates)
    chosen = next(candidate for candidate in candidates if candidate[-1] == top_action)
    return best[-1] if best[0] - chosen[0] >= _MIN_SITE_QUALITY_GAIN - 1e-12 else top_action


def _frontier_sites(observation, edge_category):
    nodes = observation[152:800]
    edges = observation[800:1016]
    occupancy = [
        max(range(5), key=lambda category: nodes[vertex * 12 + category])
        for vertex in range(54)
    ]
    connected = set()
    for edge, vertices in enumerate(EDGE_VERTICES):
        if edges[edge * 3 + edge_category] > 0.5:
            connected.update(int(vertex) for vertex in vertices)
    return {
        vertex for vertex in connected
        if occupancy[vertex] == 0
        and all(occupancy[int(neighbor)] == 0 for neighbor in VERTEX_NEIGHBORS[vertex])
    }


def _robber_frontier_action(ranked, observation):
    """Prefer a close legal robber move that blocks the opponent's best frontier.

    Friendly Robber filtering remains authoritative in the engine.  This term only reranks
    already-legal moves and refuses tiles touching our buildings or best road frontier.
    """
    top_action, top_value = ranked[0]
    if not 247 <= top_action < 304:
        return top_action
    nodes = observation[152:800]
    own_buildings = {
        vertex for vertex in range(54)
        if max(range(5), key=lambda category: nodes[vertex * 12 + category]) in (1, 2)
    }
    own_frontier = _frontier_sites(observation, 1)
    opponent_frontier = _frontier_sites(observation, 2)
    if not opponent_frontier:
        return top_action
    target = max(opponent_frontier, key=lambda vertex: (_site_production(observation, vertex), -vertex))
    protected = own_buildings | {
        vertex for vertex in own_frontier
        if _site_production(observation, vertex)
        == max((_site_production(observation, site) for site in own_frontier), default=-1.0)
    }
    candidates = []
    for action, value in ranked:
        if not 247 <= action < 304 or top_value - value > _ROBBER_FRONTIER_MARGIN + 1e-12:
            continue
        tile = _ROBBER_ACTION_TILES[action - 247]
        vertices = {int(vertex) for vertex in TILE_VERTICES[tile]}
        if target in vertices and not vertices.intersection(protected):
            candidates.append((value, -action, action))
    return max(candidates)[-1] if candidates else top_action


def _longest_road_contest_adjustment(game, action, observation, observer):
    """Discount a length-only road in proportion to how unrealistic the contest is."""
    if not 54 <= action < 126:
        return 0.0
    root_sites = _buildable_sites(observation)
    child = game.clone_game()
    child.apply(action)
    child_observation = child.observation(observer)
    if _buildable_sites(child_observation) - root_sites:
        return 0.0
    own_road = float(child_observation[1016 + 13]) * 15.0
    opponent_road = float(child_observation[1041 + 13]) * 15.0
    own_road_income = sum(_production_by_resource(child_observation, 0)[:2])
    opponent_road_income = sum(_production_by_resource(child_observation, 1)[:2])
    income_pressure = opponent_road_income / max(
        own_road_income + opponent_road_income, 1.0 / 36.0)
    length_pressure = min(1.0, max(0.0, (opponent_road + 2.0 - own_road) / 5.0))
    opponent_holds_title = float(child_observation[1041 + 14]) > 0.5
    difficulty = min(1.0, 0.55 * income_pressure + 0.35 * length_pressure
                     + 0.10 * opponent_holds_title)
    return -_LONGEST_ROAD_REALISM_WEIGHT * difficulty


def _opponent_one_road_sites(observation):
    """Empty legal-distance vertices on or one unbuilt edge from opponent roads."""
    nodes = observation[152:800]
    edges = observation[800:1016]
    occupancy = [
        max(range(5), key=lambda category: nodes[vertex * 12 + category])
        for vertex in range(54)
    ]
    connected = set()
    for edge, vertices in enumerate(EDGE_VERTICES):
        if edges[edge * 3 + 2] > 0.5:
            connected.update(map(int, vertices))
    reachable = set(connected)
    for edge, vertices in enumerate(EDGE_VERTICES):
        if edges[edge * 3] < 0.5:
            continue
        left, right = map(int, vertices)
        if left in connected:
            reachable.add(right)
        if right in connected:
            reachable.add(left)
    return {
        vertex for vertex in reachable
        if occupancy[vertex] == 0
        and all(occupancy[int(neighbor)] == 0 for neighbor in VERTEX_NEIGHBORS[vertex])
    }


def _opponent_road_distance(observation, target):
    """Shortest legal empty-edge route from the opponent network to ``target``."""
    nodes = observation[152:800]
    edges = observation[800:1016]
    occupancy = [
        max(range(5), key=lambda category: nodes[vertex * 12 + category])
        for vertex in range(54)
    ]
    reached = {
        vertex for vertex, owner in enumerate(occupancy) if owner in (3, 4)
    }
    for edge, vertices in enumerate(EDGE_VERTICES):
        if edges[edge * 3 + 2] > 0.5:
            reached.update(map(int, vertices))
    if not reached:
        return None
    distances = {vertex: 0 for vertex in reached}
    frontier = list(reached)
    while frontier:
        vertex = frontier.pop(0)
        if vertex == target:
            return distances[vertex]
        # An opposing building blocks a road from continuing through this vertex.
        if distances[vertex] and occupancy[vertex] in (1, 2):
            continue
        for edge, vertices in enumerate(EDGE_VERTICES):
            if edges[edge * 3] < 0.5:
                continue
            left, right = map(int, vertices)
            if vertex not in (left, right):
                continue
            neighbor = right if vertex == left else left
            if neighbor not in distances:
                distances[neighbor] = distances[vertex] + 1
                frontier.append(neighbor)
    return None


def _resource_affordability_probability(own, opponent_cards, costs):
    """Count-only public resource posterior probability of covering ``costs``."""
    pool = [max(0, 19 - int(count)) for count in own]
    total = sum(pool)
    draws = max(0, min(int(opponent_cards), total))
    if draws < sum(costs) or not total:
        return 0.0
    ways = {0: 1}
    for available, needed in zip(pool, costs):
        next_ways = {}
        for used, count in ways.items():
            for take in range(needed, min(available, draws - used) + 1):
                next_ways[used + take] = (next_ways.get(used + take, 0)
                                           + count * math.comb(available, take))
        ways = next_ways
    return ways.get(draws, 0) / math.comb(total, draws)


def _public_threat_context(game, observer):
    """Public resource/dev posteriors plus pressure from our city/dev plans."""
    if game is None or not hasattr(game, "state_json"):
        return None
    state = json.loads(game.state_json(observer))
    players = {int(player["seat"]): player for player in state["players"]}
    own = players[int(observer)]
    opponent = players[1 - int(observer)]
    resources = [int(value) for value in own["resources"]]

    initial = [14, 2, 2, 2, 5]
    own_dev = [int(value) for value in own["dev_hand"]]
    played = [sum(int(player["dev_played"][index]) for player in players.values())
              for index in range(5)]
    pool = [max(0, initial[index] - own_dev[index] - played[index])
            for index in range(5)]
    hidden = max(0, min(int(opponent["dev_count"]), sum(pool)))
    total = sum(pool)
    road_building = 0.0
    if hidden and total:
        without = (math.comb(total - pool[3], hidden)
                   if total - pool[3] >= hidden else 0)
        road_building = 1.0 - without / math.comb(total, hidden)

    city_progress = 0.5 * (min(resources[4] / 3.0, 1.0)
                           + min(resources[3] / 2.0, 1.0))
    dev_progress = sum(min(resources[index], 1) for index in (2, 3, 4)) / 3.0
    return {
        "own_resources": resources,
        "opponent_cards": int(opponent["resource_count"]),
        "road_building_probability": road_building,
        "competing_plan_pressure": max(city_progress, dev_progress),
    }


def _contested_spot_urgency_adjustment(action, observation, game=None, observer=0,
                                        threat_context=None):
    """Reward settling only when the opponent can credibly take the site next turn."""
    if not 126 <= action < 180:
        return 0.0
    vertex = action - 126
    distance = _opponent_road_distance(observation, vertex)
    context = threat_context or _public_threat_context(game, observer)
    if distance is None or context is None:
        return 0.0
    roads_built = float(observation[1041 + 13]) * 15.0
    if distance > max(0, 15 - round(roads_built)):
        return 0.0
    settlement_cost = (1, 1, 1, 1, 0)
    cash_cost = (distance + 1, distance + 1, 1, 1, 0)
    affordable = _resource_affordability_probability(
        context["own_resources"], context["opponent_cards"], cash_cost)
    road_building = context["road_building_probability"]
    dev_reach = 0.0
    if 0 < distance <= 2 and road_building >= _MEANINGFUL_ROAD_BUILDING_PROBABILITY:
        dev_reach = road_building * _resource_affordability_probability(
            context["own_resources"], context["opponent_cards"], settlement_cost)
    threat = max(affordable, dev_reach)
    if threat <= 0.0:
        return 0.0
    quality = min(1.0, _site_production(observation, vertex) / (5.0 / 36.0))
    plan_weight = 1.0 - 0.5 * context["competing_plan_pressure"]
    return (_CONTESTED_SPOT_URGENCY_WEIGHT * threat * plan_weight
            * (0.5 + 0.5 * quality))


def _robber_opponent_need_score(observation, tile):
    """Score blocked opponent pips by scarcity and city/dev resource demand."""
    base = tile * 8
    resource = max(range(6), key=lambda category: observation[base + category])
    if resource >= 5:
        return 0.0
    nodes = observation[152:800]
    opponent_units = 0
    own_units = 0
    for vertex in TILE_VERTICES[tile]:
        occupancy = max(
            range(5), key=lambda category: nodes[int(vertex) * 12 + category])
        opponent_units += 1 if occupancy == 3 else 2 if occupancy == 4 else 0
        own_units += 1 if occupancy == 1 else 2 if occupancy == 2 else 0
    if not opponent_units:
        return 0.0
    production = _production_by_resource(observation, 1)
    peak = max(max(production), 1.0 / 36.0)
    scarcity = 1.0 + max(0.0, peak - production[resource]) / peak
    plan_weight = (0.75, 0.75, 1.0, 1.55, 1.65)[resource]
    pips = float(observation[base + 6]) * 5.0 / 36.0
    return pips * opponent_units * plan_weight * scarcity - pips * own_units * 2.0


def _dev_combo_adjustment(action, observation):
    """Bounded tempo bonus for backed Monopoly/Year-of-Plenty continuations."""
    action = int(action)
    if 305 <= action <= 309:
        resource = action - 305
        production = _production_by_resource(observation, 1)
        peak = max(max(production), 1.0 / 36.0)
        plan_weight = (0.75, 0.75, 1.0, 1.55, 1.65)[resource]
        return _DEV_COMBO_WEIGHT * plan_weight * production[resource] / (1.65 * peak)
    if 311 <= action <= 330:
        # The backed turn leaf already captures which requested pair completes a build.
        return _DEV_COMBO_WEIGHT * 0.75
    return 0.0


def _road_building_tempo_adjustment(action, observation):
    """Reward RB only when two free roads reach expansion or contest a real site."""
    if int(action) != 310:
        return 0.0
    nodes = observation[152:800]
    edges = observation[800:1016]
    occupancy = [
        max(range(5), key=lambda category: nodes[vertex * 12 + category])
        for vertex in range(54)
    ]
    reached = {vertex for vertex, owner in enumerate(occupancy) if owner in (1, 2)}
    for edge, vertices in enumerate(EDGE_VERTICES):
        if edges[edge * 3 + 1] > 0.5:
            reached.update(map(int, vertices))
    distances = {vertex: 0 for vertex in reached}
    frontier = list(reached)
    while frontier:
        vertex = frontier.pop(0)
        if distances[vertex] >= 2 or (distances[vertex] and occupancy[vertex] in (3, 4)):
            continue
        for edge, vertices in enumerate(EDGE_VERTICES):
            if edges[edge * 3] < 0.5:
                continue
            left, right = map(int, vertices)
            if vertex not in (left, right):
                continue
            neighbor = right if vertex == left else left
            distance = distances[vertex] + 1
            if neighbor not in distances or distance < distances[neighbor]:
                distances[neighbor] = distance
                frontier.append(neighbor)
    sites = {
        vertex for vertex, distance in distances.items() if 1 <= distance <= 2
        and occupancy[vertex] == 0
        and all(occupancy[int(neighbor)] == 0 for neighbor in VERTEX_NEIGHBORS[vertex])
    }
    if not sites:
        return 0.0
    contested = _opponent_one_road_sites(observation)
    score = max(
        min(1.0, _site_production(observation, vertex) / (5.0 / 36.0))
        + (0.35 if vertex in contested else 0.0)
        for vertex in sites)
    return _ROAD_BUILDING_TEMPO_WEIGHT * min(1.0, score)


def _strategic_feature_ranked(game, ranked, observation, observer, *,
                              longest_road_contest_realism=False,
                              contested_spot_urgency=False,
                              robber_opponent_need=False,
                              dev_combo_search=False,
                              road_building_tempo=False):
    """Blend optional strategic terms into backed action values."""
    robber_scores = {}
    if robber_opponent_need:
        robber_scores = {
            action: _robber_opponent_need_score(observation, _ROBBER_ACTION_TILES[action - 247])
            for action, _value in ranked if 247 <= action < 304
        }
    robber_scale = max([abs(value) for value in robber_scores.values()] or [0.0])
    threat_context = (_public_threat_context(game, observer)
                      if contested_spot_urgency else None)
    adjusted = []
    for action, value in ranked:
        bonus = 0.0
        if longest_road_contest_realism:
            bonus += _longest_road_contest_adjustment(
                game, action, observation, observer)
        if contested_spot_urgency:
            bonus += _contested_spot_urgency_adjustment(
                action, observation, game, observer, threat_context)
        if robber_scale:
            bonus += (_ROBBER_OPPONENT_NEED_WEIGHT
                      * robber_scores.get(action, 0.0) / robber_scale)
        if dev_combo_search:
            bonus += _dev_combo_adjustment(action, observation)
        if road_building_tempo:
            bonus += _road_building_tempo_adjustment(action, observation)
        adjusted.append((action, value + bonus))
    return sorted(adjusted, key=lambda item: (-item[1], item[0]))


def choose_greedy_action(game, action_values, *, endgame_dev_card_liquidation=False,
                         build_spending_guard=False, setup_resource_coverage=False,
                         setup_draft_search=False, setup_draft_features=False,
                         pre_roll_knight_only=True,
                         robber_frontier=False,
                         longest_road_contest_realism=False,
                         contested_spot_urgency=False,
                         robber_opponent_need=False,
                         dev_combo_search=False,
                         road_building_tempo=False,
                         setup_draft_settlement_limit=6, setup_draft_max_leaves=30_000,
                         weights=None):
    """Choose a backed root action with optional bounded policy corrections.

    The linear evaluator gives a held development card the same positive value at every
    phase.  Near the end of the game that can make the turn search repeatedly promise to
    play a card later in its plan, then defer it again after replanning.  The opt-in policy
    only breaks a close endgame decision in favour of the best already-backed dev play; it
    never rescues a dev action more than one percentage point below the planner optimum.

    The spending guard separately prefers an already-backed settlement over a nearly tied
    development-card purchase and points close road choices at materially more productive
    expansion sites.  Before rolling, the default correctness filter permits Knight but removes
    all other development-card plays from the backed root actions.
    """
    ranked = sorted(action_values, key=lambda item: (-item[1], item[0]))
    if not ranked:
        raise RuntimeError("turn search returned no backed action")
    if pre_roll_knight_only and any(int(action) == _ROLL for action, _ in ranked):
        ranked = [
            item for item in ranked
            if not (_PLAY_KNIGHT < int(item[0]) <= _DEV_PLAY_ACTION_MAX)
        ]
    selected = ranked[0][0]
    seat = int(game.current_player)
    if setup_draft_search and game.observation(seat)[-9] > 0.5:
        if weights is None:
            raise ValueError("setup draft search requires the player's weight vector")
        from catan_ranked.setup_draft import (DraftBudgetExceeded,
                                              choose_setup_draft_action)
        try:
            return choose_setup_draft_action(
                game, weights, preferred=selected,
                settlement_limit=setup_draft_settlement_limit,
                max_leaves=setup_draft_max_leaves,
                feature_terms=setup_draft_features,
            ).action
        except DraftBudgetExceeded:
            return selected
    if (endgame_dev_card_liquidation
            and int(game.actual_vp[seat]) >= _ENDGAME_DEV_PLAY_MIN_VP):
        dev_actions = [
            item for item in ranked
            if _DEV_PLAY_ACTION_MIN <= int(item[0]) <= _DEV_PLAY_ACTION_MAX
        ]
        if (dev_actions
                and ranked[0][1] - dev_actions[0][1]
                <= _ENDGAME_DEV_PLAY_MARGIN + 1e-12):
            selected = dev_actions[0][0]
    observation = None
    if (longest_road_contest_realism or contested_spot_urgency
            or robber_opponent_need or dev_combo_search or road_building_tempo):
        observation = game.observation(seat)
        ranked = _strategic_feature_ranked(
            game, ranked, observation, seat,
            longest_road_contest_realism=longest_road_contest_realism,
            contested_spot_urgency=contested_spot_urgency,
            robber_opponent_need=robber_opponent_need,
            dev_combo_search=dev_combo_search,
            road_building_tempo=road_building_tempo,
        )
        selected = ranked[0][0]
    if setup_resource_coverage and 126 <= selected < 180:
        observation = game.observation(seat)
        selected = _setup_coverage_settlement(game, ranked, observation, seat)
    if robber_frontier and 247 <= selected < 304:
        if observation is None:
            observation = game.observation(seat)
        selected = _robber_frontier_action(ranked, observation)
    if not build_spending_guard:
        return selected

    if observation is None:
        observation = game.observation(seat)
    if selected == _BUY_DEV_CARD:
        settlements = [item for item in ranked if 126 <= item[0] < 180]
        if (settlements
                and ranked[0][1] - settlements[0][1]
                <= _SETTLEMENT_OVER_DEV_MARGIN + 1e-12):
            return settlements[0][0]
    if 54 <= selected < 126:
        return _expansion_road(game, ranked, observation, seat)
    return selected


def _value(spec):
    kind = {"play_only_mlp": "mlp", "play_only_residual": "residual",
            "simple_greedy_builder": "linear"}.get(
        spec["kind"], spec["kind"])
    key = (kind, spec.get("checkpoint"), tuple(spec.get("weights", ())))
    if key not in _VALUE_CACHE:
        if kind == "linear":
            _VALUE_CACHE[key] = LinearValue()
        elif kind == "weighted_linear":
            _VALUE_CACHE[key] = WeightedLinearValue(spec["weights"])
        elif kind == "mlp":
            _VALUE_CACHE[key] = NumpyMLPValue(spec["checkpoint"])
        elif kind == "residual":
            _VALUE_CACHE[key] = NumpyResidualValue(spec["checkpoint"])
        else:
            raise ValueError(f"unknown value kind {spec['kind']!r}")
    return _VALUE_CACHE[key]


def choose_policy_action(game, spec, *, max_plans=128, max_actions=14,
                         chance_samples=4, opponent_plans=12,
                         opponent_max_actions=10):
    """Choose one action for an arena policy from an arbitrary game root."""
    if spec["kind"] == "champion":
        return game.champion_action(spec.get("depth", 2))
    search = game.turn_search(
        max_plans, max_actions=max_actions, chance_samples=chance_samples,
        opponent_lookahead=spec.get("opponent_lookahead", False),
        opponent_plans=spec.get("opponent_plans", opponent_plans),
        opponent_max_actions=spec.get("opponent_max_actions", opponent_max_actions),
        opponent_model=spec.get("opponent_model", "min"))
    value = (_value(spec) if not spec["kind"].startswith("play_only_")
             or game.current_prompt == 2 else _value({"kind": "linear"}))
    leaf_values = value.leaf_values(search)
    if (spec.get("opponent_lookahead", False)
            and spec.get("opponent_model", "min") in {"self_max", "roll_only"}):
        action_values = search.action_values_with_opponent(
            leaf_values, value.opponent_leaf_values(search))
    else:
        action_values = search.action_values(leaf_values)
    if spec["kind"] == "simple_greedy_builder":
        builds = [(int(action), float(score)) for action, score in action_values
                  if 0 <= int(action) <= 180]
        if builds:
            return min(builds, key=lambda item: (-item[1], item[0]))[0]
    return choose_greedy_action(
        game, action_values,
        endgame_dev_card_liquidation=spec.get("endgame_dev_card_liquidation", False),
        build_spending_guard=spec.get("build_spending_guard", False),
        setup_resource_coverage=spec.get("setup_resource_coverage", False),
        setup_draft_search=spec.get("setup_draft_search", False),
        setup_draft_features=spec.get("setup_draft_features", False),
        pre_roll_knight_only=spec.get("pre_roll_knight_only", True),
        robber_frontier=spec.get("robber_frontier", False),
        longest_road_contest_realism=spec.get("longest_road_contest_realism", False),
        contested_spot_urgency=spec.get("contested_spot_urgency", False),
        robber_opponent_need=spec.get("robber_opponent_need", False),
        dev_combo_search=spec.get("dev_combo_search", False),
        road_building_tempo=spec.get("road_building_tempo", False),
        setup_draft_settlement_limit=spec.get("setup_draft_settlement_limit", 6),
        setup_draft_max_leaves=spec.get("setup_draft_max_leaves", 30_000),
        weights=spec.get("weights"))


def play_game(seed, candidate_seat, candidate, baseline, *, max_plans=128,
              max_actions=14, chance_samples=4, opponent_plans=12,
              opponent_max_actions=10, max_plies=8_000):
    game = Game(seed)
    specs = [candidate if seat == candidate_seat else baseline for seat in range(2)]
    action_mix = [{name: 0 for name in _ACTION_CATEGORIES.values()} for _ in range(2)]
    decision_ms = [[], []]
    for ply in range(max_plies):
        seat = game.current_player
        spec = specs[seat]
        decision_started = time.perf_counter()
        action = choose_policy_action(
            game, spec, max_plans=max_plans, max_actions=max_actions,
            chance_samples=chance_samples, opponent_plans=opponent_plans,
            opponent_max_actions=opponent_max_actions)
        decision_ms[seat].append((time.perf_counter() - decision_started) * 1_000.0)
        if action < 186:
            kind = 0 if action < 54 else 1 if action < 126 else 2 if action < 180 else 3 if action == 180 else 4
            category = _ACTION_CATEGORIES.get(kind)
            if category is not None:
                action_mix[seat][category] += 1
        game.apply(action)
        if game.winner >= 0:
            scores = game.actual_vp
            return {"winner": game.winner, "candidate_seat": candidate_seat,
                    "candidate_points": float(game.winner == candidate_seat),
                    "candidate_vp": scores[candidate_seat],
                    "baseline_vp": scores[1 - candidate_seat],
                    "candidate_action_mix": action_mix[candidate_seat],
                    "baseline_action_mix": action_mix[1 - candidate_seat],
                    "candidate_decision_ms": decision_ms[candidate_seat],
                    "baseline_decision_ms": decision_ms[1 - candidate_seat],
                    "plies": ply + 1, "capped": False}
    scores = game.actual_vp
    candidate_score, baseline_score = scores[candidate_seat], scores[1 - candidate_seat]
    points = 1.0 if candidate_score > baseline_score else 0.0 if candidate_score < baseline_score else 0.5
    return {"winner": -1, "candidate_seat": candidate_seat,
            "candidate_points": points, "candidate_vp": candidate_score,
            "baseline_vp": baseline_score,
            "candidate_action_mix": action_mix[candidate_seat],
            "baseline_action_mix": action_mix[1 - candidate_seat],
            "candidate_decision_ms": decision_ms[candidate_seat],
            "baseline_decision_ms": decision_ms[1 - candidate_seat],
            "plies": max_plies, "capped": True}


def play_pair(pair_index, candidate, baseline, *, base_seed=2_140_000_000, **game_options):
    seed = base_seed + pair_index
    games = [play_game(seed, seat, candidate, baseline, **game_options) for seat in (0, 1)]
    points = sum(game["candidate_points"] for game in games)
    category = int(round(points * 2.0))
    if category not in range(5) or abs(points * 2.0 - category) > 1e-9:
        raise RuntimeError("invalid paired result")
    return {"pair_index": pair_index, "board_seed": seed, "category": category,
            "candidate_points": points, "games": games}


def _bisect_root(function, low, high, tolerance=1e-13):
    f_low = function(low)
    if not (f_low <= 0 <= function(high) or function(high) <= 0 <= f_low):
        raise ValueError("constrained-MLE root is not bracketed")
    for _ in range(200):
        middle = (low + high) / 2.0
        f_middle = function(middle)
        if abs(f_middle) <= tolerance or high - low <= tolerance:
            return middle
        if (f_low <= 0 <= f_middle) or (f_middle <= 0 <= f_low):
            high = middle
        else:
            low, f_low = middle, f_middle
    raise ValueError("constrained-MLE root did not converge")


def _constrained_mle(probabilities, target):
    offsets = tuple(score - target for score in PAIR_SCORES)
    low = -1.0 / max(offsets) + 1e-12
    high = -1.0 / min(offsets) - 1e-12
    multiplier = _bisect_root(
        lambda value: sum(probability * offset / (1.0 + value * offset)
                          for probability, offset in zip(probabilities, offsets)),
        low, high)
    fitted = tuple(probability / (1.0 + multiplier * offset)
                   for probability, offset in zip(probabilities, offsets))
    total = sum(fitted)
    return tuple(value / total for value in fitted)


def logistic_score(elo):
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


def pentanomial_llr(counts, elo0=0.0, elo1=40.0):
    counts = tuple(int(value) for value in counts)
    if len(counts) != 5 or any(value < 0 for value in counts):
        raise ValueError("counts must be five nonnegative integers")
    if not sum(counts):
        return 0.0
    regularized = tuple(float(value) if value else _EPSILON for value in counts)
    total = sum(regularized)
    empirical = tuple(value / total for value in regularized)
    null = _constrained_mle(empirical, logistic_score(elo0))
    alternative = _constrained_mle(empirical, logistic_score(elo1))
    return total * sum(probability * math.log(p1 / p0)
                       for probability, p0, p1 in zip(empirical, null, alternative))


def evaluate_sprt(counts, *, elo0=0.0, elo1=40.0, alpha=0.05, beta=0.05):
    llr = pentanomial_llr(counts, elo0, elo1)
    lower = math.log(beta / (1.0 - alpha))
    upper = math.log((1.0 - beta) / alpha)
    decision = "accept_h0" if llr <= lower else "accept_h1" if llr >= upper else "continue"
    pairs = sum(counts)
    score = (sum(count * score for count, score in zip(counts, PAIR_SCORES)) / pairs
             if pairs else None)
    return {"counts": list(counts), "pairs": pairs, "pair_score": score,
            "llr": llr, "lower_bound": lower, "upper_bound": upper,
            "decision": decision, "elo0": elo0, "elo1": elo1}


def _pair_task(arguments):
    pair_index, candidate, baseline, options = arguments
    return play_pair(pair_index, candidate, baseline, **options)


def run_sprt(candidate, baseline, *, max_pairs=4_000, workers=1, batch_pairs=None,
             elo0=0.0, elo1=40.0, alpha=0.05, beta=0.05,
             base_seed=2_140_000_000, stop_on_decision=True, **game_options):
    if max_pairs <= 0 or workers <= 0 or elo1 <= elo0:
        raise ValueError("invalid arena configuration")
    batch_pairs = batch_pairs or workers * 4
    counts = [0] * 5
    game_count = 0
    final_vp = {"candidate": 0.0, "baseline": 0.0}
    action_mix = {
        side: {name: 0 for name in _ACTION_CATEGORIES.values()}
        for side in ("candidate", "baseline")
    }
    decision_ms = {"candidate": [], "baseline": []}
    started = time.perf_counter()
    terminal = None
    options = {"base_seed": base_seed, **game_options}
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        for start in range(0, max_pairs, batch_pairs):
            indices = range(start, min(start + batch_pairs, max_pairs))
            tasks = [(index, candidate, baseline, options) for index in indices]
            results = (pool.map(_pair_task, tasks) if pool else map(_pair_task, tasks))
            for result in sorted(results, key=lambda item: item["pair_index"]):
                counts[result["category"]] += 1
                for game in result["games"]:
                    game_count += 1
                    for side in ("candidate", "baseline"):
                        final_vp[side] += game[f"{side}_vp"]
                        decision_ms[side].extend(game[f"{side}_decision_ms"])
                        for name, count in game[f"{side}_action_mix"].items():
                            action_mix[side][name] += count
                terminal = evaluate_sprt(counts, elo0=elo0, elo1=elo1,
                                         alpha=alpha, beta=beta)
                if stop_on_decision and terminal["decision"] != "continue":
                    break
            if stop_on_decision and terminal["decision"] != "continue":
                break
    finally:
        if pool is not None:
            pool.shutdown()
    elapsed = time.perf_counter() - started
    action_mix_per_game = {
        side: {name: count / game_count for name, count in counts_by_name.items()}
        for side, counts_by_name in action_mix.items()
    }
    decision_timing_ms = {}
    for side, values in decision_ms.items():
        ordered = sorted(values)
        count = len(ordered)
        decision_timing_ms[side] = {
            "count": count,
            "mean": sum(ordered) / count if count else None,
            "median": ((ordered[(count - 1) // 2] + ordered[count // 2]) / 2.0
                       if count else None),
            "p90": ordered[max(0, math.ceil(0.9 * count) - 1)] if count else None,
        }
    return {**terminal, "elapsed_seconds": elapsed,
            "pairs_per_hour": terminal["pairs"] / elapsed * 3600.0,
            "games": game_count,
            "mean_final_vp": {side: total / game_count for side, total in final_vp.items()},
            "action_mix": action_mix,
            "action_mix_per_game": action_mix_per_game,
            "decision_timing_ms": decision_timing_ms,
            "candidate": candidate, "baseline": baseline,
            "base_seed": base_seed, "max_pairs": max_pairs,
            "stop_on_decision": stop_on_decision}
