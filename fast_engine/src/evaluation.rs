//! Exact Rust port of `tuned_consensus` from `weights/tuned_consensus.json`.

use crate::observation::OBSERVATION_DIM;
use crate::state::{BRICK, ORE, RankedState, SHEEP, WHEAT, WOOD};
use crate::topology::{EDGE_VERTICES, TILE_VERTICES, VERTEX_NEIGHBORS};

pub const WEIGHT_COUNT: usize = 24;
pub const CONSENSUS_WEIGHTS: [f64; WEIGHT_COUNT] = [
    1.16465, 0.96372, 0.94685, 1.23796, 0.70161, 0.2981, 0.50993, 0.46285, 0.27988, 0.258, 0.2411,
    0.20713, 1.09979, 1.17024, 0.99872, 1.75782, 1.80343, 0.45319, 0.41919, 0.27732, 0.01199,
    -0.03921, 0.08352, -0.12056,
];

const VP_PROGRESS: f64 = CONSENSUS_WEIGHTS[0];
const VP_LEAD: f64 = 0.96372;
const PRODUCTION: f64 = 0.94685;
const OPP_PRODUCTION: f64 = 1.23796;
const PROD_DIVERSITY: f64 = 0.70161;
const BUILDABLE: f64 = 0.2981;
const LONGEST_ROAD: f64 = 0.50993;
const HAS_LONGEST_ROAD: f64 = 0.46285;
const ARMY: f64 = 0.27988;
const HAS_LARGEST_ARMY: f64 = 0.258;
const HAND_RESOURCES: f64 = 0.2411;
const HAND_DEVS: f64 = 0.20713;
const RESOURCE_WEIGHTS: [f64; 5] = [1.09979, 1.17024, 0.99872, 1.75782, 1.80343];
const SCALE: f64 = 0.45319;
const SYNERGY_EXPANSION: f64 = 0.41919;
const SYNERGY_CITY: f64 = 0.27732;
const BONUS_VP_REACH: f64 = 0.01199;
const BONUS_VP_DENIAL: f64 = -0.03921;
const EXPANSION_TEMPO: f64 = 0.08352;
const CONTESTED_SHARE: f64 = -0.12056;

fn number_probability(number: u8) -> f64 {
    if !(2..=12).contains(&number) {
        0.0
    } else {
        (6 - (7i16 - number as i16).abs()) as f64 / 36.0
    }
}

fn expected_production(state: &RankedState, player: usize) -> [f64; 5] {
    let mut production = [0.0; 5];
    for tile in 0..19 {
        if tile == state.robber_tile as usize || state.tile_resources[tile] >= 5 {
            continue;
        }
        let probability = number_probability(state.tile_numbers[tile]);
        let resource = state.tile_resources[tile] as usize;
        for &node in &TILE_VERTICES[tile] {
            let building = state.vertices[node as usize];
            if building >= 0 && building as usize % 2 == player {
                production[resource] += probability * if building >= 2 { 2.0 } else { 1.0 };
            }
        }
    }
    production
}

fn buildable_mask(state: &RankedState, player: usize) -> u64 {
    let connected = state.components[player]
        .iter()
        .fold(0u64, |all, &component| all | component);
    let mut buildable = 0u64;
    for node in 0..54 {
        if connected & (1u64 << node) == 0 || state.vertices[node] >= 0 {
            continue;
        }
        if VERTEX_NEIGHBORS[node]
            .iter()
            .all(|&neighbor| neighbor < 0 || state.vertices[neighbor as usize] < 0)
        {
            buildable |= 1u64 << node;
        }
    }
    buildable
}

fn bonus_vp_reach(state: &RankedState, player: usize, production: &[f64; 5]) -> f64 {
    let road_fuel = production[WOOD as usize].min(production[BRICK as usize]);
    let mut road = (state.road_lengths[player] as f64 / 5.0).min(1.0)
        * (0.5 + 0.5 * (road_fuel / 0.5).min(1.0));
    if state.has_road[player] {
        road = 1.0;
    }
    let dev_fuel = production[ORE as usize]
        .min(production[SHEEP as usize])
        .min(production[WHEAT as usize]);
    let mut army = (state.dev_played[player][0] as f64 / 3.0).min(1.0)
        * (0.5 + 0.5 * (dev_fuel / 0.35).min(1.0));
    if state.has_army[player] {
        army = 1.0;
    }
    road.max(army)
}

fn expansion_tempo(state: &RankedState, player: usize, production: &[f64; 5]) -> f64 {
    let mut turns: f64 = 0.0;
    for resource in [WOOD, BRICK, SHEEP, WHEAT] {
        let index = resource as usize;
        let deficit = (1.0 - state.resources[player][index] as f64).max(0.0);
        let rate = production[index];
        turns = turns.max(if rate > 1e-6 { deficit / rate } else { 25.0 });
    }
    1.0 / (1.0 + turns)
}

pub fn tuned_consensus(state: &RankedState, observer: usize, opp_hidden_vp: f64) -> f64 {
    weighted_consensus(state, observer, opp_hidden_vp, &CONSENSUS_WEIGHTS)
}

/// Evaluate the same 24 consensus terms with caller-supplied weights.
///
/// The order is the declared `TERM_NAMES` order: the twelve scalar terms, five resource
/// weights, scale, then the six interaction terms.
pub fn weighted_consensus(
    state: &RankedState,
    observer: usize,
    opp_hidden_vp: f64,
    weights: &[f64; WEIGHT_COUNT],
) -> f64 {
    let opponent = 1 - observer;
    let my_vp = state.actual_vp[observer] as f64;
    if my_vp >= 15.0 {
        return 1.0;
    }
    let opp_vp = state.visible_vp[opponent] as f64 + opp_hidden_vp;
    let my_prod = expected_production(state, observer);
    let opp_prod = expected_production(state, opponent);
    let my_prod_w: f64 = my_prod
        .iter()
        .zip(&weights[12..17])
        .map(|(production, weight)| production * weight)
        .sum();
    let opp_prod_w: f64 = opp_prod
        .iter()
        .zip(&weights[12..17])
        .map(|(production, weight)| production * weight)
        .sum();
    let diversity = my_prod.iter().filter(|&&value| value > 1e-6).count();
    let my_buildable = buildable_mask(state, observer);
    let opp_buildable = buildable_mask(state, opponent);

    let mut score = weights[0] * (my_vp / 15.0)
        + weights[1] * ((my_vp - opp_vp) / 15.0)
        + weights[2] * my_prod_w
        - weights[3] * opp_prod_w
        + weights[4] * (diversity as f64 / 5.0)
        + weights[5] * (my_buildable.count_ones() as f64 / 10.0)
        + weights[6] * (state.road_lengths[observer] as f64 / 10.0)
        + weights[7] * state.has_road[observer] as u8 as f64
        + weights[8] * (state.dev_played[observer][0] as f64 / 3.0)
        + weights[9] * state.has_army[observer] as u8 as f64
        + weights[10] * (state.total_resources(observer).min(9) as f64 / 9.0)
        + weights[11]
            * state.dev_hand[observer]
                .iter()
                .map(|&x| x as f64)
                .sum::<f64>();
    let contested = if my_buildable == 0 {
        0.0
    } else {
        (my_buildable & opp_buildable).count_ones() as f64 / my_buildable.count_ones() as f64
    };
    score += weights[18] * my_prod[WOOD as usize].min(my_prod[BRICK as usize])
        + weights[19] * my_prod[ORE as usize].min(my_prod[WHEAT as usize])
        + weights[20] * bonus_vp_reach(state, observer, &my_prod)
        - weights[21] * bonus_vp_reach(state, opponent, &opp_prod)
        + weights[22] * expansion_tempo(state, observer, &my_prod)
        + weights[23] * contested;
    (weights[17] * score).tanh()
}

/// Reconstruct the observer-visible consensus value from a serialized observation.
///
/// Generation zero predates stored bootstrap values. This deliberately uses only the public
/// observation contract, including the observer's typed development cards, so offline TD(lambda)
/// targets cannot recover or depend on an opponent's hidden card identities.
pub fn tuned_consensus_observation(observation: &[f32]) -> Result<f64, String> {
    if observation.len() != OBSERVATION_DIM || observation.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "expected {OBSERVATION_DIM} finite observation values"
        ));
    }
    const TILE_OFFSET: usize = 0;
    const NODE_OFFSET: usize = 19 * 8;
    const EDGE_OFFSET: usize = NODE_OFFSET + 54 * 12;
    const OWN_OFFSET: usize = EDGE_OFFSET + 72 * 3;
    const OPP_OFFSET: usize = OWN_OFFSET + 25;
    const DEV_MAX: [f64; 5] = [14.0, 2.0, 2.0, 2.0, 5.0];

    let category = |offset: usize, width: usize| -> usize {
        (0..width)
            .max_by(|&left, &right| {
                observation[offset + left].total_cmp(&observation[offset + right])
            })
            .unwrap()
    };
    let mut production = [[0.0; 5]; 2];
    let mut occupied = [false; 54];
    for node in 0..54 {
        let occupancy = category(NODE_OFFSET + node * 12, 5);
        occupied[node] = occupancy != 0;
    }
    for tile in 0..19 {
        let base = TILE_OFFSET + tile * 8;
        let resource = category(base, 6);
        if resource >= 5 || observation[base + 7] > 0.5 {
            continue;
        }
        let probability = observation[base + 6] as f64 * 5.0 / 36.0;
        for &node in &TILE_VERTICES[tile] {
            match category(NODE_OFFSET + node as usize * 12, 5) {
                1 => production[0][resource] += probability,
                2 => production[0][resource] += 2.0 * probability,
                3 => production[1][resource] += probability,
                4 => production[1][resource] += 2.0 * probability,
                _ => {}
            }
        }
    }
    let mut connected = [0u64; 2];
    for (edge, vertices) in EDGE_VERTICES.iter().enumerate() {
        let owner = category(EDGE_OFFSET + edge * 3, 3);
        if owner > 0 {
            connected[owner - 1] |= 1u64 << vertices[0] | 1u64 << vertices[1];
        }
    }
    let buildable = |player: usize| -> u64 {
        let mut mask = 0u64;
        for node in 0..54 {
            if connected[player] & (1u64 << node) == 0 || occupied[node] {
                continue;
            }
            if VERTEX_NEIGHBORS[node]
                .iter()
                .all(|&neighbor| neighbor < 0 || !occupied[neighbor as usize])
            {
                mask |= 1u64 << node;
            }
        }
        mask
    };
    let my_buildable = buildable(0);
    let opp_buildable = buildable(1);
    let visible_vp = |base: usize| observation[base + 12] as f64 * 15.0;
    let road_length = |base: usize| observation[base + 13] as f64 * 15.0;
    let has_road = |base: usize| observation[base + 14] > 0.5;
    let has_army = |base: usize| observation[base + 15] > 0.5;
    let played_knights = |base: usize| observation[base + 11] as f64 * 14.0;
    let my_vp = visible_vp(OWN_OFFSET) + observation[OWN_OFFSET + 9] as f64 * DEV_MAX[4];
    if my_vp >= 15.0 - 1e-6 {
        return Ok(1.0);
    }
    let opp_vp = visible_vp(OPP_OFFSET);
    let weighted = |values: &[f64; 5]| {
        values
            .iter()
            .zip(RESOURCE_WEIGHTS)
            .map(|(value, weight)| value * weight)
            .sum::<f64>()
    };
    let bonus = |player: usize, base: usize| {
        let prod = &production[player];
        let road_fuel = prod[WOOD as usize].min(prod[BRICK as usize]);
        let mut road =
            (road_length(base) / 5.0).min(1.0) * (0.5 + 0.5 * (road_fuel / 0.5).min(1.0));
        if has_road(base) {
            road = 1.0;
        }
        let dev_fuel = prod[ORE as usize]
            .min(prod[SHEEP as usize])
            .min(prod[WHEAT as usize]);
        let mut army =
            (played_knights(base) / 3.0).min(1.0) * (0.5 + 0.5 * (dev_fuel / 0.35).min(1.0));
        if has_army(base) {
            army = 1.0;
        }
        road.max(army)
    };
    let my_prod = &production[0];
    let opp_prod = &production[1];
    let diversity = my_prod.iter().filter(|&&value| value > 1e-6).count();
    let resources: [f64; 5] =
        std::array::from_fn(|index| observation[OWN_OFFSET + index] as f64 * 19.0);
    let hand_resources = resources.iter().sum::<f64>().min(9.0) / 9.0;
    let hand_devs = (0..5)
        .map(|index| observation[OWN_OFFSET + 5 + index] as f64 * DEV_MAX[index])
        .sum::<f64>();
    let mut score = VP_PROGRESS * (my_vp / 15.0)
        + VP_LEAD * ((my_vp - opp_vp) / 15.0)
        + PRODUCTION * weighted(my_prod)
        - OPP_PRODUCTION * weighted(opp_prod)
        + PROD_DIVERSITY * (diversity as f64 / 5.0)
        + BUILDABLE * (my_buildable.count_ones() as f64 / 10.0)
        + LONGEST_ROAD * (road_length(OWN_OFFSET) / 10.0)
        + HAS_LONGEST_ROAD * has_road(OWN_OFFSET) as u8 as f64
        + ARMY * (played_knights(OWN_OFFSET) / 3.0)
        + HAS_LARGEST_ARMY * has_army(OWN_OFFSET) as u8 as f64
        + HAND_RESOURCES * hand_resources
        + HAND_DEVS * hand_devs;
    let contested = if my_buildable == 0 {
        0.0
    } else {
        (my_buildable & opp_buildable).count_ones() as f64 / my_buildable.count_ones() as f64
    };
    let settlement_resources = [
        WOOD as usize,
        BRICK as usize,
        SHEEP as usize,
        WHEAT as usize,
    ];
    let turns = settlement_resources
        .iter()
        .fold(0.0_f64, |current, &resource| {
            let deficit = (1.0 - resources[resource]).max(0.0);
            let rate = my_prod[resource];
            current.max(if rate > 1e-6 { deficit / rate } else { 25.0 })
        });
    score += SYNERGY_EXPANSION * my_prod[WOOD as usize].min(my_prod[BRICK as usize])
        + SYNERGY_CITY * my_prod[ORE as usize].min(my_prod[WHEAT as usize])
        + BONUS_VP_REACH * bonus(0, OWN_OFFSET)
        - BONUS_VP_DENIAL * bonus(1, OPP_OFFSET)
        + EXPANSION_TEMPO * (1.0 / (1.0 + turns))
        + CONTESTED_SHARE * contested;
    Ok((SCALE * score).tanh())
}
