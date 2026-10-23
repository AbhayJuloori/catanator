use crate::state::RankedState;
use crate::topology::PORT_VERTICES;

pub const OBSERVATION_DIM: usize = 1193;
pub const OBSERVATION_VERSION: &str = "value-afterstate-observer/1";

const DEV_MAX: [f32; 5] = [14.0, 2.0, 2.0, 2.0, 5.0];
const DICE_MAX: [f32; 11] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

fn one_hot(out: &mut Vec<f32>, width: usize, index: usize) {
    out.extend((0..width).map(|candidate| (candidate == index) as u8 as f32));
}

fn owned_ports(state: &RankedState, player: usize) -> [f32; 6] {
    let mut ports = [0.0; 6];
    for (port, vertices) in PORT_VERTICES.iter().enumerate() {
        if vertices.iter().any(|&vertex| {
            let value = state.vertices[vertex as usize];
            value >= 0 && value as usize % 2 == player
        }) {
            let kind = state.port_types[port];
            ports[if kind < 0 { 0 } else { kind as usize + 1 }] = 1.0;
        }
    }
    ports
}

fn player_block(state: &RankedState, player: usize, observer: usize, out: &mut Vec<f32>) {
    out.extend(
        state.resources[player]
            .iter()
            .map(|&value| value as f32 / 19.0),
    );
    if player == observer {
        out.extend(
            state.dev_hand[player]
                .iter()
                .enumerate()
                .map(|(card, &value)| value as f32 / DEV_MAX[card]),
        );
        out.push(0.0);
    } else {
        out.extend([0.0; 5]);
        out.push(
            state.dev_hand[player]
                .iter()
                .map(|&x| x as f32)
                .sum::<f32>()
                / 25.0,
        );
    }
    out.push(state.dev_played[player][0] as f32 / 14.0);
    out.push(state.visible_vp[player] as f32 / 15.0);
    out.push(state.road_lengths[player] as f32 / 15.0);
    out.push(state.has_road[player] as u8 as f32);
    out.push(state.has_army[player] as u8 as f32);
    out.push(state.roads_available[player] as f32 / 15.0);
    out.push(state.settlements_available[player] as f32 / 5.0);
    out.push(state.cities_available[player] as f32 / 4.0);
    out.extend(owned_ports(state, player));
}

pub fn encode(state: &RankedState, observer: usize) -> Vec<f32> {
    assert!(observer < 2);
    let opponent = 1 - observer;
    let mut out = Vec::with_capacity(OBSERVATION_DIM);

    // 19 tiles x (six resource/desert categories, pip count, robber flag).
    for tile in 0..19 {
        one_hot(&mut out, 6, state.tile_resources[tile] as usize);
        let number = state.tile_numbers[tile] as i16;
        let pips = if number == 0 || number == 7 {
            0
        } else {
            6 - (7 - number).abs()
        };
        out.push(pips as f32 / 5.0);
        out.push((state.robber_tile as usize == tile) as u8 as f32);
    }

    // 54 nodes x (empty/my settlement/my city/opponent settlement/opponent city,
    // no port/generic/resource-specific port).
    for node in 0..54 {
        let value = state.vertices[node];
        let occupancy = if value < 0 {
            0
        } else {
            let owner = value as usize % 2;
            let city = value >= 2;
            match (owner == observer, city) {
                (true, false) => 1,
                (true, true) => 2,
                (false, false) => 3,
                (false, true) => 4,
            }
        };
        one_hot(&mut out, 5, occupancy);
        let port_category = PORT_VERTICES
            .iter()
            .position(|vertices| vertices.contains(&(node as u8)))
            .map(|port| {
                let kind = state.port_types[port];
                if kind < 0 { 1 } else { kind as usize + 2 }
            })
            .unwrap_or(0);
        one_hot(&mut out, 7, port_category);
    }

    // 72 edges x (empty/mine/opponent).
    for &owner in &state.edges {
        let category = if owner < 0 {
            0
        } else if owner as usize == observer {
            1
        } else {
            2
        };
        one_hot(&mut out, 3, category);
    }

    player_block(state, observer, observer, &mut out);
    player_block(state, opponent, observer, &mut out);
    out.extend(state.bank.iter().map(|&value| value as f32 / 19.0));
    out.push(state.dev_len as f32 / 25.0);

    // Exact public Balanced-Dice Markov state and derived next-roll distributions.
    for (offset, &maximum) in DICE_MAX.iter().enumerate() {
        out.push(state.dice.counts[offset + 2] as f32 / maximum);
    }
    out.push(state.dice.cards_left as f32 / 36.0);
    for position in 0..5 {
        let category = if position < state.dice.recent_len as usize {
            state.dice.recent[position] as usize - 1 // total 2..12 -> category 1..11
        } else {
            0
        };
        one_hot(&mut out, 12, category);
    }
    out.push(state.dice.total_sevens[state.colors[observer] as usize] as f32 / 16.0);
    out.push(state.dice.total_sevens[state.colors[opponent] as usize] as f32 / 16.0);
    out.push(state.dice.seen_player[state.colors[observer] as usize] as u8 as f32);
    out.push(state.dice.seen_player[state.colors[opponent] as usize] as u8 as f32);
    let streak = state.dice.streak_color;
    let streak_category = if streak < 0 {
        0
    } else if streak as u8 == state.colors[observer] {
        1
    } else {
        2
    };
    one_hot(&mut out, 3, streak_category);
    out.push(state.dice.streak_count as f32 / 5.0);
    out.extend(
        state.dice.distribution(state.colors[observer] as usize)[2..13]
            .iter()
            .map(|&value| value as f32),
    );
    out.extend(
        state.dice.distribution(state.colors[opponent] as usize)[2..13]
            .iter()
            .map(|&value| value as f32),
    );

    out.push(state.num_turns.min(200) as f32 / 200.0);
    one_hot(
        &mut out,
        2,
        (state.current_player as usize != observer) as usize,
    );
    one_hot(
        &mut out,
        2,
        (state.current_turn as usize != observer) as usize,
    );
    one_hot(&mut out, 5, state.prompt as usize);
    out.push(state.initial as u8 as f32);
    out.push(state.has_rolled[observer] as u8 as f32);
    out.push(state.has_rolled[opponent] as u8 as f32);
    out.push(state.played_dev_this_turn[observer] as u8 as f32);
    out.push(state.played_dev_this_turn[opponent] as u8 as f32);
    out.push(state.discard_counts[observer] as f32 / 10.0);
    out.push(state.discard_counts[opponent] as f32 / 10.0);
    out.push(state.road_building as u8 as f32);
    out.push(state.free_roads as f32 / 2.0);

    debug_assert_eq!(out.len(), OBSERVATION_DIM);
    out
}
