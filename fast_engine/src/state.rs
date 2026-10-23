use crate::rng::PythonRandom;
use crate::topology::{PORT_VERTICES, SPIRAL_TILE_ORDER};

pub const WOOD: u8 = 0;
pub const BRICK: u8 = 1;
pub const SHEEP: u8 = 2;
pub const WHEAT: u8 = 3;
pub const ORE: u8 = 4;
pub const DESERT: u8 = 5;

pub const KNIGHT: u8 = 0;
pub const YEAR_OF_PLENTY: u8 = 1;
pub const MONOPOLY: u8 = 2;
pub const ROAD_BUILDING: u8 = 3;
pub const VICTORY_POINT: u8 = 4;

pub const PROMPT_INITIAL_SETTLEMENT: u8 = 0;
pub const PROMPT_INITIAL_ROAD: u8 = 1;
pub const PROMPT_PLAY: u8 = 2;
pub const PROMPT_DISCARD: u8 = 3;
pub const PROMPT_ROBBER: u8 = 4;

const TILE_RESOURCE_TEMPLATE: [u8; 19] = [
    WOOD, WOOD, WOOD, WOOD, BRICK, BRICK, BRICK, SHEEP, SHEEP, SHEEP, SHEEP, WHEAT, WHEAT, WHEAT,
    WHEAT, ORE, ORE, ORE, DESERT,
];
const PORT_TEMPLATE: [i8; 9] = [
    WOOD as i8,
    BRICK as i8,
    SHEEP as i8,
    WHEAT as i8,
    ORE as i8,
    -1,
    -1,
    -1,
    -1,
];
const NUMBER_TEMPLATE: [u8; 18] = [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12];
const SPIRAL_NUMBERS: [u8; 18] = [5, 2, 6, 3, 8, 10, 9, 12, 11, 4, 8, 10, 9, 4, 5, 6, 3, 11];
const DEV_TEMPLATE: [u8; 25] = [
    KNIGHT,
    KNIGHT,
    KNIGHT,
    KNIGHT,
    KNIGHT,
    KNIGHT,
    KNIGHT,
    KNIGHT,
    KNIGHT,
    KNIGHT,
    KNIGHT,
    KNIGHT,
    KNIGHT,
    KNIGHT,
    YEAR_OF_PLENTY,
    YEAR_OF_PLENTY,
    ROAD_BUILDING,
    ROAD_BUILDING,
    MONOPOLY,
    MONOPOLY,
    VICTORY_POINT,
    VICTORY_POINT,
    VICTORY_POINT,
    VICTORY_POINT,
    VICTORY_POINT,
];

#[derive(Clone)]
pub struct BalancedDice {
    pub counts: [u8; 13],
    pub cards_left: u8,
    pub recent: [u8; 5],
    pub recent_len: u8,
    pub total_sevens: [u16; 2],
    pub seen_player: [bool; 2], // keyed by absolute color
    pub streak_color: i8,
    pub streak_count: u16,
}

impl BalancedDice {
    pub fn new() -> Self {
        let mut out = Self {
            counts: [0; 13],
            cards_left: 36,
            recent: [0; 5],
            recent_len: 0,
            total_sevens: [0; 2],
            seen_player: [false; 2],
            streak_color: -1,
            streak_count: 0,
        };
        out.reshuffle();
        out
    }

    fn reshuffle(&mut self) {
        self.counts = [0; 13];
        for a in 1..=6 {
            for b in 1..=6 {
                self.counts[a + b] += 1;
            }
        }
        self.cards_left = 36;
    }

    fn prepare(&mut self, color: usize) {
        if self.cards_left < 13 {
            self.reshuffle();
        }
        self.seen_player[color] = true;
    }

    fn recent_count(&self, total: usize) -> u8 {
        self.recent[..self.recent_len as usize]
            .iter()
            .filter(|&&x| x as usize == total)
            .count() as u8
    }

    pub fn distribution(&self, color: usize) -> [f64; 13] {
        let mut prepared = self.clone();
        prepared.prepare(color);
        let mut weights = [0.0; 13];
        for total in 2..=12 {
            let base = prepared.counts[total] as f64 / prepared.cards_left as f64;
            let multiplier = (1.0 - prepared.recent_count(total) as f64 * 0.34).max(0.0);
            weights[total] = base * multiplier;
        }
        let seen = prepared.seen_player.iter().filter(|&&v| v).count();
        let total_sevens: u16 = prepared.total_sevens.iter().sum();
        let imbalance = if total_sevens < seen as u16 {
            1.0
        } else {
            let actual = prepared.total_sevens[color] as f64 / total_sevens as f64;
            let ideal = 1.0 / seen as f64;
            1.0 + ((ideal - actual) / ideal)
        };
        let sign = if prepared.streak_color == color as i8 {
            -1.0
        } else {
            1.0
        };
        let multiplier = (imbalance + 0.4 * prepared.streak_count as f64 * sign).clamp(0.0, 2.0);
        weights[7] *= multiplier;
        let sum: f64 = weights.iter().sum();
        for weight in &mut weights {
            *weight /= sum;
        }
        weights
    }

    pub fn consume(&mut self, color: usize, total: u8) -> bool {
        self.prepare(color);
        if !(2..=12).contains(&total) || self.counts[total as usize] == 0 {
            return false;
        }
        self.counts[total as usize] -= 1;
        self.cards_left -= 1;
        if self.recent_len < 5 {
            self.recent[self.recent_len as usize] = total;
            self.recent_len += 1;
        } else {
            self.recent.copy_within(1..5, 0);
            self.recent[4] = total;
        }
        if total == 7 {
            self.total_sevens[color] += 1;
            if self.streak_color == color as i8 {
                self.streak_count += 1;
            } else {
                self.streak_color = color as i8;
                self.streak_count = 1;
            }
        }
        true
    }

    pub fn draw(&mut self, color: usize, rng: &mut PythonRandom) -> u8 {
        let weights = self.distribution(color);
        let mut target = rng.random() * weights.iter().sum::<f64>();
        let mut total = 12u8;
        for candidate in 2..=12 {
            target -= weights[candidate];
            if target <= 0.0 {
                total = candidate as u8;
                break;
            }
        }
        self.consume(color, total);
        total
    }
}

impl RankedState {
    /// Canonical observer-view identity used for planner transposition and leaf deduplication.
    /// Opponent development-card identities and deck order are deliberately absent.
    pub fn observer_digest(&self, observer: usize) -> Vec<i16> {
        let opponent = 1 - observer;
        let mut out = Vec::with_capacity(320);
        out.extend(self.tile_resources.iter().map(|&x| x as i16));
        out.extend(self.tile_numbers.iter().map(|&x| x as i16));
        out.extend(self.port_types.iter().map(|&x| x as i16));
        out.extend(self.vertices.iter().map(|&x| x as i16));
        out.extend(self.edges.iter().map(|&x| x as i16));
        out.extend(self.resources[observer]);
        out.extend(self.resources[opponent]);
        out.extend(self.bank);
        out.extend(self.dev_hand[observer].iter().map(|&x| x as i16));
        out.push(self.dev_hand[opponent].iter().map(|&x| x as i16).sum());
        out.extend(self.dev_played[observer].iter().map(|&x| x as i16));
        out.extend(self.dev_played[opponent].iter().map(|&x| x as i16));
        out.push(self.visible_vp[observer] as i16);
        out.push(self.visible_vp[opponent] as i16);
        out.push(self.actual_vp[observer] as i16);
        out.extend(self.roads_available.iter().map(|&x| x as i16));
        out.extend(self.settlements_available.iter().map(|&x| x as i16));
        out.extend(self.cities_available.iter().map(|&x| x as i16));
        out.extend(self.road_lengths.iter().map(|&x| x as i16));
        out.extend(self.dev_owned_at_start[observer].iter().map(|&x| x as i16));
        out.push(self.has_road[observer] as i16);
        out.push(self.has_road[opponent] as i16);
        out.push(self.has_army[observer] as i16);
        out.push(self.has_army[opponent] as i16);
        out.push(self.robber_tile as i16);
        out.push(self.current_player as i16);
        out.push(self.current_turn as i16);
        out.push(self.prompt as i16);
        out.push(self.initial as i16);
        out.extend(self.has_rolled.iter().map(|&x| x as i16));
        out.extend(self.played_dev_this_turn.iter().map(|&x| x as i16));
        out.extend(self.discard_counts.iter().map(|&x| x as i16));
        out.push(self.road_building as i16);
        out.push(self.free_roads as i16);
        out.push(self.dev_len as i16);
        out.push(self.num_turns as i16);
        out.extend(self.dice.counts.iter().map(|&x| x as i16));
        out.push(self.dice.cards_left as i16);
        out.extend(self.dice.recent.iter().map(|&x| x as i16));
        out.push(self.dice.recent_len as i16);
        out.extend(self.dice.total_sevens.iter().map(|&x| x as i16));
        out.extend(self.dice.seen_player.iter().map(|&x| x as i16));
        out.push(self.dice.streak_color as i16);
        out.push(self.dice.streak_count as i16);
        out
    }
}

#[derive(Clone)]
pub struct RankedState {
    pub seed: u64,
    pub colors: [u8; 2], // 0=RED, 1=BLUE, in seat order
    pub tile_resources: [u8; 19],
    pub tile_numbers: [u8; 19],
    pub port_types: [i8; 9], // -1=3:1, otherwise resource id
    pub dev_deck: [u8; 25],  // Python list order; draws pop from the end
    pub dev_len: u8,
    pub robber_tile: u8,
    pub rng: PythonRandom,
    pub dice: BalancedDice,

    pub vertices: [i8; 54], // -1 empty; seat 0/1 settlement; seat+2 city
    pub edges: [i8; 72],
    pub components: [Vec<u64>; 2],
    pub resources: [[i16; 5]; 2],
    pub bank: [i16; 5],
    pub dev_hand: [[u8; 5]; 2],
    pub dev_played: [[u8; 5]; 2],
    pub dev_owned_at_start: [[bool; 4]; 2],
    pub visible_vp: [i8; 2],
    pub actual_vp: [i8; 2],
    pub roads_available: [u8; 2],
    pub settlements_available: [u8; 2],
    pub cities_available: [u8; 2],
    pub has_road: [bool; 2],
    pub has_army: [bool; 2],
    pub road_cache: [u8; 2],
    pub road_lengths: [u8; 2],
    pub road_owner: i8,
    pub board_road_length: u8,
    pub army_owner: i8,
    pub army_size: u8,

    pub current_player: u8,
    pub current_turn: u8,
    pub prompt: u8,
    pub initial: bool,
    pub has_rolled: [bool; 2],
    pub played_dev_this_turn: [bool; 2],
    pub discard_counts: [u8; 2],
    pub road_building: bool,
    pub free_roads: u8,
    pub num_turns: u16,
    pub last_settlement: [u8; 2],
}

impl RankedState {
    pub fn new(seed: u64) -> Self {
        let mut rng = PythonRandom::from_u64(seed);
        let colors = rng.sample_all([0u8, 1u8]);
        let sampled_ports = rng.sample_all(PORT_TEMPLATE);
        let sampled_resources = rng.sample_all(TILE_RESOURCE_TEMPLATE);
        let _sampled_numbers = rng.sample_all(NUMBER_TEMPLATE);

        let mut port_types = [0i8; 9];
        for i in 0..PORT_VERTICES.len() {
            port_types[i] = sampled_ports[8 - i];
        }
        let mut tile_resources = [0u8; 19];
        for i in 0..19 {
            tile_resources[i] = sampled_resources[18 - i];
        }
        let robber_tile = tile_resources.iter().position(|&r| r == DESERT).unwrap() as u8;
        let mut tile_numbers = [0u8; 19];
        let mut number_index = 0;
        for &tile in &SPIRAL_TILE_ORDER {
            let tile = tile as usize;
            if tile_resources[tile] != DESERT {
                tile_numbers[tile] = SPIRAL_NUMBERS[number_index];
                number_index += 1;
            }
        }
        let mut dev_deck = DEV_TEMPLATE;
        rng.shuffle(&mut dev_deck);
        Self {
            seed,
            colors,
            tile_resources,
            tile_numbers,
            port_types,
            dev_deck,
            dev_len: 25,
            robber_tile,
            rng,
            dice: BalancedDice::new(),
            vertices: [-1; 54],
            edges: [-1; 72],
            components: [Vec::new(), Vec::new()],
            resources: [[0; 5]; 2],
            bank: [19; 5],
            dev_hand: [[0; 5]; 2],
            dev_played: [[0; 5]; 2],
            dev_owned_at_start: [[false; 4]; 2],
            visible_vp: [0; 2],
            actual_vp: [0; 2],
            roads_available: [15; 2],
            settlements_available: [5; 2],
            cities_available: [4; 2],
            has_road: [false; 2],
            has_army: [false; 2],
            road_cache: [0; 2],
            road_lengths: [0; 2],
            road_owner: -1,
            board_road_length: 0,
            army_owner: -1,
            army_size: 0,
            current_player: 0,
            current_turn: 0,
            prompt: PROMPT_INITIAL_SETTLEMENT,
            initial: true,
            has_rolled: [false; 2],
            played_dev_this_turn: [false; 2],
            discard_counts: [0; 2],
            road_building: false,
            free_roads: 0,
            num_turns: 0,
            last_settlement: [255; 2],
        }
    }

    pub fn seat_for_color(&self, color: u8) -> usize {
        if self.colors[0] == color { 0 } else { 1 }
    }

    pub fn total_resources(&self, player: usize) -> i16 {
        self.resources[player].iter().sum()
    }
}
