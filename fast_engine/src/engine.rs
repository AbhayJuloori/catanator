use crate::state::*;
use crate::topology::{
    ACTION_A, ACTION_B, ACTION_C, ACTION_DIM, ACTION_KIND, EDGE_VERTICES, PORT_VERTICES,
    TILE_VERTICES, VERTEX_EDGES, VERTEX_NEIGHBORS,
};

#[derive(Clone, Copy, Debug, Default)]
pub struct ActionOutcome {
    pub roll_total: Option<u8>,
    pub stolen_resource: Option<u8>,
    pub bought_dev: Option<u8>,
}

const ROAD_COST: [i16; 5] = [1, 1, 0, 0, 0];
const SETTLEMENT_COST: [i16; 5] = [1, 1, 1, 1, 0];
const CITY_COST: [i16; 5] = [0, 0, 0, 2, 3];
const DEV_COST: [i16; 5] = [0, 0, 1, 1, 1];

impl RankedState {
    fn can_afford(&self, player: usize, cost: &[i16; 5]) -> bool {
        (0..5).all(|r| self.resources[player][r] >= cost[r])
    }

    fn pay(&mut self, player: usize, cost: &[i16; 5]) {
        for r in 0..5 {
            self.resources[player][r] -= cost[r];
            self.bank[r] += cost[r];
        }
    }

    fn vertex_owner(&self, vertex: usize) -> i8 {
        let value = self.vertices[vertex];
        if value < 0 { -1 } else { value % 2 }
    }

    fn incident_player_road(&self, player: usize, vertex: usize) -> bool {
        VERTEX_EDGES[vertex]
            .iter()
            .any(|&edge| edge >= 0 && self.edges[edge as usize] == player as i8)
    }

    fn component_nodes(&self, player: usize) -> u64 {
        self.components[player]
            .iter()
            .fold(0, |all, &component| all | component)
    }

    fn settlement_legal(&self, player: usize, vertex: usize, initial: bool) -> bool {
        if vertex >= 54 || self.vertices[vertex] >= 0 {
            return false;
        }
        if VERTEX_NEIGHBORS[vertex]
            .iter()
            .any(|&v| v >= 0 && self.vertices[v as usize] >= 0)
        {
            return false;
        }
        initial
            || (self.settlements_available[player] > 0
                && self.can_afford(player, &SETTLEMENT_COST)
                && self.component_nodes(player) & (1u64 << vertex) != 0)
    }

    fn road_legal(&self, player: usize, edge: usize, free: bool) -> bool {
        edge < 72
            && self.edges[edge] < 0
            && self.roads_available[player] > 0
            && (free || self.can_afford(player, &ROAD_COST))
            && EDGE_VERTICES[edge]
                .iter()
                .any(|&v| self.component_nodes(player) & (1u64 << v) != 0)
    }

    fn initial_road_legal(&self, player: usize, edge: usize) -> bool {
        edge < 72
            && self.edges[edge] < 0
            && self.roads_available[player] > 0
            && EDGE_VERTICES[edge].contains(&self.last_settlement[player])
    }

    fn can_play_dev(&self, player: usize, card: u8) -> bool {
        let owned = match card {
            KNIGHT => self.dev_owned_at_start[player][0],
            MONOPOLY => self.dev_owned_at_start[player][1],
            YEAR_OF_PLENTY => self.dev_owned_at_start[player][2],
            ROAD_BUILDING => self.dev_owned_at_start[player][3],
            _ => false,
        };
        !self.played_dev_this_turn[player] && owned && self.dev_hand[player][card as usize] > 0
    }

    fn yop_options(&self) -> ([bool; 5], [[bool; 5]; 5]) {
        let mut singles = [false; 5];
        let mut pairs = [[false; 5]; 5];
        for first in 0..5 {
            for second in first..5 {
                let need_first = if first == second { 2 } else { 1 };
                if self.bank[first] >= need_first && (first == second || self.bank[second] >= 1) {
                    pairs[first][second] = true;
                } else {
                    if self.bank[first] >= 1 {
                        singles[first] = true;
                    }
                    if self.bank[second] >= 1 {
                        singles[second] = true;
                    }
                }
            }
        }
        (singles, pairs)
    }

    fn tile_is_protected(&self, actor: usize, tile: usize) -> bool {
        TILE_VERTICES[tile].iter().any(|&vertex| {
            let owner = self.vertex_owner(vertex as usize);
            owner >= 0 && owner != actor as i8 && self.visible_vp[owner as usize] < 3
        })
    }

    fn robber_original_legal(&self, actor: usize, tile: usize, victim_color: u8) -> bool {
        if tile == self.robber_tile as usize {
            return false;
        }
        let opponent = 1 - actor;
        let stealable = TILE_VERTICES[tile].iter().any(|&vertex| {
            self.vertex_owner(vertex as usize) == opponent as i8
                && self.total_resources(opponent) > 0
        });
        if stealable {
            victim_color == self.colors[opponent]
        } else {
            victim_color == 2
        }
    }

    fn any_unprotected_robber_action(&self, actor: usize) -> bool {
        (0..19)
            .any(|tile| tile != self.robber_tile as usize && !self.tile_is_protected(actor, tile))
    }

    fn port_rate(&self, player: usize, resource: usize) -> i16 {
        let mut any = false;
        let mut specific = false;
        for (port, vertices) in PORT_VERTICES.iter().enumerate() {
            if vertices
                .iter()
                .any(|&v| self.vertex_owner(v as usize) == player as i8)
            {
                if self.port_types[port] < 0 {
                    any = true;
                } else if self.port_types[port] as usize == resource {
                    specific = true;
                }
            }
        }
        if specific {
            2
        } else if any {
            3
        } else {
            4
        }
    }

    pub fn is_legal(&self, action: usize) -> bool {
        if action >= ACTION_DIM {
            return false;
        }
        let player = self.current_player as usize;
        let kind = ACTION_KIND[action];
        let a = ACTION_A[action] as usize;
        let b = ACTION_B[action] as usize;
        let c = ACTION_C[action] as usize;
        match self.prompt {
            PROMPT_INITIAL_SETTLEMENT => kind == 2 && self.settlement_legal(player, a, true),
            PROMPT_INITIAL_ROAD => kind == 1 && self.initial_road_legal(player, a),
            PROMPT_DISCARD => {
                kind == 4 && self.discard_counts[player] > 0 && self.resources[player][a] > 0
            }
            PROMPT_ROBBER => {
                if kind != 7 || !self.robber_original_legal(player, a, b as u8) {
                    return false;
                }
                !self.any_unprotected_robber_action(player) || !self.tile_is_protected(player, a)
            }
            PROMPT_PLAY => {
                if self.road_building {
                    return kind == 1 && self.road_legal(player, a, true);
                }
                match kind {
                    8 => self.can_play_dev(player, KNIGHT),
                    9 => self.can_play_dev(player, MONOPOLY),
                    10 => {
                        self.can_play_dev(player, ROAD_BUILDING)
                            && (0..72).any(|edge| self.road_legal(player, edge, true))
                    }
                    11 => {
                        if !self.can_play_dev(player, YEAR_OF_PLENTY) {
                            return false;
                        }
                        let (singles, pairs) = self.yop_options();
                        if b == 255 {
                            singles[a]
                        } else {
                            pairs[a.min(b)][a.max(b)]
                        }
                    }
                    12 => !self.has_rolled[player],
                    _ if !self.has_rolled[player] => false,
                    5 => true,
                    1 => self.road_legal(player, a, false),
                    2 => self.settlement_legal(player, a, false),
                    0 => {
                        a < 54
                            && self.vertices[a] == player as i8
                            && self.cities_available[player] > 0
                            && self.can_afford(player, &CITY_COST)
                    }
                    3 => self.dev_len > 0 && self.can_afford(player, &DEV_COST),
                    6 => {
                        b as i16 == self.port_rate(player, a)
                            && a != c
                            && self.resources[player][a] >= b as i16
                            && self.bank[c] > 0
                    }
                    _ => false,
                }
            }
            _ => false,
        }
    }

    pub fn legal_actions(&self) -> Vec<u16> {
        let mut actions = Vec::with_capacity(32);
        self.fill_legal_actions(&mut actions);
        actions
    }

    pub fn fill_legal_actions(&self, actions: &mut Vec<u16>) {
        actions.clear();
        let player = self.current_player as usize;
        match self.prompt {
            PROMPT_INITIAL_SETTLEMENT => {
                for action in 126..180 {
                    let vertex = ACTION_A[action] as usize;
                    if self.vertices[vertex] < 0
                        && !VERTEX_NEIGHBORS[vertex]
                            .iter()
                            .any(|&v| v >= 0 && self.vertices[v as usize] >= 0)
                    {
                        actions.push(action as u16);
                    }
                }
            }
            PROMPT_INITIAL_ROAD => {
                if self.roads_available[player] > 0 {
                    for action in 54..126 {
                        let edge = ACTION_A[action] as usize;
                        if self.edges[edge] < 0
                            && EDGE_VERTICES[edge].contains(&self.last_settlement[player])
                        {
                            actions.push(action as u16);
                        }
                    }
                }
            }
            PROMPT_DISCARD => {
                for action in 181..186 {
                    if self.resources[player][ACTION_A[action] as usize] > 0 {
                        actions.push(action as u16);
                    }
                }
            }
            PROMPT_ROBBER => {
                let permit_protected = !self.any_unprotected_robber_action(player);
                for action in 247..304 {
                    let tile = ACTION_A[action] as usize;
                    if self.robber_original_legal(player, tile, ACTION_B[action])
                        && (permit_protected || !self.tile_is_protected(player, tile))
                    {
                        actions.push(action as u16);
                    }
                }
            }
            PROMPT_PLAY if self.road_building => {
                let nodes = self.component_nodes(player);
                if self.roads_available[player] > 0 {
                    for action in 54..126 {
                        let edge = ACTION_A[action] as usize;
                        if self.edges[edge] < 0
                            && EDGE_VERTICES[edge]
                                .iter()
                                .any(|&v| nodes & (1u64 << v) != 0)
                        {
                            actions.push(action as u16);
                        }
                    }
                }
            }
            PROMPT_PLAY => {
                if self.has_rolled[player] {
                    if self.cities_available[player] > 0 && self.can_afford(player, &CITY_COST) {
                        for action in 0..54 {
                            if self.vertices[ACTION_A[action] as usize] == player as i8 {
                                actions.push(action as u16);
                            }
                        }
                    }
                    if self.roads_available[player] > 0 && self.can_afford(player, &ROAD_COST) {
                        let nodes = self.component_nodes(player);
                        for action in 54..126 {
                            let edge = ACTION_A[action] as usize;
                            if self.edges[edge] < 0
                                && EDGE_VERTICES[edge]
                                    .iter()
                                    .any(|&v| nodes & (1u64 << v) != 0)
                            {
                                actions.push(action as u16);
                            }
                        }
                    }
                    if self.settlements_available[player] > 0
                        && self.can_afford(player, &SETTLEMENT_COST)
                    {
                        let nodes = self.component_nodes(player);
                        for action in 126..180 {
                            let vertex = ACTION_A[action] as usize;
                            if self.vertices[vertex] < 0
                                && nodes & (1u64 << vertex) != 0
                                && !VERTEX_NEIGHBORS[vertex]
                                    .iter()
                                    .any(|&v| v >= 0 && self.vertices[v as usize] >= 0)
                            {
                                actions.push(action as u16);
                            }
                        }
                    }
                    if self.dev_len > 0 && self.can_afford(player, &DEV_COST) {
                        actions.push(180);
                    }
                    actions.push(186);
                    let rates = [
                        self.port_rate(player, 0),
                        self.port_rate(player, 1),
                        self.port_rate(player, 2),
                        self.port_rate(player, 3),
                        self.port_rate(player, 4),
                    ];
                    for action in 187..247 {
                        let give = ACTION_A[action] as usize;
                        let rate = ACTION_B[action] as i16;
                        let receive = ACTION_C[action] as usize;
                        if rate == rates[give]
                            && give != receive
                            && self.resources[player][give] >= rate
                            && self.bank[receive] > 0
                        {
                            actions.push(action as u16);
                        }
                    }
                }

                if self.can_play_dev(player, KNIGHT) {
                    actions.push(304);
                }
                if self.can_play_dev(player, MONOPOLY) {
                    actions.extend(305..310);
                }
                if self.can_play_dev(player, ROAD_BUILDING)
                    && (0..72).any(|edge| self.road_legal(player, edge, true))
                {
                    actions.push(310);
                }
                if self.can_play_dev(player, YEAR_OF_PLENTY) {
                    let (singles, pairs) = self.yop_options();
                    for action in 311..331 {
                        let first = ACTION_A[action] as usize;
                        let second = ACTION_B[action];
                        if (second == 255 && singles[first])
                            || (second != 255
                                && pairs[first.min(second as usize)][first.max(second as usize)])
                        {
                            actions.push(action as u16);
                        }
                    }
                }
                if !self.has_rolled[player] {
                    actions.push(331);
                }
            }
            _ => {}
        }
    }

    fn maintain_longest_road(&mut self, previous: i8, current: i8) {
        if current < 0 || previous == current {
            return;
        }
        let winner = current as usize;
        self.has_road[winner] = true;
        self.visible_vp[winner] += 2;
        self.actual_vp[winner] += 2;
        if previous >= 0 {
            let loser = previous as usize;
            self.has_road[loser] = false;
            self.visible_vp[loser] -= 2;
            self.actual_vp[loser] -= 2;
        }
    }

    fn longest_on_component(&self, player: usize, component: u64) -> u8 {
        fn walk(state: &RankedState, player: usize, at: u8, used: u128) -> u8 {
            let mut best = 0;
            for &edge in &VERTEX_EDGES[at as usize] {
                if edge < 0 || state.edges[edge as usize] != player as i8 {
                    continue;
                }
                let bit = 1u128 << edge;
                if used & bit != 0 {
                    continue;
                }
                let [x, y] = EDGE_VERTICES[edge as usize];
                let next = if x == at { y } else { x };
                if state.vertex_owner(next as usize) >= 0
                    && state.vertex_owner(next as usize) != player as i8
                {
                    continue;
                }
                best = best.max(1 + walk(state, player, next, used | bit));
            }
            best
        }
        let mut best = 0;
        let mut nodes = component;
        while nodes != 0 {
            let start = nodes.trailing_zeros() as u8;
            nodes &= nodes - 1;
            best = best.max(walk(self, player, start, 0));
        }
        best
    }

    fn longest_road_for(&self, player: usize) -> u8 {
        self.components[player]
            .iter()
            .map(|&c| self.longest_on_component(player, c))
            .max()
            .unwrap_or(0)
    }

    fn component_index(&self, player: usize, vertex: u8) -> Option<usize> {
        self.components[player]
            .iter()
            .position(|&component| component & (1u64 << vertex) != 0)
    }

    fn dfs_walk(&self, player: usize, start: u8) -> u64 {
        let mut agenda = vec![start];
        let mut visited = 0u64;
        while let Some(node) = agenda.pop() {
            let bit = 1u64 << node;
            if visited & bit != 0 {
                continue;
            }
            visited |= bit;
            let owner = self.vertex_owner(node as usize);
            if owner >= 0 && owner != player as i8 {
                continue;
            }
            for &edge in &VERTEX_EDGES[node as usize] {
                if edge < 0 || self.edges[edge as usize] != player as i8 {
                    continue;
                }
                let [a, b] = EDGE_VERTICES[edge as usize];
                let next = if a == node { b } else { a };
                if visited & (1u64 << next) == 0 {
                    agenda.push(next);
                }
            }
        }
        visited
    }

    fn connect_road_component(&mut self, player: usize, edge: usize) {
        let [a, b] = EDGE_VERTICES[edge];
        let ai = self.component_index(player, a);
        let bi = self.component_index(player, b);
        if ai.is_none() && self.vertex_owner(a as usize) != (1 - player) as i8 {
            self.components[player][bi.unwrap()] |= 1u64 << a;
        } else if bi.is_none() && self.vertex_owner(b as usize) != (1 - player) as i8 {
            self.components[player][ai.unwrap()] |= 1u64 << b;
        } else if let (Some(ai), Some(bi)) = (ai, bi) {
            if ai != bi {
                let union = self.components[player][ai] | self.components[player][bi];
                self.components[player][ai] = union;
                self.components[player].remove(bi);
            }
        }
    }

    fn place_road(&mut self, player: usize, edge: usize, free: bool) {
        let previous = self.road_owner;
        if !free {
            self.pay(player, &ROAD_COST);
        }
        self.edges[edge] = player as i8;
        self.roads_available[player] -= 1;
        self.connect_road_component(player, edge);
        let candidate = self.longest_road_for(player);
        self.road_cache[player] = self.road_cache[player].max(candidate);
        self.road_lengths = self.road_cache;
        if candidate >= 5 && candidate > self.board_road_length {
            self.road_owner = player as i8;
            self.board_road_length = candidate;
        }
        self.maintain_longest_road(previous, self.road_owner);
    }

    fn place_settlement(&mut self, player: usize, vertex: usize, free: bool) {
        let previous = self.road_owner;
        if !free {
            self.pay(player, &SETTLEMENT_COST);
        }
        self.vertices[vertex] = player as i8;
        self.settlements_available[player] -= 1;
        self.visible_vp[player] += 1;
        self.actual_vp[player] += 1;
        self.last_settlement[player] = vertex as u8;
        if free {
            self.components[player].push(1u64 << vertex);
        } else {
            let opponent = 1 - player;
            let incident: Vec<usize> = VERTEX_EDGES[vertex]
                .iter()
                .filter_map(|&edge| {
                    (edge >= 0 && self.edges[edge as usize] == opponent as i8)
                        .then_some(edge as usize)
                })
                .collect();
            if incident.len() == 2 {
                let other = |edge: usize| {
                    let [a, b] = EDGE_VERTICES[edge];
                    if a as usize == vertex { b } else { a }
                };
                let left = self.dfs_walk(opponent, other(incident[0]));
                let right = self.dfs_walk(opponent, other(incident[1]));
                let index = self.component_index(opponent, vertex as u8).unwrap();
                self.components[opponent].remove(index);
                self.components[opponent].push(left);
                self.components[opponent].push(right);
                self.road_cache[opponent] = self.longest_road_for(opponent);
                self.road_lengths = self.road_cache;
                let new_owner = if self.road_cache[0] >= self.road_cache[1] {
                    0
                } else {
                    1
                };
                self.road_owner = new_owner;
                self.board_road_length = self.road_cache[new_owner as usize];
                self.maintain_longest_road(previous, self.road_owner);
            } else if incident.len() == 1 {
                if let Some(index) = self.component_index(opponent, vertex as u8) {
                    self.components[opponent][index] &= !(1u64 << vertex);
                }
            }
        }
    }

    fn opening_resource_grant(&mut self, player: usize, vertex: usize) {
        for tile in 0..19 {
            if TILE_VERTICES[tile].contains(&(vertex as u8)) {
                let resource = self.tile_resources[tile];
                if resource != DESERT {
                    self.bank[resource as usize] -= 1;
                    self.resources[player][resource as usize] += 1;
                }
            }
        }
    }

    fn distribute(&mut self, total: u8) {
        let mut payout = [[0i16; 5]; 2];
        for tile in 0..19 {
            if self.tile_numbers[tile] != total || self.robber_tile as usize == tile {
                continue;
            }
            let resource = self.tile_resources[tile] as usize;
            for &vertex in &TILE_VERTICES[tile] {
                let value = self.vertices[vertex as usize];
                if value < 0 {
                    continue;
                }
                let owner = (value % 2) as usize;
                payout[owner][resource] += if value >= 2 { 2 } else { 1 };
            }
        }
        for resource in 0..5 {
            let need = payout[0][resource] + payout[1][resource];
            if need > self.bank[resource] {
                continue;
            }
            self.bank[resource] -= need;
            for player in 0..2 {
                self.resources[player][resource] += payout[player][resource];
            }
        }
    }

    fn play_dev(&mut self, player: usize, card: u8) {
        self.dev_hand[player][card as usize] -= 1;
        self.dev_played[player][card as usize] += 1;
        self.played_dev_this_turn[player] = true;
    }

    fn update_largest_army(&mut self, player: usize) {
        let candidate = self.dev_played[player][KNIGHT as usize];
        if candidate < 3 {
            return;
        }
        if self.army_owner < 0 {
            self.army_owner = player as i8;
            self.army_size = candidate;
            self.has_army[player] = true;
            self.visible_vp[player] += 2;
            self.actual_vp[player] += 2;
        } else if self.army_owner != player as i8 && candidate > self.army_size {
            let loser = self.army_owner as usize;
            self.has_army[loser] = false;
            self.visible_vp[loser] -= 2;
            self.actual_vp[loser] -= 2;
            self.army_owner = player as i8;
            self.army_size = candidate;
            self.has_army[player] = true;
            self.visible_vp[player] += 2;
            self.actual_vp[player] += 2;
        } else if self.army_owner == player as i8 {
            self.army_size = candidate;
        }
    }

    fn random_resource_from(&mut self, victim: usize) -> u8 {
        let pick = self.rng.randbelow(self.total_resources(victim) as usize);
        let mut cursor = 0usize;
        for resource in 0..5 {
            cursor += self.resources[victim][resource] as usize;
            if pick < cursor {
                return resource as u8;
            }
        }
        unreachable!()
    }

    pub fn apply(
        &mut self,
        action: usize,
        forced_roll: Option<u8>,
        forced_steal: Option<u8>,
    ) -> Result<ActionOutcome, String> {
        if !self.is_legal(action) {
            return Err(format!("illegal action {action}"));
        }
        let player = self.current_player as usize;
        let kind = ACTION_KIND[action];
        let a = ACTION_A[action] as usize;
        let b = ACTION_B[action] as usize;
        let c = ACTION_C[action] as usize;
        let mut outcome = ActionOutcome::default();
        match kind {
            0 => {
                self.pay(player, &CITY_COST);
                self.vertices[a] = player as i8 + 2;
                self.settlements_available[player] += 1;
                self.cities_available[player] -= 1;
                self.visible_vp[player] += 1;
                self.actual_vp[player] += 1;
            }
            1 => {
                if self.initial {
                    self.edges[a] = player as i8;
                    self.roads_available[player] -= 1;
                    self.connect_road_component(player, a);
                    self.road_cache[player] = self.longest_road_for(player);
                    let buildings = [
                        5 - self.settlements_available[0],
                        5 - self.settlements_available[1],
                    ];
                    let total = buildings[0] + buildings[1];
                    if total < 2 {
                        self.current_player = 1 - self.current_player;
                        self.current_turn = self.current_player;
                        self.num_turns += 1;
                        self.prompt = PROMPT_INITIAL_SETTLEMENT;
                    } else if total == 2 {
                        self.prompt = PROMPT_INITIAL_SETTLEMENT;
                    } else if total == 4 {
                        self.initial = false;
                        self.prompt = PROMPT_PLAY;
                    } else {
                        self.current_player = 1 - self.current_player;
                        self.current_turn = self.current_player;
                        self.num_turns += 1;
                        self.prompt = PROMPT_INITIAL_SETTLEMENT;
                    }
                } else if self.road_building {
                    self.place_road(player, a, true);
                    self.free_roads -= 1;
                    if self.free_roads == 0
                        || !(0..72).any(|edge| self.road_legal(player, edge, true))
                    {
                        self.road_building = false;
                        self.free_roads = 0;
                    }
                } else {
                    self.place_road(player, a, false);
                }
            }
            2 => {
                if self.initial {
                    self.place_settlement(player, a, true);
                    if 5 - self.settlements_available[player] == 2 {
                        self.opening_resource_grant(player, a);
                    }
                    self.prompt = PROMPT_INITIAL_ROAD;
                } else {
                    self.place_settlement(player, a, false);
                }
            }
            3 => {
                self.pay(player, &DEV_COST);
                self.dev_len -= 1;
                let card = self.dev_deck[self.dev_len as usize];
                self.dev_hand[player][card as usize] += 1;
                outcome.bought_dev = Some(card);
                if card == VICTORY_POINT {
                    self.actual_vp[player] += 1;
                }
            }
            4 => {
                self.resources[player][a] -= 1;
                self.bank[a] += 1;
                self.discard_counts[player] -= 1;
                if self.discard_counts[player] == 0 {
                    if let Some(next) = ((player + 1)..2).find(|&p| self.discard_counts[p] > 0) {
                        self.current_player = next as u8;
                    } else {
                        self.current_player = self.current_turn;
                        self.discard_counts = [0; 2];
                        self.prompt = PROMPT_ROBBER;
                    }
                }
            }
            5 => {
                self.played_dev_this_turn[player] = false;
                self.has_rolled[player] = false;
                self.dev_owned_at_start[player] = [
                    self.dev_hand[player][KNIGHT as usize] > 0,
                    self.dev_hand[player][MONOPOLY as usize] > 0,
                    self.dev_hand[player][YEAR_OF_PLENTY as usize] > 0,
                    self.dev_hand[player][ROAD_BUILDING as usize] > 0,
                ];
                self.current_player = 1 - self.current_player;
                self.current_turn = self.current_player;
                self.num_turns += 1;
            }
            6 => {
                self.resources[player][a] -= b as i16;
                self.bank[a] += b as i16;
                self.resources[player][c] += 1;
                self.bank[c] -= 1;
            }
            7 => {
                let tile = a;
                if b != 2 {
                    let victim = self.seat_for_color(b as u8);
                    let resource =
                        forced_steal.unwrap_or_else(|| self.random_resource_from(victim));
                    if self.resources[victim][resource as usize] <= 0 {
                        return Err(format!("victim lacks forced resource {resource}"));
                    }
                    self.resources[victim][resource as usize] -= 1;
                    self.resources[player][resource as usize] += 1;
                    outcome.stolen_resource = Some(resource);
                }
                self.robber_tile = tile as u8;
                self.prompt = PROMPT_PLAY;
            }
            8 => {
                self.play_dev(player, KNIGHT);
                self.update_largest_army(player);
                self.prompt = PROMPT_ROBBER;
            }
            9 => {
                self.play_dev(player, MONOPOLY);
                let opponent = 1 - player;
                let amount = self.resources[opponent][a];
                self.resources[opponent][a] = 0;
                self.resources[player][a] += amount;
            }
            10 => {
                self.play_dev(player, ROAD_BUILDING);
                self.road_building = true;
                self.free_roads = 2;
            }
            11 => {
                self.play_dev(player, YEAR_OF_PLENTY);
                self.bank[a] -= 1;
                self.resources[player][a] += 1;
                if b != 255 {
                    self.bank[b] -= 1;
                    self.resources[player][b] += 1;
                }
            }
            12 => {
                self.has_rolled[player] = true;
                let color = self.colors[player] as usize;
                let total = if let Some(total) = forced_roll {
                    if !self.dice.consume(color, total) {
                        return Err(format!("balanced total {total} exhausted"));
                    }
                    total
                } else {
                    self.dice.draw(color, &mut self.rng)
                };
                outcome.roll_total = Some(total);
                if total == 7 {
                    for p in 0..2 {
                        let count = self.total_resources(p);
                        self.discard_counts[p] = if count > 9 { (count / 2) as u8 } else { 0 };
                    }
                    if let Some(first) = (0..2).find(|&p| self.discard_counts[p] > 0) {
                        self.current_player = first as u8;
                        self.prompt = PROMPT_DISCARD;
                    } else {
                        self.prompt = PROMPT_ROBBER;
                    }
                } else {
                    self.distribute(total);
                    self.prompt = PROMPT_PLAY;
                }
            }
            _ => unreachable!(),
        }
        Ok(outcome)
    }

    pub fn winner(&self) -> i8 {
        let mut result = -1;
        for player in 0..2 {
            if self.actual_vp[player] >= 15 {
                result = player as i8;
            }
        }
        result
    }
}
