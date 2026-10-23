mod champion;
mod engine;
mod evaluation;
mod observation;
mod planner;
mod rng;
mod state;
mod topology;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde_json::{Value, json};
use state::RankedState;

const RESOURCE_NAMES: [&str; 5] = ["wood", "brick", "sheep", "wheat", "ore"];
const DEV_NAMES: [&str; 5] = [
    "knight",
    "year_of_plenty",
    "monopoly",
    "road_building",
    "victory_point",
];
const PROMPT_NAMES: [&str; 5] = [
    "initial_settlement",
    "initial_road",
    "play",
    "discard",
    "robber",
];

fn action_value(action: usize) -> PyResult<Value> {
    if action >= topology::ACTION_DIM {
        return Err(PyValueError::new_err("action is outside the action space"));
    }
    let kind = topology::ACTION_KIND[action];
    let a = topology::ACTION_A[action];
    let b = topology::ACTION_B[action];
    let c = topology::ACTION_C[action];
    let payload = match kind {
        0 => json!({"id": action, "kind": "build_city", "vertex": a}),
        1 => json!({"id": action, "kind": "build_road", "edge": a}),
        2 => json!({"id": action, "kind": "build_settlement", "vertex": a}),
        3 => json!({"id": action, "kind": "buy_dev_card"}),
        4 => json!({"id": action, "kind": "discard", "resource": RESOURCE_NAMES[a as usize]}),
        5 => json!({"id": action, "kind": "end_turn"}),
        6 => json!({
            "id": action,
            "kind": "maritime_trade",
            "give": RESOURCE_NAMES[a as usize],
            "rate": b,
            "receive": RESOURCE_NAMES[c as usize],
        }),
        7 => json!({
            "id": action,
            "kind": "move_robber",
            "tile": a,
            "victim_color": if b == 2 { Value::Null } else { json!(b) },
        }),
        8 => json!({"id": action, "kind": "play_knight"}),
        9 => json!({"id": action, "kind": "play_monopoly", "resource": RESOURCE_NAMES[a as usize]}),
        10 => json!({"id": action, "kind": "play_road_building"}),
        11 => json!({
            "id": action,
            "kind": "play_year_of_plenty",
            "resources": if b == 255 {
                vec![RESOURCE_NAMES[a as usize]]
            } else {
                vec![RESOURCE_NAMES[a as usize], RESOURCE_NAMES[b as usize]]
            },
        }),
        12 => json!({"id": action, "kind": "roll"}),
        _ => return Err(PyValueError::new_err("unknown action kind")),
    };
    Ok(payload)
}

#[pyfunction]
fn decode_action_json(action: usize) -> PyResult<String> {
    Ok(action_value(action)?.to_string())
}

#[pyfunction]
fn topology_json() -> String {
    json!({
        "land_coords": topology::LAND_COORDS,
        "tile_vertices": topology::TILE_VERTICES,
        "edge_vertices": topology::EDGE_VERTICES.to_vec(),
        "port_vertices": topology::PORT_VERTICES,
        "action_dim": topology::ACTION_DIM,
    })
    .to_string()
}

#[pyclass(module = "catan_ranked_fast")]
#[derive(Clone)]
struct TurnSearch {
    graph: planner::TurnGraph,
    observer: usize,
    opponent_model: planner::OpponentModel,
}

#[pymethods]
impl TurnSearch {
    fn observations(&self) -> Vec<Vec<f32>> {
        self.graph
            .leaves
            .iter()
            .map(|leaf| observation::encode(&leaf.state, self.observer))
            .collect()
    }

    fn opponent_observations(&self) -> Vec<Vec<f32>> {
        self.graph
            .leaves
            .iter()
            .map(|leaf| observation::encode(&leaf.state, 1 - self.observer))
            .collect()
    }

    fn plans(&self) -> Vec<Vec<u16>> {
        self.graph
            .leaves
            .iter()
            .map(|leaf| leaf.actions.clone())
            .collect()
    }

    fn linear_values(&self) -> Vec<f64> {
        self.graph
            .leaves
            .iter()
            .map(|leaf| evaluation::tuned_consensus(&leaf.state, self.observer, 0.0))
            .collect()
    }

    fn opponent_linear_values(&self) -> Vec<f64> {
        self.graph
            .leaves
            .iter()
            .map(|leaf| evaluation::tuned_consensus(&leaf.state, 1 - self.observer, 0.0))
            .collect()
    }

    fn weighted_linear_values(&self, weights: Vec<f64>) -> PyResult<Vec<f64>> {
        let weights: [f64; evaluation::WEIGHT_COUNT] =
            weights.try_into().map_err(|values: Vec<f64>| {
                PyValueError::new_err(format!(
                    "expected {} weights, got {}",
                    evaluation::WEIGHT_COUNT,
                    values.len()
                ))
            })?;
        if weights.iter().any(|value| !value.is_finite()) {
            return Err(PyValueError::new_err("weights must be finite"));
        }
        Ok(self
            .graph
            .leaves
            .iter()
            .map(|leaf| evaluation::weighted_consensus(&leaf.state, self.observer, 0.0, &weights))
            .collect())
    }

    fn opponent_weighted_linear_values(&self, weights: Vec<f64>) -> PyResult<Vec<f64>> {
        let weights: [f64; evaluation::WEIGHT_COUNT] =
            weights.try_into().map_err(|values: Vec<f64>| {
                PyValueError::new_err(format!(
                    "expected {} weights, got {}",
                    evaluation::WEIGHT_COUNT,
                    values.len()
                ))
            })?;
        if weights.iter().any(|value| !value.is_finite()) {
            return Err(PyValueError::new_err("weights must be finite"));
        }
        Ok(self
            .graph
            .leaves
            .iter()
            .map(|leaf| {
                evaluation::weighted_consensus(&leaf.state, 1 - self.observer, 0.0, &weights)
            })
            .collect())
    }

    fn backup(&self, leaf_values: Vec<f64>) -> PyResult<(u16, f64)> {
        if self.opponent_model != planner::OpponentModel::Min {
            return Err(PyValueError::new_err(
                "self_max and roll_only require opponent leaf values",
            ));
        }
        self.graph
            .backup(&leaf_values)
            .map_err(PyValueError::new_err)
    }

    fn action_values(&self, leaf_values: Vec<f64>) -> PyResult<Vec<(u16, f64)>> {
        if self.opponent_model != planner::OpponentModel::Min {
            return Err(PyValueError::new_err(
                "self_max and roll_only require opponent leaf values",
            ));
        }
        self.graph
            .root_action_values(&leaf_values)
            .map_err(PyValueError::new_err)
    }

    fn action_values_with_opponent(
        &self,
        root_values: Vec<f64>,
        opponent_values: Vec<f64>,
    ) -> PyResult<Vec<(u16, f64)>> {
        if self.opponent_model == planner::OpponentModel::Min {
            return Err(PyValueError::new_err(
                "opponent leaf values are only used by self_max and roll_only",
            ));
        }
        self.graph
            .root_action_values_self_max(&root_values, &opponent_values)
            .map_err(PyValueError::new_err)
    }

    fn first_action_family_counts(&self) -> Vec<(u8, usize)> {
        self.graph.first_action_family_counts()
    }

    #[getter]
    fn leaf_count(&self) -> usize {
        self.graph.leaves.len()
    }

    #[getter]
    fn node_count(&self) -> usize {
        self.graph.nodes.len()
    }
}

#[pyclass(module = "catan_ranked_fast")]
#[derive(Clone)]
struct Game {
    inner: RankedState,
}

#[pymethods]
impl Game {
    #[new]
    fn new(seed: u64) -> Self {
        Self {
            inner: RankedState::new(seed),
        }
    }

    fn clone_game(&self) -> Self {
        self.clone()
    }

    fn legal_actions(&self) -> Vec<u16> {
        self.inner.legal_actions()
    }

    #[pyo3(signature = (depth=2))]
    fn champion_action(&mut self, depth: u8) -> PyResult<u16> {
        if depth == 0 {
            return Err(PyValueError::new_err("champion depth must be positive"));
        }
        champion::choose_action_at_depth(&mut self.inner, depth)
            .ok_or_else(|| PyValueError::new_err("champion search produced no legal action"))
    }

    #[pyo3(signature = (action, roll_total=None, stolen_resource=None))]
    fn apply(
        &mut self,
        action: usize,
        roll_total: Option<u8>,
        stolen_resource: Option<u8>,
    ) -> PyResult<()> {
        self.inner
            .apply(action, roll_total, stolen_resource)
            .map(|_| ())
            .map_err(PyValueError::new_err)
    }

    #[pyo3(signature = (action, roll_total=None, stolen_resource=None))]
    fn apply_logged(
        &mut self,
        action: usize,
        roll_total: Option<u8>,
        stolen_resource: Option<u8>,
    ) -> PyResult<(Option<u8>, Option<u8>, Option<u8>)> {
        let outcome = self
            .inner
            .apply(action, roll_total, stolen_resource)
            .map_err(PyValueError::new_err)?;
        Ok((
            outcome.roll_total,
            outcome.stolen_resource,
            outcome.bought_dev,
        ))
    }

    #[getter]
    fn winner(&self) -> i8 {
        self.inner.winner()
    }

    #[pyo3(signature = (observer=None))]
    fn observation(&self, observer: Option<usize>) -> PyResult<Vec<f32>> {
        let observer = observer.unwrap_or(self.inner.current_player as usize);
        if observer >= 2 {
            return Err(PyValueError::new_err("observer must be seat 0 or 1"));
        }
        Ok(observation::encode(&self.inner, observer))
    }

    #[pyo3(signature = (observer=None, opp_hidden_vp=0.0))]
    fn linear_value(&self, observer: Option<usize>, opp_hidden_vp: f64) -> PyResult<f64> {
        let observer = observer.unwrap_or(self.inner.current_player as usize);
        if observer >= 2 {
            return Err(PyValueError::new_err("observer must be seat 0 or 1"));
        }
        Ok(evaluation::tuned_consensus(
            &self.inner,
            observer,
            opp_hidden_vp,
        ))
    }

    #[getter]
    fn current_player(&self) -> usize {
        self.inner.current_player as usize
    }

    #[getter]
    fn current_prompt(&self) -> u8 {
        self.inner.prompt
    }

    #[getter]
    fn num_turns(&self) -> u16 {
        self.inner.num_turns
    }

    #[getter]
    fn actual_vp(&self) -> Vec<i8> {
        self.inner.actual_vp.to_vec()
    }

    fn state_json(&self, observer: usize) -> PyResult<String> {
        if observer >= 2 {
            return Err(PyValueError::new_err("observer must be seat 0 or 1"));
        }
        let winner = self.inner.winner();
        let players: Vec<Value> = (0..2)
            .map(|seat| {
                let is_observer = seat == observer;
                let reveal_actual = is_observer || winner >= 0;
                json!({
                    "seat": seat,
                    "color": self.inner.colors[seat],
                    "visible_vp": self.inner.visible_vp[seat],
                    "actual_vp": if reveal_actual { json!(self.inner.actual_vp[seat]) } else { Value::Null },
                    "resource_count": self.inner.resources[seat].iter().sum::<i16>(),
                    "resources": if is_observer { json!(self.inner.resources[seat]) } else { Value::Null },
                    "dev_count": self.inner.dev_hand[seat].iter().sum::<u8>(),
                    "dev_hand": if is_observer { json!(self.inner.dev_hand[seat]) } else { Value::Null },
                    "dev_played": self.inner.dev_played[seat],
                    "knights_played": self.inner.dev_played[seat][state::KNIGHT as usize],
                    "roads_left": self.inner.roads_available[seat],
                    "settlements_left": self.inner.settlements_available[seat],
                    "cities_left": self.inner.cities_available[seat],
                    "longest_road": self.inner.road_lengths[seat],
                    "has_longest_road": self.inner.has_road[seat],
                    "has_largest_army": self.inner.has_army[seat],
                })
            })
            .collect();
        Ok(json!({
            "schema": "catan-ranked-fast.state/1",
            "seed": self.inner.seed,
            "observer": observer,
            "resource_names": RESOURCE_NAMES,
            "dev_names": DEV_NAMES,
            "tile_resources": self.inner.tile_resources,
            "tile_numbers": self.inner.tile_numbers,
            "port_types": self.inner.port_types,
            "vertices": self.inner.vertices.to_vec(),
            "edges": self.inner.edges.to_vec(),
            "robber_tile": self.inner.robber_tile,
            "bank": self.inner.bank,
            "dev_deck_count": self.inner.dev_len,
            "players": players,
            "current_player": self.inner.current_player,
            "current_turn": self.inner.current_turn,
            "prompt": self.inner.prompt,
            "prompt_name": PROMPT_NAMES[self.inner.prompt as usize],
            "initial": self.inner.initial,
            "has_rolled": self.inner.has_rolled,
            "discard_counts": self.inner.discard_counts,
            "road_building": self.inner.road_building,
            "free_roads": self.inner.free_roads,
            "num_turns": self.inner.num_turns,
            "winner": if winner >= 0 { json!(winner) } else { Value::Null },
            "dice": {
                "recent": &self.inner.dice.recent[..self.inner.dice.recent_len as usize],
                "cards_left": self.inner.dice.cards_left,
            },
        })
        .to_string())
    }

    fn replay_state_json(&self) -> String {
        json!({
            "schema": "catan-ranked-fast.replay-state/1",
            "seed": self.inner.seed,
            "colors": self.inner.colors,
            "tile_resources": self.inner.tile_resources,
            "tile_numbers": self.inner.tile_numbers,
            "port_types": self.inner.port_types,
            "dev_deck": self.inner.dev_deck,
            "dev_len": self.inner.dev_len,
            "robber_tile": self.inner.robber_tile,
            "dice": {
                "counts": self.inner.dice.counts,
                "cards_left": self.inner.dice.cards_left,
                "recent": self.inner.dice.recent,
                "recent_len": self.inner.dice.recent_len,
                "total_sevens": self.inner.dice.total_sevens,
                "seen_player": self.inner.dice.seen_player,
                "streak_color": self.inner.dice.streak_color,
                "streak_count": self.inner.dice.streak_count,
            },
            "vertices": self.inner.vertices.to_vec(),
            "edges": self.inner.edges.to_vec(),
            "resources": self.inner.resources,
            "bank": self.inner.bank,
            "dev_hand": self.inner.dev_hand,
            "dev_played": self.inner.dev_played,
            "dev_owned_at_start": self.inner.dev_owned_at_start,
            "visible_vp": self.inner.visible_vp,
            "actual_vp": self.inner.actual_vp,
            "roads_available": self.inner.roads_available,
            "settlements_available": self.inner.settlements_available,
            "cities_available": self.inner.cities_available,
            "has_road": self.inner.has_road,
            "has_army": self.inner.has_army,
            "road_cache": self.inner.road_cache,
            "road_lengths": self.inner.road_lengths,
            "road_owner": self.inner.road_owner,
            "board_road_length": self.inner.board_road_length,
            "army_owner": self.inner.army_owner,
            "army_size": self.inner.army_size,
            "current_player": self.inner.current_player,
            "current_turn": self.inner.current_turn,
            "prompt": self.inner.prompt,
            "initial": self.inner.initial,
            "has_rolled": self.inner.has_rolled,
            "played_dev_this_turn": self.inner.played_dev_this_turn,
            "discard_counts": self.inner.discard_counts,
            "road_building": self.inner.road_building,
            "free_roads": self.inner.free_roads,
            "num_turns": self.inner.num_turns,
            "last_settlement": self.inner.last_settlement,
        })
        .to_string()
    }

    #[pyo3(signature = (max_plans, max_actions=12, chance_samples=4))]
    fn turn_afterstates(
        &self,
        max_plans: usize,
        max_actions: usize,
        chance_samples: usize,
    ) -> PyResult<(Vec<Vec<u16>>, Vec<Vec<f32>>)> {
        if max_plans == 0 || max_actions == 0 {
            return Err(PyValueError::new_err(
                "max_plans and max_actions must be positive",
            ));
        }
        let observer = self.inner.current_player as usize;
        let graph = planner::TurnGraph::build(&self.inner, max_plans, max_actions, chance_samples);
        let plans = graph
            .leaves
            .iter()
            .map(|leaf| leaf.actions.clone())
            .collect();
        let observations = graph
            .leaves
            .iter()
            .map(|leaf| observation::encode(&leaf.state, observer))
            .collect();
        Ok((plans, observations))
    }

    #[pyo3(signature = (
        max_plans,
        max_actions=12,
        chance_samples=4,
        opponent_lookahead=false,
        opponent_plans=12,
        opponent_max_actions=10,
        opponent_model="min"
    ))]
    fn turn_search(
        &self,
        max_plans: usize,
        max_actions: usize,
        chance_samples: usize,
        opponent_lookahead: bool,
        opponent_plans: usize,
        opponent_max_actions: usize,
        opponent_model: &str,
    ) -> PyResult<TurnSearch> {
        if max_plans == 0 || max_actions == 0 || chance_samples == 0 {
            return Err(PyValueError::new_err(
                "max_plans, max_actions, and chance_samples must be positive",
            ));
        }
        if opponent_lookahead && (opponent_plans == 0 || opponent_max_actions == 0) {
            return Err(PyValueError::new_err(
                "opponent_plans and opponent_max_actions must be positive",
            ));
        }
        let opponent_model = if opponent_lookahead {
            match opponent_model {
                "min" => planner::OpponentModel::Min,
                "self_max" => planner::OpponentModel::SelfMax,
                "roll_only" => planner::OpponentModel::RollOnly,
                _ => {
                    return Err(PyValueError::new_err(
                        "opponent_model must be min, self_max, or roll_only",
                    ));
                }
            }
        } else {
            planner::OpponentModel::Min
        };
        let graph = if opponent_lookahead {
            planner::TurnGraph::build_with_opponent(
                &self.inner,
                max_plans,
                max_actions,
                chance_samples,
                opponent_plans,
                opponent_max_actions,
                opponent_model,
            )
            .map_err(PyValueError::new_err)?
        } else {
            planner::TurnGraph::build(&self.inner, max_plans, max_actions, chance_samples)
        };
        Ok(TurnSearch {
            graph,
            observer: self.inner.current_player as usize,
            opponent_model,
        })
    }

    fn set_bank_for_test(&mut self, resource: usize, count: i16) -> PyResult<()> {
        if resource >= 5 || count < 0 {
            return Err(PyValueError::new_err("invalid bank override"));
        }
        self.inner.bank[resource] = count;
        Ok(())
    }

    fn set_dev_hand_for_test(&mut self, player: usize, counts: Vec<u8>) -> PyResult<()> {
        if player >= 2 || counts.len() != 5 {
            return Err(PyValueError::new_err(
                "player must be a seat and counts must have length five",
            ));
        }
        self.inner.dev_hand[player].copy_from_slice(&counts);
        self.inner.dev_owned_at_start[player] =
            [counts[0] > 0, counts[2] > 0, counts[1] > 0, counts[3] > 0];
        self.inner.actual_vp[player] = self.inner.visible_vp[player] + counts[4] as i8;
        Ok(())
    }

    fn initialization_digest(&self) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<i8>, Vec<u8>, u8) {
        (
            self.inner.colors.to_vec(),
            self.inner.tile_resources.to_vec(),
            self.inner.tile_numbers.to_vec(),
            self.inner.port_types.to_vec(),
            self.inner.dev_deck.to_vec(),
            self.inner.robber_tile,
        )
    }

    #[allow(clippy::type_complexity)]
    fn canonical_digest(
        &self,
    ) -> (
        (Vec<u8>, Vec<i16>, Vec<i16>, Vec<i8>, Vec<i8>),
        (Vec<u8>, Vec<u8>, Vec<i8>, Vec<i8>),
        (u8, u8, u8, bool, u16),
        (Vec<u8>, i8, u8, i8, u8),
        (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>),
    ) {
        let resources = self.inner.resources.iter().flatten().copied().collect();
        let dev_hand = self.inner.dev_hand.iter().flatten().copied().collect();
        let dev_played = self.inner.dev_played.iter().flatten().copied().collect();
        (
            (
                self.inner.colors.to_vec(),
                resources,
                self.inner.bank.to_vec(),
                self.inner.vertices.to_vec(),
                self.inner.edges.to_vec(),
            ),
            (
                dev_hand,
                dev_played,
                self.inner.visible_vp.to_vec(),
                self.inner.actual_vp.to_vec(),
            ),
            (
                self.inner.robber_tile,
                self.inner.current_player,
                self.inner.current_turn,
                self.inner.initial,
                self.inner.num_turns,
            ),
            (
                self.inner.road_lengths.to_vec(),
                self.inner.road_owner,
                self.inner.board_road_length,
                self.inner.army_owner,
                self.inner.army_size,
            ),
            (
                self.inner.roads_available.to_vec(),
                self.inner.settlements_available.to_vec(),
                self.inner.cities_available.to_vec(),
                self.inner.discard_counts.to_vec(),
            ),
        )
    }
}

#[pyclass(module = "catan_ranked_fast")]
#[derive(Clone)]
struct VecEnv {
    games: Vec<RankedState>,
}

#[pymethods]
impl VecEnv {
    #[new]
    #[pyo3(signature = (num_envs, seed=0))]
    fn new(num_envs: usize, seed: u64) -> PyResult<Self> {
        if num_envs == 0 {
            return Err(PyValueError::new_err("num_envs must be positive"));
        }
        Ok(Self {
            games: (0..num_envs)
                .map(|i| RankedState::new(seed + i as u64))
                .collect(),
        })
    }

    fn clone_env(&self) -> Self {
        self.clone()
    }

    fn reset(&mut self, seed: u64) {
        for (i, game) in self.games.iter_mut().enumerate() {
            *game = RankedState::new(seed + i as u64);
        }
    }

    fn legal_actions(&self) -> Vec<Vec<u16>> {
        self.games.iter().map(RankedState::legal_actions).collect()
    }

    #[pyo3(signature = (actions, roll_totals=None, stolen_resources=None))]
    fn step(
        &mut self,
        actions: Vec<usize>,
        roll_totals: Option<Vec<i16>>,
        stolen_resources: Option<Vec<i16>>,
    ) -> PyResult<(Vec<Vec<u16>>, Vec<bool>, Vec<i8>, Vec<Vec<f32>>)> {
        if actions.len() != self.games.len() {
            return Err(PyValueError::new_err("wrong action batch length"));
        }
        if roll_totals
            .as_ref()
            .is_some_and(|v| v.len() != actions.len())
            || stolen_resources
                .as_ref()
                .is_some_and(|v| v.len() != actions.len())
        {
            return Err(PyValueError::new_err("wrong chance batch length"));
        }
        for (i, (&action, game)) in actions.iter().zip(self.games.iter_mut()).enumerate() {
            let roll = roll_totals
                .as_ref()
                .and_then(|v| (v[i] >= 0).then_some(v[i] as u8));
            let steal = stolen_resources
                .as_ref()
                .and_then(|v| (v[i] >= 0).then_some(v[i] as u8));
            game.apply(action, roll, steal)
                .map_err(PyValueError::new_err)?;
        }
        let legal = self.legal_actions();
        let winners: Vec<i8> = self.games.iter().map(RankedState::winner).collect();
        let done = winners.iter().map(|&winner| winner >= 0).collect();
        let observations = self
            .games
            .iter()
            .map(|state| observation::encode(state, state.current_player as usize))
            .collect();
        Ok((legal, done, winners, observations))
    }

    #[getter]
    fn num_envs(&self) -> usize {
        self.games.len()
    }
}

#[pyfunction]
#[pyo3(signature = (games, seed=0, max_plies=8000))]
fn play_random_games(games: usize, seed: u64, max_plies: usize) -> (usize, usize, usize) {
    let mut completed = 0;
    let mut capped = 0;
    let mut plies = 0;
    for game_index in 0..games {
        let mut game = RankedState::new(seed + game_index as u64);
        let mut game_plies = 0;
        let mut legal = Vec::with_capacity(32);
        while game.winner() < 0 && game_plies < max_plies {
            game.fill_legal_actions(&mut legal);
            let selected = game.rng.randbelow(legal.len());
            game.apply(legal[selected] as usize, None, None).unwrap();
            game_plies += 1;
        }
        plies += game_plies;
        if game.winner() >= 0 {
            completed += 1;
        } else {
            capped += 1;
        }
    }
    (completed, plies, capped)
}

#[pyfunction]
fn linear_values_from_observations(observations: Vec<Vec<f32>>) -> PyResult<Vec<f64>> {
    observations
        .iter()
        .map(|observation| {
            evaluation::tuned_consensus_observation(observation).map_err(PyValueError::new_err)
        })
        .collect()
}

#[pymodule]
fn catan_ranked_fast(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Game>()?;
    module.add_class::<TurnSearch>()?;
    module.add_class::<VecEnv>()?;
    module.add_function(wrap_pyfunction!(play_random_games, module)?)?;
    module.add_function(wrap_pyfunction!(linear_values_from_observations, module)?)?;
    module.add_function(wrap_pyfunction!(decode_action_json, module)?)?;
    module.add_function(wrap_pyfunction!(topology_json, module)?)?;
    module.add("ACTION_DIM", topology::ACTION_DIM)?;
    module.add("OBSERVATION_DIM", observation::OBSERVATION_DIM)?;
    module.add("OBSERVATION_VERSION", observation::OBSERVATION_VERSION)?;
    Ok(())
}
