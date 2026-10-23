use std::collections::{HashMap, HashSet};

use crate::state::{PROMPT_PLAY, RankedState};
use crate::topology::{ACTION_B, ACTION_KIND};

#[derive(Clone)]
pub struct TurnLeaf {
    pub actions: Vec<u16>,
    pub state: RankedState,
}

#[derive(Clone)]
pub enum GraphNode {
    Decision(Vec<(u16, usize)>),
    MinDecision(Vec<(u16, usize)>),
    Chance(Vec<(f64, usize)>),
    Leaf(usize),
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum OpponentModel {
    Min,
    SelfMax,
    RollOnly,
}

#[derive(Clone)]
pub struct TurnGraph {
    pub nodes: Vec<GraphNode>,
    pub root: usize,
    pub leaves: Vec<TurnLeaf>,
}

struct GraphBuilder {
    graph: TurnGraph,
    memo: HashMap<Vec<i16>, usize>,
    observer: usize,
    origin_turn: u8,
    max_actions: usize,
    max_nodes: usize,
    chance_samples: usize,
    maximizing: bool,
    stop_at_play: bool,
}

fn mix(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn digest_seed(digest: &[i16], action: u16) -> u64 {
    digest.iter().fold(action as u64, |hash, &value| {
        mix(hash ^ (value as u16 as u64))
    })
}

fn quantile(seed: u64, sample: usize, samples: usize) -> f64 {
    let rotation = mix(seed) as f64 / u64::MAX as f64;
    (rotation + (sample as f64 + 0.5) / samples as f64).fract()
}

fn weighted_choice(weights: &[u32], target: f64) -> Option<u8> {
    let total: u32 = weights.iter().sum();
    if total == 0 {
        return None;
    }
    let mut cursor = target * total as f64;
    for (index, &weight) in weights.iter().enumerate() {
        cursor -= weight as f64;
        if cursor < 0.0 && weight > 0 {
            return Some(index as u8);
        }
    }
    weights
        .iter()
        .rposition(|&weight| weight > 0)
        .map(|i| i as u8)
}

fn public_dev_counts(state: &RankedState, observer: usize) -> [u32; 5] {
    let mut counts = [14, 2, 2, 2, 5];
    for card in 0..5 {
        counts[card] -= state.dev_hand[observer][card] as u32;
        counts[card] -= state.dev_played[0][card] as u32;
        counts[card] -= state.dev_played[1][card] as u32;
    }
    counts
}

fn public_dev_world(state: &RankedState, observer: usize) -> RankedState {
    let opponent = 1 - observer;
    let hidden = state.dev_hand[opponent]
        .iter()
        .map(|&count| count as usize)
        .sum();
    let mut counts = public_dev_counts(state, observer);
    let seed = digest_seed(&state.observer_digest(observer), u16::MAX);
    let mut world = state.clone();
    world.dev_hand[opponent] = [0; 5];
    for draw in 0..hidden {
        let Some(card) = weighted_choice(&counts, quantile(seed, draw, hidden)) else {
            break;
        };
        world.dev_hand[opponent][card as usize] += 1;
        counts[card as usize] -= 1;
    }
    world.dev_owned_at_start[opponent] = [
        world.dev_hand[opponent][0] > 0,
        world.dev_hand[opponent][2] > 0,
        world.dev_hand[opponent][1] > 0,
        world.dev_hand[opponent][3] > 0,
    ];
    world.actual_vp[opponent] = world.visible_vp[opponent] + world.dev_hand[opponent][4] as i8;
    world
}

fn chance_children(
    state: &RankedState,
    action: u16,
    observer: usize,
    samples: usize,
) -> Vec<RankedState> {
    if samples == 0 {
        return Vec::new();
    }
    let digest = state.observer_digest(observer);
    let seed = digest_seed(&digest, action);
    let kind = ACTION_KIND[action as usize];
    let mut children = Vec::with_capacity(samples);
    for sample in 0..samples {
        let q = quantile(seed, sample, samples);
        let mut child = state.clone();
        let result = match kind {
            3 => {
                let counts = public_dev_counts(state, observer);
                let Some(card) = weighted_choice(&counts, q) else {
                    continue;
                };
                child.dev_deck[(child.dev_len - 1) as usize] = card;
                child.apply(action as usize, None, None)
            }
            7 if ACTION_B[action as usize] != 2 => {
                let victim = child.seat_for_color(ACTION_B[action as usize]);
                let weights = child.resources[victim].map(|count| count.max(0) as u32);
                let Some(resource) = weighted_choice(&weights, q) else {
                    continue;
                };
                child.apply(action as usize, None, Some(resource))
            }
            12 => {
                let color = child.colors[child.current_player as usize] as usize;
                let distribution = child.dice.distribution(color);
                let weights: Vec<u32> = distribution
                    .iter()
                    .map(|probability| (probability * 1_000_000.0).round() as u32)
                    .collect();
                let Some(total) = weighted_choice(&weights, q) else {
                    continue;
                };
                child.apply(action as usize, Some(total), None)
            }
            _ => child.apply(action as usize, None, None),
        };
        if result.is_ok() {
            children.push(child);
        }
    }
    children
}

fn is_chance(action: u16) -> bool {
    let kind = ACTION_KIND[action as usize];
    kind == 3 || kind == 12 || (kind == 7 && ACTION_B[action as usize] != 2)
}

fn action_priority(action: u16) -> (u8, u16) {
    let kind = ACTION_KIND[action as usize];
    let priority = match kind {
        5 => 0,      // Always preserve the do-nothing complete turn.
        0 => 1,      // City.
        2 => 2,      // Settlement.
        1 => 3,      // Road.
        8..=11 => 4, // Development-card plays.
        3 => 5,      // Development-card purchase.
        6 => 6,      // Maritime trade.
        12 => 7,     // Roll.
        4 | 7 => 8,  // Forced discard / robber placement.
        _ => 9,
    };
    (priority, action)
}

fn ordered_actions(state: &RankedState) -> Vec<u16> {
    let mut actions = state.legal_actions();
    actions.sort_unstable_by_key(|&action| action_priority(action));
    actions
}

impl GraphBuilder {
    fn leaf(&mut self, state: RankedState, actions: Vec<u16>, leaf_limit: usize) -> Option<usize> {
        let digest = state.observer_digest(self.observer);
        if let Some(&node) = self.memo.get(&digest) {
            return Some(node);
        }
        if self.graph.leaves.len() >= leaf_limit || self.graph.nodes.len() >= self.max_nodes {
            return None;
        }
        let leaf_index = self.graph.leaves.len();
        let node = self.graph.nodes.len();
        self.graph.leaves.push(TurnLeaf { actions, state });
        self.graph.nodes.push(GraphNode::Leaf(leaf_index));
        self.memo.insert(digest, node);
        Some(node)
    }

    fn state_node(
        &mut self,
        state: RankedState,
        actions: Vec<u16>,
        leaf_limit: usize,
    ) -> Option<usize> {
        let digest = state.observer_digest(self.observer);
        if let Some(&node) = self.memo.get(&digest) {
            return Some(node);
        }
        if state.winner() >= 0
            || state.current_turn != self.origin_turn
            || (self.stop_at_play
                && state.prompt == PROMPT_PLAY
                && state.current_player == self.origin_turn)
        {
            return self.leaf(state, actions, leaf_limit);
        }
        if self.graph.nodes.len() >= self.max_nodes {
            return None;
        }
        if actions.len() >= self.max_actions
            && state.prompt == PROMPT_PLAY
            && state.current_player == self.origin_turn
        {
            // Forced discard/robber chains may exceed the strategic depth. Close a completed
            // forced chain at the consistent afterstate boundary instead of dropping the roll.
            let mut child = state.clone();
            if child.is_legal(186) && child.apply(186, None, None).is_ok() {
                let mut plan = actions;
                plan.push(186);
                return self.leaf(child, plan, leaf_limit);
            }
            return None;
        }
        let node = self.graph.nodes.len();
        self.graph.nodes.push(if self.maximizing {
            GraphNode::Decision(Vec::new())
        } else {
            GraphNode::MinDecision(Vec::new())
        });
        self.memo.insert(digest.clone(), node);
        let mut legal = ordered_actions(&state);
        if state.current_player != self.origin_turn {
            legal.truncate(1);
        }
        let mut edges = Vec::new();
        for action in legal {
            if self.graph.leaves.len() >= leaf_limit {
                break;
            }
            if let Some(child) = self.action_node(&state, &actions, action, leaf_limit) {
                edges.push((action, child));
            }
        }
        if edges.is_empty() {
            self.memo.remove(&digest);
            return None;
        }
        self.graph.nodes[node] = if self.maximizing {
            GraphNode::Decision(edges)
        } else {
            GraphNode::MinDecision(edges)
        };
        Some(node)
    }

    fn action_node(
        &mut self,
        state: &RankedState,
        actions: &[u16],
        action: u16,
        leaf_limit: usize,
    ) -> Option<usize> {
        let mut plan = actions.to_vec();
        plan.push(action);
        if !is_chance(action) {
            let mut child = state.clone();
            child.apply(action as usize, None, None).ok()?;
            return self.state_node(child, plan, leaf_limit);
        }

        // A chance action is atomic: either every deterministic sample is represented or the
        // action is omitted. Roll back orphaned nodes if the budget cannot complete all samples.
        let graph_before = self.graph.clone();
        let memo_before = self.memo.clone();
        let outcomes = chance_children(state, action, self.observer, self.chance_samples);
        if outcomes.len() != self.chance_samples || outcomes.is_empty() {
            return None;
        }
        let mut edges = Vec::with_capacity(outcomes.len());
        for (sample, child) in outcomes.into_iter().enumerate() {
            let remaining_samples = self.chance_samples - sample;
            let remaining_leaves = leaf_limit.saturating_sub(self.graph.leaves.len());
            if remaining_leaves < remaining_samples {
                self.graph = graph_before;
                self.memo = memo_before;
                return None;
            }
            let sample_share = remaining_leaves.div_ceil(remaining_samples);
            let sample_limit = self.graph.leaves.len() + sample_share;
            let Some(child_node) = self.state_node(child, plan.clone(), sample_limit) else {
                self.graph = graph_before;
                self.memo = memo_before;
                return None;
            };
            edges.push((1.0 / self.chance_samples as f64, child_node));
        }
        if self.graph.nodes.len() >= self.max_nodes {
            self.graph = graph_before;
            self.memo = memo_before;
            return None;
        }
        let node = self.graph.nodes.len();
        self.graph.nodes.push(GraphNode::Chance(edges));
        Some(node)
    }
}

impl TurnGraph {
    fn build_phase(
        root: &RankedState,
        observer: usize,
        max_plans: usize,
        max_actions: usize,
        chance_samples: usize,
        maximizing: bool,
    ) -> Self {
        let mut builder = GraphBuilder {
            graph: Self {
                nodes: vec![if maximizing {
                    GraphNode::Decision(Vec::new())
                } else {
                    GraphNode::MinDecision(Vec::new())
                }],
                root: 0,
                leaves: Vec::with_capacity(max_plans),
            },
            memo: HashMap::new(),
            observer,
            origin_turn: root.current_turn,
            max_actions,
            max_nodes: max_plans.saturating_mul(256).max(256),
            chance_samples,
            maximizing,
            stop_at_play: false,
        };
        if max_plans == 0 || max_actions == 0 || root.winner() >= 0 {
            return builder.graph;
        }

        // Generation-zero policy for setup and exceptional prompts: expose legal one-action
        // successors and let the caller score them. Ordinary own turns use the full DAG below.
        if root.initial || root.prompt != PROMPT_PLAY || root.current_player != root.current_turn {
            let mut edges = Vec::new();
            for action in ordered_actions(root).into_iter().take(max_plans) {
                let mut child = root.clone();
                if child.apply(action as usize, None, None).is_err() {
                    continue;
                }
                if let Some(node) = builder.leaf(child, vec![action], max_plans) {
                    edges.push((action, node));
                }
            }
            builder.graph.nodes[0] = if maximizing {
                GraphNode::Decision(edges)
            } else {
                GraphNode::MinDecision(edges)
            };
            return builder.graph;
        }

        let ordered = ordered_actions(root);
        let mut families: Vec<(u8, Vec<u16>)> = Vec::new();
        for action in ordered {
            let kind = ACTION_KIND[action as usize];
            if let Some((_, actions)) = families
                .iter_mut()
                .find(|(candidate, _)| *candidate == kind)
            {
                actions.push(action);
            } else {
                families.push((kind, vec![action]));
            }
        }
        let family_count = families.len();
        if family_count == 0 {
            return builder.graph;
        }
        let mut root_edges = Vec::new();
        for (family_index, (_kind, actions)) in families.into_iter().enumerate() {
            let family_quota =
                max_plans / family_count + usize::from(family_index < max_plans % family_count);
            if family_quota == 0 {
                continue;
            }
            let action_count = actions.len();
            for (action_index, action) in actions.into_iter().enumerate() {
                let action_quota = family_quota / action_count
                    + usize::from(action_index < family_quota % action_count);
                if action_quota == 0 {
                    continue;
                }
                let leaf_limit = (builder.graph.leaves.len() + action_quota).min(max_plans);
                if let Some(child) = builder.action_node(root, &[], action, leaf_limit) {
                    root_edges.push((action, child));
                }
            }
        }
        builder.graph.nodes[0] = if maximizing {
            GraphNode::Decision(root_edges)
        } else {
            GraphNode::MinDecision(root_edges)
        };
        builder.graph
    }

    pub fn build(
        root: &RankedState,
        max_plans: usize,
        max_actions: usize,
        chance_samples: usize,
    ) -> Self {
        Self::build_phase(
            root,
            root.current_player as usize,
            max_plans,
            max_actions,
            chance_samples,
            true,
        )
    }

    fn build_opponent_turn(
        root: &RankedState,
        observer: usize,
        max_plans: usize,
        max_actions: usize,
        chance_samples: usize,
        roll_only: bool,
    ) -> Result<Self, String> {
        let world = public_dev_world(root, observer);
        let roll = ordered_actions(&world)
            .into_iter()
            .find(|&action| ACTION_KIND[action as usize] == 12)
            .ok_or_else(|| "opponent turn has no legal roll".to_string())?;
        let outcomes = chance_children(&world, roll, observer, chance_samples);
        if outcomes.len() != chance_samples || outcomes.is_empty() {
            return Err("opponent roll did not produce every chance sample".into());
        }
        let mut builder = GraphBuilder {
            graph: Self {
                nodes: vec![GraphNode::Chance(Vec::new())],
                root: 0,
                leaves: Vec::with_capacity(max_plans),
            },
            memo: HashMap::new(),
            observer,
            origin_turn: world.current_turn,
            max_actions,
            max_nodes: max_plans.saturating_mul(256).max(256),
            chance_samples,
            maximizing: false,
            stop_at_play: roll_only,
        };
        let mut edges = Vec::with_capacity(chance_samples);
        for (sample, child) in outcomes.into_iter().enumerate() {
            let remaining_samples = chance_samples - sample;
            let remaining_leaves = max_plans.saturating_sub(builder.graph.leaves.len());
            if remaining_leaves < remaining_samples {
                return Err("opponent plan budget cannot cover every roll sample".into());
            }
            let sample_share = remaining_leaves.div_ceil(remaining_samples);
            let sample_limit = builder.graph.leaves.len() + sample_share;
            let child_node = builder
                .state_node(child, vec![roll], sample_limit)
                .ok_or_else(|| "opponent roll branch produced no complete reply".to_string())?;
            edges.push((1.0 / chance_samples as f64, child_node));
        }
        builder.graph.nodes[0] = GraphNode::Chance(edges);
        Ok(builder.graph)
    }

    fn replace_leaf_with_graph(
        &mut self,
        leaf_index: usize,
        replacement: &Self,
        prefix: &[u16],
    ) -> Result<(), String> {
        let target_root = self
            .nodes
            .iter()
            .position(|node| matches!(node, GraphNode::Leaf(index) if *index == leaf_index))
            .ok_or_else(|| "turn-search leaf node is missing".to_string())?;
        let leaf_offset = self.leaves.len();
        self.leaves.extend(replacement.leaves.iter().map(|leaf| {
            let mut actions = prefix.to_vec();
            actions.extend_from_slice(&leaf.actions);
            TurnLeaf {
                actions,
                state: leaf.state.clone(),
            }
        }));
        let mut node_map = vec![target_root; replacement.nodes.len()];
        for source in 0..replacement.nodes.len() {
            if source != replacement.root {
                node_map[source] = self.nodes.len();
                self.nodes.push(GraphNode::Leaf(0));
            }
        }
        for (source, node) in replacement.nodes.iter().enumerate() {
            let mapped = match node {
                GraphNode::Decision(edges) => GraphNode::Decision(
                    edges
                        .iter()
                        .map(|&(action, child)| (action, node_map[child]))
                        .collect(),
                ),
                GraphNode::MinDecision(edges) => GraphNode::MinDecision(
                    edges
                        .iter()
                        .map(|&(action, child)| (action, node_map[child]))
                        .collect(),
                ),
                GraphNode::Chance(edges) => GraphNode::Chance(
                    edges
                        .iter()
                        .map(|&(probability, child)| (probability, node_map[child]))
                        .collect(),
                ),
                GraphNode::Leaf(index) => GraphNode::Leaf(leaf_offset + index),
            };
            self.nodes[node_map[source]] = mapped;
        }
        Ok(())
    }

    fn compact_leaves(&mut self) {
        let old_leaves = std::mem::take(&mut self.leaves);
        let mut mapping = HashMap::new();
        for node in &mut self.nodes {
            let GraphNode::Leaf(index) = node else {
                continue;
            };
            let next = *mapping.entry(*index).or_insert_with(|| {
                let next = self.leaves.len();
                self.leaves.push(old_leaves[*index].clone());
                next
            });
            *index = next;
        }
    }

    pub fn build_with_opponent(
        root: &RankedState,
        max_plans: usize,
        max_actions: usize,
        chance_samples: usize,
        opponent_plans: usize,
        opponent_max_actions: usize,
        opponent_model: OpponentModel,
    ) -> Result<Self, String> {
        if opponent_plans < chance_samples {
            return Err("opponent_plans must be at least chance_samples".into());
        }
        let observer = root.current_player as usize;
        let mut graph = Self::build(root, max_plans, max_actions, chance_samples);
        if root.initial || root.prompt != PROMPT_PLAY || root.current_player != root.current_turn {
            return Ok(graph);
        }
        let own_leaf_count = graph.leaves.len();
        for leaf_index in 0..own_leaf_count {
            let leaf = graph.leaves[leaf_index].clone();
            if leaf.state.winner() >= 0 {
                continue;
            }
            let reply = Self::build_opponent_turn(
                &leaf.state,
                observer,
                opponent_plans,
                opponent_max_actions,
                chance_samples,
                opponent_model == OpponentModel::RollOnly,
            )?;
            graph.replace_leaf_with_graph(leaf_index, &reply, &leaf.actions)?;
        }
        graph.compact_leaves();
        Ok(graph)
    }

    fn backed_value(&self, node: usize, leaf_values: &[f64], memo: &mut [Option<f64>]) -> f64 {
        if let Some(value) = memo[node] {
            return value;
        }
        let value = match &self.nodes[node] {
            GraphNode::Leaf(index) => leaf_values[*index],
            GraphNode::Chance(edges) => edges
                .iter()
                .map(|(probability, child)| {
                    probability * self.backed_value(*child, leaf_values, memo)
                })
                .sum(),
            GraphNode::Decision(edges) => edges
                .iter()
                .map(|(_, child)| self.backed_value(*child, leaf_values, memo))
                .fold(f64::NEG_INFINITY, f64::max),
            GraphNode::MinDecision(edges) => edges
                .iter()
                .map(|(_, child)| self.backed_value(*child, leaf_values, memo))
                .fold(f64::INFINITY, f64::min),
        };
        memo[node] = Some(value);
        value
    }

    fn backed_values_self_max(
        &self,
        node: usize,
        root_values: &[f64],
        opponent_values: &[f64],
        memo: &mut [Option<(f64, f64)>],
    ) -> (f64, f64) {
        if let Some(values) = memo[node] {
            return values;
        }
        let values = match &self.nodes[node] {
            GraphNode::Leaf(index) => (root_values[*index], opponent_values[*index]),
            GraphNode::Chance(edges) => {
                edges.iter().fold((0.0, 0.0), |sum, (probability, child)| {
                    let child =
                        self.backed_values_self_max(*child, root_values, opponent_values, memo);
                    (sum.0 + probability * child.0, sum.1 + probability * child.1)
                })
            }
            GraphNode::Decision(edges) => edges
                .iter()
                .map(|(action, child)| {
                    (
                        *action,
                        self.backed_values_self_max(*child, root_values, opponent_values, memo),
                    )
                })
                .max_by(|left, right| {
                    left.1
                        .0
                        .total_cmp(&right.1.0)
                        .then_with(|| right.0.cmp(&left.0))
                })
                .map(|(_, values)| values)
                .unwrap_or((f64::NEG_INFINITY, f64::NEG_INFINITY)),
            GraphNode::MinDecision(edges) => edges
                .iter()
                .map(|(action, child)| {
                    (
                        *action,
                        self.backed_values_self_max(*child, root_values, opponent_values, memo),
                    )
                })
                .max_by(|left, right| {
                    left.1
                        .1
                        .total_cmp(&right.1.1)
                        .then_with(|| right.0.cmp(&left.0))
                })
                .map(|(_, values)| values)
                .unwrap_or((f64::NEG_INFINITY, f64::NEG_INFINITY)),
        };
        memo[node] = Some(values);
        values
    }

    pub fn backup(&self, leaf_values: &[f64]) -> Result<(u16, f64), String> {
        self.root_action_values(leaf_values)?
            .into_iter()
            .max_by(|left, right| {
                left.1
                    .total_cmp(&right.1)
                    .then_with(|| right.0.cmp(&left.0))
            })
            .ok_or_else(|| "turn search produced no backed action".into())
    }

    pub fn root_action_values(&self, leaf_values: &[f64]) -> Result<Vec<(u16, f64)>, String> {
        if leaf_values.len() != self.leaves.len() {
            return Err(format!(
                "expected {} leaf values, got {}",
                self.leaves.len(),
                leaf_values.len()
            ));
        }
        let GraphNode::Decision(edges) = &self.nodes[self.root] else {
            return Err("turn-search root is not a decision node".into());
        };
        if edges.is_empty() {
            return Err("turn search produced no complete plan".into());
        }
        let mut memo = vec![None; self.nodes.len()];
        Ok(edges
            .iter()
            .map(|(action, child)| (*action, self.backed_value(*child, leaf_values, &mut memo)))
            .collect())
    }

    pub fn root_action_values_self_max(
        &self,
        root_values: &[f64],
        opponent_values: &[f64],
    ) -> Result<Vec<(u16, f64)>, String> {
        if root_values.len() != self.leaves.len() || opponent_values.len() != self.leaves.len() {
            return Err(format!(
                "expected {} root and opponent leaf values, got {} and {}",
                self.leaves.len(),
                root_values.len(),
                opponent_values.len()
            ));
        }
        let GraphNode::Decision(edges) = &self.nodes[self.root] else {
            return Err("turn-search root is not a decision node".into());
        };
        if edges.is_empty() {
            return Err("turn search produced no complete plan".into());
        }
        let mut memo = vec![None; self.nodes.len()];
        Ok(edges
            .iter()
            .map(|(action, child)| {
                let values =
                    self.backed_values_self_max(*child, root_values, opponent_values, &mut memo);
                (*action, values.0)
            })
            .collect())
    }

    pub fn first_action_family_counts(&self) -> Vec<(u8, usize)> {
        fn visit(graph: &TurnGraph, node: usize, leaves: &mut HashSet<usize>) {
            match &graph.nodes[node] {
                GraphNode::Leaf(index) => {
                    leaves.insert(*index);
                }
                GraphNode::Chance(edges) => {
                    for (_, child) in edges {
                        visit(graph, *child, leaves);
                    }
                }
                GraphNode::Decision(edges) => {
                    for (_, child) in edges {
                        visit(graph, *child, leaves);
                    }
                }
                GraphNode::MinDecision(edges) => {
                    for (_, child) in edges {
                        visit(graph, *child, leaves);
                    }
                }
            }
        }
        let GraphNode::Decision(edges) = &self.nodes[self.root] else {
            return Vec::new();
        };
        let mut by_kind: HashMap<u8, HashSet<usize>> = HashMap::new();
        for (action, child) in edges {
            visit(
                self,
                *child,
                by_kind.entry(ACTION_KIND[*action as usize]).or_default(),
            );
        }
        let mut counts: Vec<_> = by_kind
            .into_iter()
            .map(|(kind, leaves)| (kind, leaves.len()))
            .collect();
        counts.sort_unstable();
        counts
    }
}

#[cfg(test)]
mod tests {
    use super::{GraphNode, TurnGraph, TurnLeaf};
    use crate::state::{PROMPT_PLAY, RankedState};
    use crate::topology::ACTION_KIND;

    #[test]
    fn chance_backup_uses_mean_not_luckiest_leaf() {
        let state = RankedState::new(0);
        let graph = TurnGraph {
            nodes: vec![
                GraphNode::Decision(vec![(180, 1), (186, 4)]),
                GraphNode::Chance(vec![(0.5, 2), (0.5, 3)]),
                GraphNode::Leaf(0),
                GraphNode::Leaf(1),
                GraphNode::Leaf(2),
            ],
            root: 0,
            leaves: vec![
                TurnLeaf {
                    actions: vec![180],
                    state: state.clone(),
                },
                TurnLeaf {
                    actions: vec![180],
                    state: state.clone(),
                },
                TurnLeaf {
                    actions: vec![186],
                    state,
                },
            ],
        };
        assert_eq!(graph.backup(&[0.0, 1.0, 0.6]).unwrap(), (186, 0.6));
    }

    #[test]
    fn opponent_decision_backs_up_the_worst_reply_for_the_observer() {
        let state = RankedState::new(0);
        let graph = TurnGraph {
            nodes: vec![
                GraphNode::Decision(vec![(180, 1), (186, 4)]),
                GraphNode::MinDecision(vec![(0, 2), (1, 3)]),
                GraphNode::Leaf(0),
                GraphNode::Leaf(1),
                GraphNode::Leaf(2),
            ],
            root: 0,
            leaves: vec![
                TurnLeaf {
                    actions: vec![180, 0],
                    state: state.clone(),
                },
                TurnLeaf {
                    actions: vec![180, 1],
                    state: state.clone(),
                },
                TurnLeaf {
                    actions: vec![186],
                    state,
                },
            ],
        };
        assert_eq!(graph.backup(&[0.8, 0.2, 0.3]).unwrap(), (186, 0.3));
    }

    #[test]
    fn opponent_self_max_chooses_its_value_and_returns_the_observer_value() {
        let state = RankedState::new(0);
        let graph = TurnGraph {
            nodes: vec![
                GraphNode::Decision(vec![(180, 1), (186, 4)]),
                GraphNode::MinDecision(vec![(0, 2), (1, 3)]),
                GraphNode::Leaf(0),
                GraphNode::Leaf(1),
                GraphNode::Leaf(2),
            ],
            root: 0,
            leaves: vec![
                TurnLeaf {
                    actions: vec![180, 0],
                    state: state.clone(),
                },
                TurnLeaf {
                    actions: vec![180, 1],
                    state: state.clone(),
                },
                TurnLeaf {
                    actions: vec![186],
                    state,
                },
            ],
        };
        let values = graph
            .root_action_values_self_max(&[0.8, 0.2, 0.3], &[0.1, 0.9, 0.0])
            .unwrap();
        assert_eq!(values, vec![(180, 0.2), (186, 0.3)]);
    }

    #[test]
    fn roll_only_stops_at_the_first_opponent_free_decision() {
        let mut state = RankedState::new(17);
        state.initial = false;
        state.prompt = PROMPT_PLAY;
        state.current_player = 1;
        state.current_turn = 1;
        let graph = TurnGraph::build_opponent_turn(&state, 0, 32, 10, 4, true).unwrap();
        assert!(!graph.leaves.is_empty());
        for leaf in graph.leaves {
            assert_eq!(leaf.state.prompt, PROMPT_PLAY);
            assert_eq!(leaf.state.current_turn, 1);
            assert_eq!(ACTION_KIND[leaf.actions[0] as usize], 12);
            assert!(!leaf.actions.contains(&186));
        }
    }
}
