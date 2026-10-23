//! Frozen `tuned_consensus` action-level depth-two alpha-beta search.

use crate::evaluation;
use crate::rng::PythonRandom;
use crate::state::{RankedState, VICTORY_POINT};
use crate::topology::{ACTION_B, ACTION_KIND};

#[allow(dead_code)]
const DEPTH: u8 = 2;
const DEV_INITIAL: [u8; 5] = [14, 2, 2, 2, 5];

fn opponent_hidden_vp(state: &RankedState, observer: usize) -> f64 {
    let opponent = 1 - observer;
    let opponent_hidden: u32 = state.dev_hand[opponent].iter().map(|&x| x as u32).sum();
    if opponent_hidden == 0 {
        return 0.0;
    }
    let mut pool = [0u32; 5];
    for card in 0..5 {
        pool[card] = DEV_INITIAL[card] as u32
            - state.dev_hand[observer][card] as u32
            - state.dev_played[observer][card] as u32
            - state.dev_played[opponent][card] as u32;
    }
    let total: u32 = pool.iter().sum();
    if total == 0 {
        0.0
    } else {
        opponent_hidden as f64 * pool[VICTORY_POINT as usize] as f64 / total as f64
    }
}

fn number_probability(total: u8) -> f64 {
    (6 - (7i16 - total as i16).abs()) as f64 / 36.0
}

fn outcomes(
    state: &RankedState,
    action: u16,
    observer: usize,
    rng: &mut PythonRandom,
) -> Vec<(RankedState, f64)> {
    match ACTION_KIND[action as usize] {
        3 => {
            // Upstream exposes the live deck plus the opponent's hidden hand: equivalently,
            // every initial card not in our hand and not publicly played.
            let mut counts = DEV_INITIAL.map(u32::from);
            for card in 0..5 {
                counts[card] -= state.dev_hand[observer][card] as u32;
                counts[card] -= state.dev_played[0][card] as u32;
                counts[card] -= state.dev_played[1][card] as u32;
            }
            let total: u32 = counts.iter().sum();
            (0..5)
                .filter_map(|card| {
                    if counts[card] == 0 {
                        return None;
                    }
                    let mut child = state.clone();
                    child.dev_deck[(child.dev_len - 1) as usize] = card as u8;
                    child.apply(action as usize, None, None).ok()?;
                    Some((child, counts[card] as f64 / total as f64))
                })
                .collect()
        }
        7 if ACTION_B[action as usize] != 2 => (0..5)
            .map(|resource| {
                let mut child = state.clone();
                if child.apply(action as usize, None, Some(resource)).is_err() {
                    child = state.clone();
                }
                (child, 0.2)
            })
            .collect(),
        12 => {
            // Preserve the frozen champion's upstream bug: each IID probability label causes
            // a fresh Balanced-Dice draw in the copied game. The shared RNG advances through
            // the same random.random plus random.choice paths as CPython.
            (2..=12)
                .map(|label| {
                    let mut simulated_dice = state.dice.clone();
                    let color = state.colors[state.current_player as usize] as usize;
                    let total = simulated_dice.draw(color, rng);
                    let choices_before_draw = simulated_dice.counts[total as usize] as usize + 1;
                    let _ = rng.randbelow(choices_before_draw);
                    let mut child = state.clone();
                    child
                        .apply(action as usize, Some(total), None)
                        .expect("drawn balanced-dice total must be consumable");
                    (child, number_probability(label))
                })
                .collect()
        }
        _ => {
            let mut child = state.clone();
            child
                .apply(action as usize, None, None)
                .expect("legal deterministic champion action must apply");
            vec![(child, 1.0)]
        }
    }
}

fn alphabeta(
    state: &RankedState,
    observer: usize,
    hidden_vp: f64,
    depth: u8,
    mut alpha: f64,
    mut beta: f64,
    rng: &mut PythonRandom,
) -> (Option<u16>, f64) {
    if depth == 0 || state.winner() >= 0 {
        return (
            None,
            evaluation::tuned_consensus(state, observer, hidden_vp),
        );
    }
    let maximizing = state.current_player as usize == observer;
    let mut best_action = None;
    let mut best_value = if maximizing {
        f64::NEG_INFINITY
    } else {
        f64::INFINITY
    };
    for action in state.legal_actions() {
        let expected = outcomes(state, action, observer, rng)
            .into_iter()
            .map(|(child, probability)| {
                probability * alphabeta(&child, observer, hidden_vp, depth - 1, alpha, beta, rng).1
            })
            .sum();
        if (maximizing && expected > best_value) || (!maximizing && expected < best_value) {
            best_action = Some(action);
            best_value = expected;
        }
        if maximizing {
            alpha = alpha.max(best_value);
        } else {
            beta = beta.min(best_value);
        }
        if alpha >= beta {
            break;
        }
    }
    (best_action, best_value)
}

#[allow(dead_code)]
pub fn choose_action(state: &mut RankedState) -> Option<u16> {
    choose_action_at_depth(state, DEPTH)
}

/// Same frozen evaluation and chance handling, searched to `depth` actions. Depth 2 is the champion.
pub fn choose_action_at_depth(state: &mut RankedState, depth: u8) -> Option<u16> {
    let actions = state.legal_actions();
    if actions.len() <= 1 {
        return actions.first().copied();
    }
    let observer = state.current_player as usize;
    let hidden_vp = opponent_hidden_vp(state, observer);
    let mut rng = state.rng.clone();
    let action = alphabeta(
        state,
        observer,
        hidden_vp,
        depth,
        f64::NEG_INFINITY,
        f64::INFINITY,
        &mut rng,
    )
    .0;
    state.rng = rng;
    action
}

#[cfg(test)]
mod tests {
    use super::{choose_action, opponent_hidden_vp};
    use crate::state::{RankedState, VICTORY_POINT};

    #[test]
    fn initial_champion_action_is_legal() {
        let mut state = RankedState::new(0);
        let action = choose_action(&mut state).unwrap();
        assert!(state.legal_actions().contains(&action));
    }

    #[test]
    fn hidden_vp_uses_only_observer_visible_card_types() {
        let mut state = RankedState::new(0);
        state.dev_hand[0][VICTORY_POINT as usize] = 1;
        state.dev_hand[1][0] = 2;
        assert_eq!(opponent_hidden_vp(&state, 0), 2.0 * 4.0 / 24.0);
    }
}
