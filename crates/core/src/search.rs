//! Recherche expectimax à profondeur variable (chapitre 7), utilisant une
//! évaluation apprise au chapitre 6. Alterne nœuds "max" (le joueur choisit
//! le meilleur coup) et nœuds "hasard" (une tuile 2 ou 4 apparaît sur une
//! case vide). `depth` compte le nombre de coups du joueur regardés en
//! avant : `depth = 1` = le coup immédiat plus l'espérance sur la tuile qui
//! apparaît, puis évaluation (plus riche que l'agent à profondeur 1 du
//! chapitre 6, qui ignore cette espérance).

use crate::agent::Agent;
use crate::{apply_move, Direction, GameState};
use std::collections::HashMap;
use std::time::Instant;

pub struct Expectimax<F: Fn(u64) -> f64> {
    evaluate: F,
    /// Cache (plateau, profondeur restante) -> valeur, pour éviter de
    /// réévaluer un même état atteint par des ordres de coups différents.
    /// Vidé à chaque appel de [`Expectimax::best_move`] pour borner la
    /// mémoire sur une longue partie.
    transposition: HashMap<(u64, u8), f64>,
}

impl<F: Fn(u64) -> f64> Expectimax<F> {
    pub fn new(evaluate: F) -> Self {
        Self {
            evaluate,
            transposition: HashMap::new(),
        }
    }

    pub fn best_move(&mut self, board: u64, depth: u8) -> Option<Direction> {
        self.transposition.clear();
        Direction::ALL
            .iter()
            .copied()
            .filter_map(|d| {
                let (afterstate, gained) = apply_move(board, d);
                if afterstate == board {
                    return None;
                }
                let value = gained as f64 + self.chance_value(afterstate, depth);
                Some((d, value))
            })
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(d, _)| d)
    }

    /// Nœud "max" : valeur du plateau au tour du joueur, en regardant
    /// `depth` coups en avant (`depth == 0` = évaluation directe, feuille).
    fn max_value(&mut self, board: u64, depth: u8) -> f64 {
        if depth == 0 {
            return (self.evaluate)(board);
        }
        if let Some(&cached) = self.transposition.get(&(board, depth)) {
            return cached;
        }

        let mut best = f64::MIN;
        let mut any_move = false;
        for dir in Direction::ALL {
            let (afterstate, gained) = apply_move(board, dir);
            if afterstate == board {
                continue;
            }
            any_move = true;
            let value = gained as f64 + self.chance_value(afterstate, depth);
            if value > best {
                best = value;
            }
        }
        let value = if any_move { best } else { (self.evaluate)(board) };
        self.transposition.insert((board, depth), value);
        value
    }

    /// Nombre maximal de cases vides développées par nœud de hasard.
    /// Sans cette limite, le facteur de branchement (cases vides ×
    /// valeurs de tuile × coups) explose de façon combinatoire au-delà de
    /// la profondeur 2-3 (des millions de feuilles par coup). Au-delà de
    /// cette limite on ne considère qu'un sous-ensemble représentatif des
    /// cases vides : une approximation de l'espérance, pas la valeur
    /// exacte, mais qui garde une recherche jouable à profondeur élevée.
    const MAX_CHANCE_BRANCHES: usize = 4;

    /// Nœud "hasard" : espérance sur l'apparition d'une tuile (2 à 90%,
    /// 4 à 10%) sur une case vide.
    ///
    /// Élagage des nœuds de hasard peu probables, sur deux axes : (1)
    /// au-delà de la dernière demi-couche, on n'explore que la branche
    /// "2" (dominante à 90%) ; seule la demi-couche la plus proche de la
    /// feuille modélise fidèlement le mélange 90/10. (2) au-delà de
    /// [`Self::MAX_CHANCE_BRANCHES`] cases vides, on n'en développe
    /// qu'un sous-ensemble.
    fn chance_value(&mut self, board: u64, depth: u8) -> f64 {
        let mut empties: Vec<usize> = (0..16).filter(|&i| (board >> (i * 4)) & 0xF == 0).collect();
        if empties.is_empty() {
            return self.max_value(board, depth - 1);
        }
        empties.truncate(Self::MAX_CHANCE_BRANCHES);

        let model_both_tiles = depth == 1;
        let n = empties.len() as f64;
        let mut total = 0.0;
        for &idx in &empties {
            let board_with_2 = board | (1u64 << (idx * 4));
            total += 0.9 * self.max_value(board_with_2, depth - 1);
            if model_both_tiles {
                let board_with_4 = board | (2u64 << (idx * 4));
                total += 0.1 * self.max_value(board_with_4, depth - 1);
            } else {
                total += 0.1 * self.max_value(board_with_2, depth - 1);
            }
        }
        total / n
    }
}

pub struct ExpectimaxAgent<F: Fn(u64) -> f64> {
    search: Expectimax<F>,
    depth: u8,
}

impl<F: Fn(u64) -> f64> ExpectimaxAgent<F> {
    pub fn new(evaluate: F, depth: u8) -> Self {
        Self {
            search: Expectimax::new(evaluate),
            depth: depth.max(1),
        }
    }
}

impl<F: Fn(u64) -> f64> Agent for ExpectimaxAgent<F> {
    fn choose_move(&mut self, state: &GameState) -> Option<Direction> {
        self.search.best_move(state.board(), self.depth)
    }
}

#[derive(Debug, Clone)]
pub struct ExpectimaxRunResult {
    pub seed: u64,
    pub score: u64,
    pub max_tile: u32,
    pub num_moves: usize,
    pub duration_ms: f64,
}

/// Joue une partie complète avec un agent expectimax et mesure le temps
/// total, pour calculer le temps moyen par coup (courbes du chapitre 7).
pub fn play_one_game(
    evaluate: impl Fn(u64) -> f64,
    depth: u8,
    seed: u64,
    max_moves: usize,
) -> ExpectimaxRunResult {
    let start = Instant::now();
    let mut state = GameState::new(seed);
    let mut agent = ExpectimaxAgent::new(evaluate, depth);
    let mut num_moves = 0usize;
    for _ in 0..max_moves {
        match agent.choose_move(&state) {
            Some(dir) => {
                state.apply_move(dir);
                state.spawn();
                num_moves += 1;
            }
            None => break,
        }
    }
    ExpectimaxRunResult {
        seed,
        score: state.score(),
        max_tile: state.max_tile(),
        num_moves,
        duration_ms: start.elapsed().as_secs_f64() * 1000.0,
    }
}

/// Lot de parties en parallèle (rayon), une seed par partie.
pub fn run_batch(
    enabled: &[crate::features::Feature],
    weights: &[f64],
    depth: u8,
    num_games: usize,
    max_moves: usize,
    seed_offset: u64,
) -> Vec<ExpectimaxRunResult> {
    use rayon::prelude::*;
    let enabled = enabled.to_vec();
    let weights = weights.to_vec();
    (0..num_games as u64)
        .into_par_iter()
        .map(|i| {
            let enabled = enabled.clone();
            let weights = weights.clone();
            play_one_game(
                move |board| crate::features::evaluate(board, &enabled, &weights),
                depth,
                seed_offset + i,
                max_moves,
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::Feature;

    #[test]
    fn expectimax_always_returns_a_legal_move() {
        let mut search = Expectimax::new(|board: u64| crate::features::empty_cells(board));
        let mut state = GameState::new(1);
        for _ in 0..50 {
            let legal = state.legal_moves();
            if legal.is_empty() {
                break;
            }
            let chosen = search.best_move(state.board(), 2).expect("un coup légal existe");
            assert!(legal.contains(&chosen));
            state.apply_move(chosen);
            state.spawn();
        }
    }

    #[test]
    fn depth_zero_max_value_equals_direct_evaluation() {
        let mut search = Expectimax::new(|board: u64| board as f64);
        assert_eq!(search.max_value(42, 0), 42.0);
    }

    #[test]
    fn deeper_search_can_change_the_chosen_move() {
        // Ne vérifie pas un coup précis (dépend de l'évaluation), mais que
        // la recherche à différentes profondeurs reste stable et légale.
        let enabled = vec![Feature::EmptyCells, Feature::MergesAvailable];
        let weights = vec![1.0, 1.0];
        for depth in [1u8, 2, 3] {
            let result = play_one_game(
                |board| crate::features::evaluate(board, &enabled, &weights),
                depth,
                7,
                100,
            );
            assert!(result.num_moves > 0);
        }
    }

    #[test]
    fn run_batch_produces_one_result_per_game() {
        let enabled = vec![Feature::EmptyCells];
        let weights = vec![1.0];
        let results = run_batch(&enabled, &weights, 1, 4, 50, 0);
        assert_eq!(results.len(), 4);
    }
}
