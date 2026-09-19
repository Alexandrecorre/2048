//! 6b. TD(0) linéaire sur l'état après déplacement (afterstate), la
//! formulation standard pour 2048 : on apprend V(afterstate) plutôt que
//! V(state), ce qui évite d'avoir à modéliser la distribution des tuiles
//! aléatoires. Fonctionne avec n'importe quel [`LinearModel`] : les
//! features du chapitre 5 (6b) ou le témoin n-tuple brut (6c).

use crate::train::{evaluate_weights, ntuple, LearningPoint};
use crate::{apply_move, features, Direction, GameState};

/// Un modèle linéaire V(board) = poids · représentation(board), avec sa
/// règle de mise à jour de gradient. Permet de réutiliser exactement le
/// même apprentissage TD pour les features conçues à la main (6b) et
/// l'encodage brut sans connaissance du jeu (6c).
pub trait LinearModel {
    fn dim(&self) -> usize;
    fn value(&self, weights: &[f64], board: u64) -> f64;
    fn update(&self, weights: &mut [f64], board: u64, error: f64, alpha: f64);
}

pub struct FeatureModel {
    pub enabled: Vec<features::Feature>,
}

impl LinearModel for FeatureModel {
    fn dim(&self) -> usize {
        self.enabled.len()
    }

    fn value(&self, weights: &[f64], board: u64) -> f64 {
        features::evaluate(board, &self.enabled, weights)
    }

    fn update(&self, weights: &mut [f64], board: u64, error: f64, alpha: f64) {
        let feats = features::compute(board, &self.enabled);
        for (w, f) in weights.iter_mut().zip(feats.iter()) {
            *w += alpha * error * f;
        }
    }
}

pub struct NTupleModel;

impl LinearModel for NTupleModel {
    fn dim(&self) -> usize {
        ntuple::DIM
    }

    fn value(&self, weights: &[f64], board: u64) -> f64 {
        ntuple::value(board, weights)
    }

    fn update(&self, weights: &mut [f64], board: u64, error: f64, alpha: f64) {
        ntuple::td_update(weights, board, error, alpha);
    }
}

#[derive(Debug, Clone)]
pub struct TdConfig {
    pub games: usize,
    pub max_moves: usize,
    pub alpha: f64,
    pub seed: u64,
    /// Enregistre un point de la courbe d'apprentissage toutes les
    /// `eval_every` parties d'entraînement.
    pub eval_every: usize,
    pub eval_games: usize,
}

/// Joue une partie et met à jour `weights` en ligne (TD(0) sur
/// afterstate). Le coup joué à chaque tour est celui qui maximise
/// `score_immédiat + V(afterstate)`, exactement la politique de
/// [`crate::agent::EvalAgent`].
fn train_one_game(model: &dyn LinearModel, weights: &mut [f64], seed: u64, max_moves: usize, alpha: f64) {
    let mut state = GameState::new(seed);
    let mut prev_afterstate: Option<u64> = None;

    for _ in 0..max_moves {
        let board = state.board();
        let mut best: Option<(Direction, u64, f64, f64)> = None;
        for dir in Direction::ALL {
            let (afterstate, gained) = apply_move(board, dir);
            if afterstate == board {
                continue;
            }
            let candidate_value = gained as f64 + model.value(weights, afterstate);
            if best.is_none_or(|(_, _, _, best_value)| candidate_value > best_value) {
                best = Some((dir, afterstate, gained as f64, candidate_value));
            }
        }

        match best {
            None => {
                if let Some(prev) = prev_afterstate {
                    let error = 0.0 - model.value(weights, prev);
                    model.update(weights, prev, error, alpha);
                }
                break;
            }
            Some((dir, afterstate, gained, _)) => {
                if let Some(prev) = prev_afterstate {
                    let target = gained + model.value(weights, afterstate);
                    let error = target - model.value(weights, prev);
                    model.update(weights, prev, error, alpha);
                }
                state.apply_move(dir);
                state.spawn();
                prev_afterstate = Some(afterstate);
            }
        }
    }
}

pub fn run(model: &dyn LinearModel, config: &TdConfig) -> (Vec<f64>, Vec<LearningPoint>) {
    let mut weights = vec![0.0; model.dim()];
    let mut history = Vec::new();

    for game_idx in 0..config.games {
        let seed = config.seed + game_idx as u64;
        train_one_game(model, &mut weights, seed, config.max_moves, config.alpha);

        if (game_idx + 1) % config.eval_every == 0 || game_idx + 1 == config.games {
            let mean_score = evaluate_weights(
                |board| model.value(&weights, board),
                config.eval_games,
                config.max_moves,
            );
            history.push(LearningPoint {
                steps: game_idx + 1,
                mean_score,
            });
        }
    }

    (weights, history)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::Feature;

    #[test]
    fn td_learns_nonzero_weights_with_feature_model() {
        let model = FeatureModel {
            enabled: vec![Feature::EmptyCells, Feature::MaxTileInCorner],
        };
        let config = TdConfig {
            games: 20,
            max_moves: 300,
            alpha: 0.01,
            seed: 1,
            eval_every: 10,
            eval_games: 4,
        };
        let (weights, history) = run(&model, &config);
        assert_eq!(weights.len(), 2);
        assert_eq!(history.len(), 2);
        assert!(weights.iter().any(|&w| w.abs() > 1e-6));
    }

    #[test]
    fn td_learns_with_ntuple_model() {
        let model = NTupleModel;
        let config = TdConfig {
            games: 10,
            max_moves: 300,
            alpha: 0.001,
            seed: 2,
            eval_every: 5,
            eval_games: 3,
        };
        let (weights, history) = run(&model, &config);
        assert_eq!(weights.len(), ntuple::DIM);
        assert_eq!(history.len(), 2);
    }
}
