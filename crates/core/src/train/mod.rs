//! Apprentissage des poids (chapitre 6) : optimisation évolutionnaire,
//! TD learning linéaire sur afterstate, et témoin n-tuple sans pistes.
//! Profondeur de recherche fixée à 1 pour isoler l'effet de l'information
//! donnée (features vs grille brute) et de la méthode d'apprentissage.

pub mod evolution;
pub mod ntuple;
pub mod td;

/// Un point de la courbe d'apprentissage : performance moyenne (score) sur
/// un échantillon de parties, mesurée à intervalles réguliers.
#[derive(Debug, Clone)]
pub struct LearningPoint {
    /// Nombre de parties (évolution) ou de coups (TD) vus jusqu'ici.
    pub steps: usize,
    pub mean_score: f64,
}

/// Joue `num_games` parties (seeds `0..num_games`, fixes pour comparer
/// équitablement plusieurs jeux de poids) avec un évaluateur à profondeur
/// 1, et retourne le score moyen.
pub fn evaluate_weights(evaluate: impl Fn(u64) -> f64, num_games: usize, max_moves: usize) -> f64 {
    let scores = scores_of(evaluate, num_games, max_moves, 0);
    scores.iter().sum::<u64>() as f64 / scores.len() as f64
}

/// Comme [`evaluate_weights`], mais retourne le score de chaque partie
/// individuellement (nécessaire pour calculer des intervalles de
/// confiance côté analyse). `seed_offset` permet de tirer un échantillon
/// de test disjoint de celui utilisé pendant l'entraînement.
pub fn scores_of(
    evaluate: impl Fn(u64) -> f64,
    num_games: usize,
    max_moves: usize,
    seed_offset: u64,
) -> Vec<u64> {
    use crate::agent::{Agent, EvalAgent};
    use crate::GameState;

    let mut agent = EvalAgent::new(evaluate);
    (0..num_games as u64)
        .map(|i| {
            let mut state = GameState::new(seed_offset + i);
            for _ in 0..max_moves {
                match agent.choose_move(&state) {
                    Some(dir) => {
                        state.apply_move(dir);
                        state.spawn();
                    }
                    None => break,
                }
            }
            state.score()
        })
        .collect()
}
