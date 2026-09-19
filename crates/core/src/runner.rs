//! Harnais d'expérimentation (chapitre 3) : config TOML, exécution d'un lot
//! de parties en parallèle (rayon), et rejeu déterministe à partir d'une
//! seed et d'une séquence de coups.

use crate::agent::{Agent, CornerAgent, GreedyAgent, MonteCarloAgent, RandomAgent};
use crate::{apply_move, Direction, GameState};
use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;
use rayon::prelude::*;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct ExperimentConfig {
    pub name: String,
    pub agent: String,
    pub num_games: usize,
    pub seed: u64,
    #[serde(default)]
    pub max_moves: Option<usize>,
    /// Nombre de simulations aléatoires par coup possible (agent "monte_carlo").
    #[serde(default = "default_mc_simulations")]
    pub mc_simulations: usize,
    /// Profondeur maximale (en coups) d'une simulation Monte Carlo.
    #[serde(default = "default_mc_rollout_moves")]
    pub mc_rollout_moves: usize,
    /// Pistes (chapitre 5) et poids (appris au chapitre 6) utilisés par
    /// l'agent "expectimax" pour évaluer les feuilles de la recherche.
    #[serde(default)]
    pub features: Vec<String>,
    #[serde(default)]
    pub weights: Vec<f64>,
    /// Profondeur de recherche (nombre de coups regardés en avant) de
    /// l'agent "expectimax" (chapitre 7).
    #[serde(default = "default_search_depth")]
    pub search_depth: u8,
}

fn default_search_depth() -> u8 {
    1
}

fn default_mc_simulations() -> usize {
    50
}

fn default_mc_rollout_moves() -> usize {
    200
}

#[derive(Debug, Clone)]
pub struct GameResult {
    pub seed: u64,
    pub score: u64,
    pub max_tile: u32,
    pub num_moves: usize,
    pub duration_ms: f64,
    pub moves: Vec<u8>,
}

pub fn parse_config(text: &str) -> Result<ExperimentConfig, toml::de::Error> {
    toml::from_str(text)
}

pub fn direction_to_u8(d: Direction) -> u8 {
    match d {
        Direction::Left => 0,
        Direction::Right => 1,
        Direction::Up => 2,
        Direction::Down => 3,
    }
}

pub fn u8_to_direction(v: u8) -> Direction {
    match v {
        0 => Direction::Left,
        1 => Direction::Right,
        2 => Direction::Up,
        _ => Direction::Down,
    }
}

fn make_agent(config: &ExperimentConfig, seed: u64) -> Result<Box<dyn Agent>, String> {
    // Décorrélée de la seed de spawn (constante golden-ratio) pour que les
    // choix aléatoires de l'agent ne soient pas synchronisés avec les tuiles.
    let agent_seed = seed ^ 0x9E37_79B9_7F4A_7C15;
    match config.agent.as_str() {
        "random" => Ok(Box::new(RandomAgent::new(Pcg64Mcg::seed_from_u64(
            agent_seed,
        )))),
        "greedy" => Ok(Box::new(GreedyAgent)),
        "corner" => Ok(Box::new(CornerAgent::new())),
        "monte_carlo" => Ok(Box::new(MonteCarloAgent::new(
            Pcg64Mcg::seed_from_u64(agent_seed),
            config.mc_simulations,
            config.mc_rollout_moves,
        ))),
        "expectimax" => {
            let enabled: Vec<crate::features::Feature> = config
                .features
                .iter()
                .map(|n| {
                    crate::features::Feature::from_name(n)
                        .ok_or_else(|| format!("feature inconnue: '{n}'"))
                })
                .collect::<Result<_, _>>()?;
            if enabled.len() != config.weights.len() {
                return Err(format!(
                    "expectimax: {} features mais {} poids",
                    enabled.len(),
                    config.weights.len()
                ));
            }
            let weights = config.weights.clone();
            Ok(Box::new(crate::search::ExpectimaxAgent::new(
                move |board| crate::features::evaluate(board, &enabled, &weights),
                config.search_depth,
            )))
        }
        other => Err(format!("agent inconnu: '{other}'")),
    }
}

/// Joue une partie complète avec l'agent donné. Le résultat contient la
/// séquence de coups jouée : `seed + moves` suffit à rejouer la partie à
/// l'identique via [`replay`], indépendamment de l'agent qui l'a produite.
pub fn play_one_game(config: &ExperimentConfig, seed: u64) -> Result<GameResult, String> {
    let start = std::time::Instant::now();
    let mut state = GameState::new(seed);
    let mut agent = make_agent(config, seed)?;
    let limit = config.max_moves.unwrap_or(usize::MAX);
    let mut moves_log = Vec::new();

    while moves_log.len() < limit {
        match agent.choose_move(&state) {
            Some(dir) => {
                let moved = state.apply_move(dir);
                debug_assert!(moved, "l'agent a proposé un coup illégal");
                moves_log.push(direction_to_u8(dir));
                state.spawn();
            }
            None => break,
        }
    }

    Ok(GameResult {
        seed,
        score: state.score(),
        max_tile: state.max_tile(),
        num_moves: moves_log.len(),
        duration_ms: start.elapsed().as_secs_f64() * 1000.0,
        moves: moves_log,
    })
}

/// Exécute un lot de parties en parallèle (une seed par partie, dérivée de
/// `config.seed`).
pub fn run_batch(config: &ExperimentConfig) -> Result<Vec<GameResult>, String> {
    (0..config.num_games)
        .into_par_iter()
        .map(|i| play_one_game(config, config.seed + i as u64))
        .collect()
}

/// Rejoue une partie à partir de sa seed et de sa séquence de coups
/// enregistrée, sans dépendre de l'agent d'origine.
pub fn replay(seed: u64, moves: &[u8]) -> Result<GameResult, String> {
    let start = std::time::Instant::now();
    let mut state = GameState::new(seed);
    for &m in moves {
        let dir = u8_to_direction(m);
        if !state.apply_move(dir) {
            return Err(format!("coup illégal enregistré à l'index {m}"));
        }
        state.spawn();
    }
    Ok(GameResult {
        seed,
        score: state.score(),
        max_tile: state.max_tile(),
        num_moves: moves.len(),
        duration_ms: start.elapsed().as_secs_f64() * 1000.0,
        moves: moves.to_vec(),
    })
}

#[derive(Debug, Clone)]
pub struct MoveOption {
    pub direction: u8,
    pub legal: bool,
    pub gained: u32,
}

#[derive(Debug, Clone)]
pub struct TrajectoryStep {
    /// Plateau avant le coup (16 exposants, case 0 = haut-gauche).
    pub board_before: [u8; 16],
    pub chosen_direction: u8,
    pub gained: u32,
    pub score_after: u64,
    /// Pour chaque direction : légale ou non, et score immédiat si jouée.
    /// Calculé sur le plateau brut du jeu, indépendamment de l'agent
    /// d'origine (chapitre 9 : viewer).
    pub move_options: Vec<MoveOption>,
    /// Valeurs des 6 features du chapitre 5 sur `board_before`, dans
    /// l'ordre de [`crate::features::Feature::ALL`] (pour le viewer).
    pub features: Vec<f64>,
}

fn board_to_exponents(board: u64) -> [u8; 16] {
    let mut cells = [0u8; 16];
    for (i, cell) in cells.iter_mut().enumerate() {
        *cell = ((board >> (i * 4)) & 0xF) as u8;
    }
    cells
}

/// Rejoue une partie coup par coup et retourne, pour chaque coup, le
/// plateau avant, le coup choisi et le score immédiat de chacune des 4
/// directions (jouable ou non) — pour le viewer de replays (chapitre 9).
pub fn replay_trajectory(seed: u64, moves: &[u8]) -> Result<Vec<TrajectoryStep>, String> {
    let mut state = GameState::new(seed);
    let mut steps = Vec::with_capacity(moves.len());

    for &m in moves {
        let board_before = state.board();
        let move_options = Direction::ALL
            .iter()
            .map(|&d| {
                let (after, gained) = apply_move(board_before, d);
                MoveOption {
                    direction: direction_to_u8(d),
                    legal: after != board_before,
                    gained,
                }
            })
            .collect();

        let dir = u8_to_direction(m);
        let (_, gained) = apply_move(board_before, dir);
        if !state.apply_move(dir) {
            return Err(format!("coup illégal enregistré à l'index {m}"));
        }
        state.spawn();

        steps.push(TrajectoryStep {
            board_before: board_to_exponents(board_before),
            chosen_direction: m,
            gained,
            score_after: state.score(),
            move_options,
            features: crate::features::compute(board_before, &crate::features::Feature::ALL),
        });
    }

    Ok(steps)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_config() -> ExperimentConfig {
        ExperimentConfig {
            name: "test".to_string(),
            agent: "random".to_string(),
            num_games: 8,
            seed: 123,
            max_moves: Some(500),
            mc_simulations: default_mc_simulations(),
            mc_rollout_moves: default_mc_rollout_moves(),
            features: Vec::new(),
            weights: Vec::new(),
            search_depth: default_search_depth(),
        }
    }

    #[test]
    fn parses_minimal_toml_config() {
        let text = r#"
            name = "baseline_random"
            agent = "random"
            num_games = 100
            seed = 42
        "#;
        let config = parse_config(text).unwrap();
        assert_eq!(config.name, "baseline_random");
        assert_eq!(config.num_games, 100);
        assert_eq!(config.max_moves, None);
    }

    #[test]
    fn run_batch_produces_one_result_per_game() {
        let config = sample_config();
        let results = run_batch(&config).unwrap();
        assert_eq!(results.len(), 8);
    }

    #[test]
    fn every_reference_agent_can_play_a_batch() {
        for agent in ["random", "greedy", "corner", "monte_carlo"] {
            let mut config = sample_config();
            config.agent = agent.to_string();
            config.num_games = 2;
            config.mc_simulations = 3;
            config.mc_rollout_moves = 20;
            let results = run_batch(&config).unwrap();
            assert_eq!(results.len(), 2);
        }
    }

    #[test]
    fn replay_trajectory_matches_replay_final_score_and_has_one_step_per_move() {
        let config = sample_config();
        let results = run_batch(&config).unwrap();
        for original in &results {
            let trajectory = replay_trajectory(original.seed, &original.moves).unwrap();
            assert_eq!(trajectory.len(), original.moves.len());
            if let Some(last) = trajectory.last() {
                assert_eq!(last.score_after, original.score);
            }
            for step in &trajectory {
                assert_eq!(step.move_options.len(), 4);
                assert!(step.move_options.iter().any(|m| m.legal));
            }
        }
    }

    #[test]
    fn replay_reproduces_the_exact_same_outcome() {
        let config = sample_config();
        let results = run_batch(&config).unwrap();
        for original in results {
            let replayed = replay(original.seed, &original.moves).unwrap();
            assert_eq!(replayed.score, original.score);
            assert_eq!(replayed.max_tile, original.max_tile);
            assert_eq!(replayed.num_moves, original.num_moves);
        }
    }

    #[test]
    fn unknown_agent_is_a_clean_error_not_a_panic() {
        let mut config = sample_config();
        config.agent = "nonexistent".to_string();
        assert!(run_batch(&config).is_err());
    }
}
