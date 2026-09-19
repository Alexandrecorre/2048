//! Agents de référence (chapitre 4) : aléatoire, glouton, coin, Monte Carlo.
//! Toutes les IA futures (chapitres 5+) seront comparées à ces lignes.

use crate::{apply_move, spawn, Direction, GameState};
use rand::Rng;

pub trait Agent {
    fn choose_move(&mut self, state: &GameState) -> Option<Direction>;
}

fn legal_moves_of(board: u64) -> Vec<Direction> {
    Direction::ALL
        .iter()
        .copied()
        .filter(|&d| apply_move(board, d).0 != board)
        .collect()
}

/// Choisit un coup légal uniformément au hasard.
pub struct RandomAgent<R: Rng> {
    rng: R,
}

impl<R: Rng> RandomAgent<R> {
    pub fn new(rng: R) -> Self {
        Self { rng }
    }
}

impl<R: Rng> Agent for RandomAgent<R> {
    fn choose_move(&mut self, state: &GameState) -> Option<Direction> {
        let moves = state.legal_moves();
        if moves.is_empty() {
            return None;
        }
        let idx = self.rng.gen_range(0..moves.len());
        Some(moves[idx])
    }
}

/// Choisit le coup qui maximise le score immédiat gagné par les fusions.
pub struct GreedyAgent;

impl Agent for GreedyAgent {
    fn choose_move(&mut self, state: &GameState) -> Option<Direction> {
        Direction::ALL
            .iter()
            .copied()
            .filter_map(|d| {
                let (new_board, gained) = apply_move(state.board(), d);
                if new_board == state.board() {
                    None
                } else {
                    Some((d, gained))
                }
            })
            .max_by_key(|&(_, gained)| gained)
            .map(|(d, _)| d)
    }
}

/// Stratégie fixe « coin » : essaie toujours les directions dans le même
/// ordre de priorité (bas, gauche, haut, droite) et joue la première qui
/// est légale, pour garder les grosses tuiles regroupées dans un coin.
pub struct CornerAgent {
    priority: [Direction; 4],
}

impl CornerAgent {
    pub fn new() -> Self {
        Self {
            priority: [
                Direction::Down,
                Direction::Left,
                Direction::Up,
                Direction::Right,
            ],
        }
    }
}

impl Default for CornerAgent {
    fn default() -> Self {
        Self::new()
    }
}

impl Agent for CornerAgent {
    fn choose_move(&mut self, state: &GameState) -> Option<Direction> {
        self.priority
            .iter()
            .copied()
            .find(|&d| apply_move(state.board(), d).0 != state.board())
    }
}

/// Pour chaque coup légal, joue `num_simulations` parties aléatoires de
/// profondeur `max_rollout_moves` et choisit le coup dont les simulations
/// donnent en moyenne le meilleur score.
pub struct MonteCarloAgent<R: Rng> {
    rng: R,
    num_simulations: usize,
    max_rollout_moves: usize,
}

impl<R: Rng> MonteCarloAgent<R> {
    pub fn new(rng: R, num_simulations: usize, max_rollout_moves: usize) -> Self {
        Self {
            rng,
            num_simulations,
            max_rollout_moves,
        }
    }

    fn rollout(&mut self, start_board: u64) -> u64 {
        let mut board = spawn(start_board, &mut self.rng);
        let mut score = 0u64;
        for _ in 0..self.max_rollout_moves {
            let legal = legal_moves_of(board);
            if legal.is_empty() {
                break;
            }
            let dir = legal[self.rng.gen_range(0..legal.len())];
            let (new_board, gained) = apply_move(board, dir);
            score += gained as u64;
            board = spawn(new_board, &mut self.rng);
        }
        score
    }
}

impl<R: Rng> Agent for MonteCarloAgent<R> {
    fn choose_move(&mut self, state: &GameState) -> Option<Direction> {
        let legal = state.legal_moves();
        if legal.is_empty() {
            return None;
        }

        let mut best_dir = legal[0];
        let mut best_avg = f64::MIN;
        for &dir in &legal {
            let (board_after_move, immediate_gain) = apply_move(state.board(), dir);
            let mut total = 0u64;
            for _ in 0..self.num_simulations {
                total += immediate_gain as u64 + self.rollout(board_after_move);
            }
            let avg = total as f64 / self.num_simulations as f64;
            if avg > best_avg {
                best_avg = avg;
                best_dir = dir;
            }
        }
        Some(best_dir)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_pcg::Pcg64Mcg;

    fn play_out(mut agent: impl Agent, seed: u64, max_moves: usize) -> GameState {
        let mut state = GameState::new(seed);
        for _ in 0..max_moves {
            match agent.choose_move(&state) {
                Some(dir) => {
                    assert!(state.apply_move(dir), "l'agent a proposé un coup illégal");
                    state.spawn();
                }
                None => break,
            }
        }
        state
    }

    #[test]
    fn random_agent_always_picks_a_legal_move() {
        let mut state = GameState::new(5);
        let mut agent = RandomAgent::new(Pcg64Mcg::seed_from_u64(99));
        for _ in 0..200 {
            let legal = state.legal_moves();
            if legal.is_empty() {
                assert_eq!(agent.choose_move(&state), None);
                break;
            }
            let chosen = agent.choose_move(&state).expect("un coup légal existe");
            assert!(legal.contains(&chosen));
            state.apply_move(chosen);
            state.spawn();
        }
    }

    #[test]
    fn greedy_agent_plays_full_games_without_illegal_moves() {
        play_out(GreedyAgent, 1, 500);
    }

    #[test]
    fn corner_agent_plays_full_games_without_illegal_moves() {
        play_out(CornerAgent::new(), 2, 500);
    }

    #[test]
    fn monte_carlo_agent_plays_full_games_without_illegal_moves() {
        let agent = MonteCarloAgent::new(Pcg64Mcg::seed_from_u64(7), 4, 20);
        play_out(agent, 3, 30);
    }
}
