//! Interface commune aux agents. Le chapitre 4 ajoutera glouton, coin et
//! Monte Carlo ; seul l'agent aléatoire est nécessaire pour valider le
//! harnais d'expérimentation du chapitre 3.

use crate::{Direction, GameState};
use rand::Rng;

pub trait Agent {
    fn choose_move(&mut self, state: &GameState) -> Option<Direction>;
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_pcg::Pcg64Mcg;

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
}
