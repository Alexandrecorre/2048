//! Moteur de jeu 2048 : représentation bitboard, déplacements déterministes
//! et apparition de tuiles aléatoire (RNG injecté, jamais global).

mod board;
pub mod agent;
pub mod naive;
pub mod runner;

pub use board::{apply_move, max_tile, spawn, Direction};

/// Utilisé par le binding pyo3 "hello world" du chapitre 1, en attendant
/// que le chapitre 3 expose l'API `run_batch`/Gymnasium complète.
pub fn hello() -> String {
    "hello from g2048-core".to_string()
}

use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;

/// État de jeu complet : plateau, score cumulé et RNG. `apply_move` et
/// `spawn` restent disponibles en fonctions libres (déterministe /
/// aléatoire séparés) pour l'expectimax du chapitre 7.
pub struct GameState {
    board: u64,
    score: u64,
    rng: Pcg64Mcg,
}

impl GameState {
    pub fn new(seed: u64) -> Self {
        let mut rng = Pcg64Mcg::seed_from_u64(seed);
        let mut board = 0u64;
        board = spawn(board, &mut rng);
        board = spawn(board, &mut rng);
        GameState { board, score: 0, rng }
    }

    pub fn board(&self) -> u64 {
        self.board
    }

    pub fn legal_moves(&self) -> Vec<Direction> {
        Direction::ALL
            .iter()
            .copied()
            .filter(|&dir| apply_move(self.board, dir).0 != self.board)
            .collect()
    }

    /// Applique un coup déterministe et ajoute le score gagné. Retourne
    /// `false` si le coup ne change pas le plateau (coup illégal).
    pub fn apply_move(&mut self, dir: Direction) -> bool {
        let (new_board, gained) = apply_move(self.board, dir);
        if new_board == self.board {
            return false;
        }
        self.board = new_board;
        self.score += gained as u64;
        true
    }

    /// Fait apparaître une nouvelle tuile en utilisant le RNG interne.
    pub fn spawn(&mut self) {
        self.board = spawn(self.board, &mut self.rng);
    }

    pub fn is_terminal(&self) -> bool {
        self.legal_moves().is_empty()
    }

    pub fn score(&self) -> u64 {
        self.score
    }

    pub fn max_tile(&self) -> u32 {
        max_tile(self.board)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_game_has_two_tiles() {
        let state = GameState::new(1);
        let occupied = (0..16)
            .filter(|&i| (state.board() >> (i * 4)) & 0xF != 0)
            .count();
        assert_eq!(occupied, 2);
    }

    #[test]
    fn same_seed_gives_same_initial_board() {
        let a = GameState::new(7);
        let b = GameState::new(7);
        assert_eq!(a.board(), b.board());
    }

    #[test]
    fn playing_a_legal_move_updates_board_and_can_spawn() {
        let mut state = GameState::new(3);
        if let Some(&dir) = state.legal_moves().first() {
            assert!(state.apply_move(dir));
            state.spawn();
        }
    }

    #[test]
    fn is_terminal_true_when_no_legal_moves() {
        // plateau plein, sans fusions possibles (damier alterné) : aucun coup légal.
        let mut cells = [0u8; 16];
        for r in 0..4 {
            for c in 0..4 {
                cells[r * 4 + c] = if (r + c) % 2 == 0 { 1 } else { 2 };
            }
        }
        let mut board = 0u64;
        for (i, &v) in cells.iter().enumerate() {
            board |= (v as u64) << (i * 4);
        }
        let state = GameState {
            board,
            score: 0,
            rng: Pcg64Mcg::seed_from_u64(0),
        };
        assert!(state.is_terminal());
    }
}
