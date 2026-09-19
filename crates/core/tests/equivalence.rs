//! Filet de sécurité du projet : vérifie sur un grand nombre d'états
//! aléatoires que le bitboard optimisé et l'implémentation naïve de
//! référence produisent exactement le même résultat.

use g2048_core::{apply_move, naive, Direction};
use proptest::prelude::*;

fn arb_board() -> impl Strategy<Value = u64> {
    // Exposants 0..=6 (cases vides à 64) pour générer des configurations
    // avec beaucoup de fusions possibles.
    prop::collection::vec(0u8..7, 16).prop_map(|cells| {
        let mut board = 0u64;
        for (i, c) in cells.iter().enumerate() {
            board |= (*c as u64) << (i * 4);
        }
        board
    })
}

fn arb_direction() -> impl Strategy<Value = Direction> {
    prop_oneof![
        Just(Direction::Left),
        Just(Direction::Right),
        Just(Direction::Up),
        Just(Direction::Down),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(20_000))]

    #[test]
    fn bitboard_matches_naive_reference(board in arb_board(), dir in arb_direction()) {
        let (naive_grid, naive_score) = naive::apply_move(naive::board_to_grid(board), dir);
        let expected_board = naive::grid_to_board(naive_grid);

        let (actual_board, actual_score) = apply_move(board, dir);

        prop_assert_eq!(actual_board, expected_board);
        prop_assert_eq!(actual_score, naive_score);
    }
}
