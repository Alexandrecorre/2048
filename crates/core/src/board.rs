//! Représentation bitboard du plateau et logique de déplacement.
//!
//! Le plateau est un `u64` : 16 cases de 4 bits chacune, contenant l'exposant
//! de la tuile (0 = case vide, e = tuile de valeur 2^e). La case d'indice
//! `i` (0..16, row-major : `i = row * 4 + col`) occupe les bits `[i*4, i*4+4)`.

use rand::Rng;
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Left,
    Right,
    Up,
    Down,
}

impl Direction {
    pub const ALL: [Direction; 4] = [
        Direction::Left,
        Direction::Right,
        Direction::Up,
        Direction::Down,
    ];
}

struct MoveTable {
    result: Box<[u16; 65536]>,
    score: Box<[u32; 65536]>,
}

/// Fusionne une ligne de 4 cases vers la gauche (index 0 = case la plus à gauche).
/// Retourne la ligne fusionnée et le score gagné.
fn merge_line(cells: [u8; 4]) -> ([u8; 4], u32) {
    let mut compact = [0u8; 4];
    let mut n = 0usize;
    for &c in cells.iter() {
        if c != 0 {
            compact[n] = c;
            n += 1;
        }
    }

    let mut merged = [0u8; 4];
    let mut m = 0usize;
    let mut score = 0u32;
    let mut i = 0usize;
    while i < n {
        if i + 1 < n && compact[i] == compact[i + 1] {
            let exp = compact[i] + 1;
            merged[m] = exp;
            score += 1u32 << exp;
            m += 1;
            i += 2;
        } else {
            merged[m] = compact[i];
            m += 1;
            i += 1;
        }
    }
    (merged, score)
}

fn build_left_table() -> MoveTable {
    let mut result = Box::new([0u16; 65536]);
    let mut score = Box::new([0u32; 65536]);
    for row_val in 0u32..65536 {
        let row = row_val as u16;
        let cells = [
            (row & 0xF) as u8,
            ((row >> 4) & 0xF) as u8,
            ((row >> 8) & 0xF) as u8,
            ((row >> 12) & 0xF) as u8,
        ];
        let (merged, s) = merge_line(cells);
        let new_row = merged[0] as u16
            | (merged[1] as u16) << 4
            | (merged[2] as u16) << 8
            | (merged[3] as u16) << 12;
        result[row_val as usize] = new_row;
        score[row_val as usize] = s;
    }
    MoveTable { result, score }
}

fn left_table() -> &'static MoveTable {
    static TABLE: OnceLock<MoveTable> = OnceLock::new();
    TABLE.get_or_init(build_left_table)
}

fn get_row(board: u64, r: usize) -> u16 {
    ((board >> (r * 16)) & 0xFFFF) as u16
}

fn set_row(board: u64, r: usize, row: u16) -> u64 {
    let mask = 0xFFFFu64 << (r * 16);
    (board & !mask) | ((row as u64) << (r * 16))
}

fn reverse_row(row: u16) -> u16 {
    let n0 = row & 0xF;
    let n1 = (row >> 4) & 0xF;
    let n2 = (row >> 8) & 0xF;
    let n3 = (row >> 12) & 0xF;
    (n0 << 12) | (n1 << 8) | (n2 << 4) | n3
}

fn get_cell(board: u64, r: usize, c: usize) -> u8 {
    ((board >> (r * 16 + c * 4)) & 0xF) as u8
}

fn set_cell(board: u64, r: usize, c: usize, v: u8) -> u64 {
    let shift = r * 16 + c * 4;
    let mask = 0xFu64 << shift;
    (board & !mask) | (((v as u64) & 0xF) << shift)
}

/// Transpose le plateau (échange lignes et colonnes), pour réutiliser la
/// table de déplacement "gauche" afin de calculer haut/bas.
fn transpose(board: u64) -> u64 {
    let mut out = 0u64;
    for r in 0..4 {
        for c in 0..4 {
            out = set_cell(out, c, r, get_cell(board, r, c));
        }
    }
    out
}

fn move_rows(board: u64, reverse: bool) -> (u64, u32) {
    let table = left_table();
    let mut new_board = 0u64;
    let mut total_score = 0u32;
    for r in 0..4 {
        let mut row = get_row(board, r);
        if reverse {
            row = reverse_row(row);
        }
        let mut new_row = table.result[row as usize];
        total_score += table.score[row as usize];
        if reverse {
            new_row = reverse_row(new_row);
        }
        new_board = set_row(new_board, r, new_row);
    }
    (new_board, total_score)
}

/// Applique un déplacement de façon déterministe. Ne fait pas apparaître de
/// nouvelle tuile : c'est le rôle de [`spawn`]. Retourne le nouveau plateau
/// et le score gagné par les fusions.
pub fn apply_move(board: u64, dir: Direction) -> (u64, u32) {
    match dir {
        Direction::Left => move_rows(board, false),
        Direction::Right => move_rows(board, true),
        Direction::Up => {
            let t = transpose(board);
            let (moved, s) = move_rows(t, false);
            (transpose(moved), s)
        }
        Direction::Down => {
            let t = transpose(board);
            let (moved, s) = move_rows(t, true);
            (transpose(moved), s)
        }
    }
}

/// Fait apparaître une tuile (2 avec probabilité 0.9, 4 avec probabilité 0.1)
/// sur une case vide choisie uniformément au hasard. Ne fait rien si le
/// plateau est plein. Le RNG est injecté, jamais global.
pub fn spawn<R: Rng + ?Sized>(board: u64, rng: &mut R) -> u64 {
    let empties: Vec<usize> = (0..16).filter(|&i| (board >> (i * 4)) & 0xF == 0).collect();
    if empties.is_empty() {
        return board;
    }
    let idx = empties[rng.gen_range(0..empties.len())];
    let exponent: u64 = if rng.gen_bool(0.9) { 1 } else { 2 };
    board | (exponent << (idx * 4))
}

pub fn max_tile(board: u64) -> u32 {
    let mut max_exp = 0u8;
    for i in 0..16 {
        let e = ((board >> (i * 4)) & 0xF) as u8;
        if e > max_exp {
            max_exp = e;
        }
    }
    if max_exp == 0 {
        0
    } else {
        1u32 << max_exp
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn board_from_cells(cells: [u8; 16]) -> u64 {
        let mut board = 0u64;
        for (i, &c) in cells.iter().enumerate() {
            board |= (c as u64) << (i * 4);
        }
        board
    }

    #[test]
    fn left_merges_adjacent_equal_tiles() {
        // ligne [2,2,0,0] (exposants: 1,1,0,0) -> [4,0,0,0] (exposant 2), score 4
        #[rustfmt::skip]
        let board = board_from_cells([
            1, 1, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
        ]);
        let (new_board, score) = apply_move(board, Direction::Left);
        assert_eq!(get_row(new_board, 0), 0x0002);
        assert_eq!(score, 4);
    }

    #[test]
    fn right_merges_toward_the_right_edge() {
        #[rustfmt::skip]
        let board = board_from_cells([
            1, 1, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
        ]);
        let (new_board, score) = apply_move(board, Direction::Right);
        assert_eq!(get_row(new_board, 0), 0x2000);
        assert_eq!(score, 4);
    }

    #[test]
    fn up_merges_a_column_toward_the_top() {
        #[rustfmt::skip]
        let board = board_from_cells([
            1, 0, 0, 0,
            1, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
        ]);
        let (new_board, score) = apply_move(board, Direction::Up);
        assert_eq!(get_cell(new_board, 0, 0), 2);
        assert_eq!(get_cell(new_board, 1, 0), 0);
        assert_eq!(score, 4);
    }

    #[test]
    fn down_merges_a_column_toward_the_bottom() {
        #[rustfmt::skip]
        let board = board_from_cells([
            1, 0, 0, 0,
            1, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
        ]);
        let (new_board, score) = apply_move(board, Direction::Down);
        assert_eq!(get_cell(new_board, 3, 0), 2);
        assert_eq!(get_cell(new_board, 0, 0), 0);
        assert_eq!(score, 4);
    }

    #[test]
    fn spawn_fills_an_empty_cell() {
        let mut rng = rand_pcg::Pcg64Mcg::seed_from_u64(42);
        let board = spawn(0u64, &mut rng);
        let occupied = (0..16).filter(|&i| (board >> (i * 4)) & 0xF != 0).count();
        assert_eq!(occupied, 1);
    }

    #[test]
    fn spawn_does_nothing_on_a_full_board() {
        let full = board_from_cells([1; 16]);
        let mut rng = rand_pcg::Pcg64Mcg::seed_from_u64(1);
        assert_eq!(spawn(full, &mut rng), full);
    }

    #[test]
    fn max_tile_reads_the_highest_exponent() {
        let board = board_from_cells([1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(max_tile(board), 8);
    }
}
