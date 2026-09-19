//! Implémentation naïve de référence (tableau 4×4 lisible), utilisée
//! uniquement pour vérifier par tests de propriétés que l'implémentation
//! bitboard optimisée produit exactement le même résultat.

use crate::Direction;

pub type Grid = [[u8; 4]; 4];

pub fn board_to_grid(board: u64) -> Grid {
    let mut g = [[0u8; 4]; 4];
    for r in 0..4 {
        for c in 0..4 {
            let i = r * 4 + c;
            g[r][c] = ((board >> (i * 4)) & 0xF) as u8;
        }
    }
    g
}

pub fn grid_to_board(g: Grid) -> u64 {
    let mut board = 0u64;
    for r in 0..4 {
        for c in 0..4 {
            let i = r * 4 + c;
            board |= (g[r][c] as u64 & 0xF) << (i * 4);
        }
    }
    board
}

fn merge_line(line: [u8; 4]) -> ([u8; 4], u32) {
    let compact: Vec<u8> = line.iter().copied().filter(|&x| x != 0).collect();
    let mut merged = Vec::with_capacity(4);
    let mut score = 0u32;
    let mut i = 0;
    while i < compact.len() {
        if i + 1 < compact.len() && compact[i] == compact[i + 1] {
            let exp = compact[i] + 1;
            merged.push(exp);
            score += 1u32 << exp;
            i += 2;
        } else {
            merged.push(compact[i]);
            i += 1;
        }
    }
    while merged.len() < 4 {
        merged.push(0);
    }
    let mut out = [0u8; 4];
    out.copy_from_slice(&merged);
    (out, score)
}

/// Applique un déplacement sur la grille (déterministe, pas de spawn).
pub fn apply_move(grid: Grid, dir: Direction) -> (Grid, u32) {
    let mut result = grid;
    let mut score = 0u32;
    match dir {
        Direction::Left => {
            for row in result.iter_mut() {
                let (new_row, s) = merge_line(*row);
                *row = new_row;
                score += s;
            }
        }
        Direction::Right => {
            for row in result.iter_mut() {
                let mut line = *row;
                line.reverse();
                let (mut new_line, s) = merge_line(line);
                new_line.reverse();
                *row = new_line;
                score += s;
            }
        }
        Direction::Up => {
            for c in 0..4 {
                let col = [result[0][c], result[1][c], result[2][c], result[3][c]];
                let (new_col, s) = merge_line(col);
                for r in 0..4 {
                    result[r][c] = new_col[r];
                }
                score += s;
            }
        }
        Direction::Down => {
            for c in 0..4 {
                let mut col = [result[0][c], result[1][c], result[2][c], result[3][c]];
                col.reverse();
                let (mut new_col, s) = merge_line(col);
                new_col.reverse();
                for r in 0..4 {
                    result[r][c] = new_col[r];
                }
                score += s;
            }
        }
    }
    (result, score)
}
