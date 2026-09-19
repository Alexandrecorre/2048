//! Bibliothèque de « pistes » (chapitre 5) : features numériques calculées
//! sur un plateau, chacune activable indépendamment par config. Une
//! évaluation = produit scalaire poids · features (chapitre 6).

/// Nom stable de chaque feature, utilisé dans les configs pour
/// activer/désactiver des pistes individuellement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Feature {
    EmptyCells,
    Monotonicity,
    Smoothness,
    MaxTileInCorner,
    MergesAvailable,
    SnakeWeighted,
}

impl Feature {
    pub const ALL: [Feature; 6] = [
        Feature::EmptyCells,
        Feature::Monotonicity,
        Feature::Smoothness,
        Feature::MaxTileInCorner,
        Feature::MergesAvailable,
        Feature::SnakeWeighted,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Feature::EmptyCells => "empty_cells",
            Feature::Monotonicity => "monotonicity",
            Feature::Smoothness => "smoothness",
            Feature::MaxTileInCorner => "max_tile_in_corner",
            Feature::MergesAvailable => "merges_available",
            Feature::SnakeWeighted => "snake_weighted",
        }
    }
}

fn cells(board: u64) -> [u8; 16] {
    let mut out = [0u8; 16];
    for (i, cell) in out.iter_mut().enumerate() {
        *cell = ((board >> (i * 4)) & 0xF) as u8;
    }
    out
}

fn cell(board: u64, r: usize, c: usize) -> u8 {
    ((board >> ((r * 4 + c) * 4)) & 0xF) as u8
}

/// Nombre de cases vides, normalisé dans `[0, 1]` (0 case vide -> 0,
/// 16 cases vides -> 1). Plus il y en a, plus le plateau a de marge de
/// manœuvre.
pub fn empty_cells(board: u64) -> f64 {
    let n = cells(board).iter().filter(|&&e| e == 0).count();
    n as f64 / 16.0
}

/// Mesure à quel point les lignes et colonnes sont triées (croissant ou
/// décroissant), normalisée par la pire monotonie possible. 1 = plateau
/// parfaitement monotone dans les deux directions, 0 = pire cas.
pub fn monotonicity(board: u64) -> f64 {
    // Pour chaque ligne/colonne, on prend le meilleur des deux sens
    // (croissant ou décroissant) et on cumule la pénalité (perte d'ordre).
    let mut total_penalty = 0i64;
    let mut worst_penalty = 0i64;

    let penalty_for_line = |line: [u8; 4]| -> i64 {
        let mut increasing = 0i64;
        let mut decreasing = 0i64;
        for w in line.windows(2) {
            let (a, b) = (w[0] as i64, w[1] as i64);
            if a > b {
                increasing += a - b;
            } else {
                decreasing += b - a;
            }
        }
        increasing.min(decreasing)
    };

    // Pire cas pour une ligne : alternance max/min sur l'échelle des
    // exposants (0..15), soit 3 différences de 15 dans le pire des sens.
    let worst_line_penalty = 15 * 3;

    for r in 0..4 {
        let line = [
            cell(board, r, 0),
            cell(board, r, 1),
            cell(board, r, 2),
            cell(board, r, 3),
        ];
        total_penalty += penalty_for_line(line);
        worst_penalty += worst_line_penalty;
    }
    for c in 0..4 {
        let line = [
            cell(board, 0, c),
            cell(board, 1, c),
            cell(board, 2, c),
            cell(board, 3, c),
        ];
        total_penalty += penalty_for_line(line);
        worst_penalty += worst_line_penalty;
    }

    1.0 - (total_penalty as f64 / worst_penalty as f64)
}

/// Mesure à quel point les tuiles voisines ont des valeurs proches
/// (différence d'exposants faible = plateau "lisse", plus facile à
/// fusionner). Normalisée dans `[0, 1]`, 1 = parfaitement lisse.
pub fn smoothness(board: u64) -> f64 {
    let mut total_diff = 0i64;
    let mut num_pairs = 0i64;
    for r in 0..4 {
        for c in 0..4 {
            let v = cell(board, r, c) as i64;
            if c + 1 < 4 {
                let right = cell(board, r, c + 1) as i64;
                total_diff += (v - right).abs();
                num_pairs += 1;
            }
            if r + 1 < 4 {
                let down = cell(board, r + 1, c) as i64;
                total_diff += (v - down).abs();
                num_pairs += 1;
            }
        }
    }
    let worst = num_pairs * 15;
    1.0 - (total_diff as f64 / worst as f64)
}

/// 1.0 si la tuile de plus grande valeur est dans un des quatre coins,
/// sinon 0.0.
pub fn max_tile_in_corner(board: u64) -> f64 {
    let all = cells(board);
    let max_exp = *all.iter().max().unwrap_or(&0);
    let corners = [
        cell(board, 0, 0),
        cell(board, 0, 3),
        cell(board, 3, 0),
        cell(board, 3, 3),
    ];
    if corners.contains(&max_exp) {
        1.0
    } else {
        0.0
    }
}

/// Nombre de fusions immédiatement disponibles (paires de tuiles voisines
/// identiques, non vides), normalisé par le nombre max de paires (24).
pub fn merges_available(board: u64) -> f64 {
    let mut count = 0u32;
    for r in 0..4 {
        for c in 0..4 {
            let v = cell(board, r, c);
            if v == 0 {
                continue;
            }
            if c + 1 < 4 && cell(board, r, c + 1) == v {
                count += 1;
            }
            if r + 1 < 4 && cell(board, r + 1, c) == v {
                count += 1;
            }
        }
    }
    count as f64 / 24.0
}

/// Somme pondérée "en serpent" : poids décroissants qui serpentent à
/// travers la grille (ligne 0 gauche->droite, ligne 1 droite->gauche,
/// etc.), pour encourager à ranger les grandes tuiles le long d'un chemin
/// continu depuis un coin. Normalisée par le maximum théorique.
const SNAKE_WEIGHTS: [[f64; 4]; 4] = [
    [15.0, 14.0, 13.0, 12.0],
    [8.0, 9.0, 10.0, 11.0],
    [7.0, 6.0, 5.0, 4.0],
    [0.0, 1.0, 2.0, 3.0],
];

pub fn snake_weighted(board: u64) -> f64 {
    let mut total = 0.0f64;
    let mut max_possible = 0.0f64;
    for r in 0..4 {
        for c in 0..4 {
            let value = if cell(board, r, c) == 0 {
                0.0
            } else {
                (1u32 << cell(board, r, c)) as f64
            };
            total += SNAKE_WEIGHTS[r][c] * value;
            max_possible += SNAKE_WEIGHTS[r][c] * (1u32 << 15) as f64;
        }
    }
    total / max_possible
}

pub fn value_of(feature: Feature, board: u64) -> f64 {
    match feature {
        Feature::EmptyCells => empty_cells(board),
        Feature::Monotonicity => monotonicity(board),
        Feature::Smoothness => smoothness(board),
        Feature::MaxTileInCorner => max_tile_in_corner(board),
        Feature::MergesAvailable => merges_available(board),
        Feature::SnakeWeighted => snake_weighted(board),
    }
}

/// Calcule toutes les features activées, dans l'ordre de `enabled`.
pub fn compute(board: u64, enabled: &[Feature]) -> Vec<f64> {
    enabled.iter().map(|&f| value_of(f, board)).collect()
}

/// Évaluation = produit scalaire poids · features.
pub fn evaluate(board: u64, enabled: &[Feature], weights: &[f64]) -> f64 {
    debug_assert_eq!(enabled.len(), weights.len());
    compute(board, enabled)
        .iter()
        .zip(weights.iter())
        .map(|(f, w)| f * w)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn board_from_cells(cs: [u8; 16]) -> u64 {
        let mut board = 0u64;
        for (i, &c) in cs.iter().enumerate() {
            board |= (c as u64) << (i * 4);
        }
        board
    }

    #[test]
    fn empty_cells_counts_zero_exponents() {
        let board = board_from_cells([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert!((empty_cells(board) - 15.0 / 16.0).abs() < 1e-9);
    }

    #[test]
    fn empty_cells_is_zero_on_a_full_board() {
        let board = board_from_cells([1; 16]);
        assert_eq!(empty_cells(board), 0.0);
    }

    #[test]
    fn monotonicity_is_maximal_on_a_perfectly_sorted_board() {
        #[rustfmt::skip]
        let board = board_from_cells([
            4, 3, 2, 1,
            4, 3, 2, 1,
            4, 3, 2, 1,
            4, 3, 2, 1,
        ]);
        assert!((monotonicity(board) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn smoothness_is_maximal_on_a_uniform_board() {
        let board = board_from_cells([3; 16]);
        assert!((smoothness(board) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn max_tile_in_corner_detects_corner_placement() {
        let mut cs = [1u8; 16];
        cs[0] = 5; // coin (0,0)
        let board = board_from_cells(cs);
        assert_eq!(max_tile_in_corner(board), 1.0);

        let mut cs2 = [1u8; 16];
        cs2[5] = 5; // pas un coin
        let board2 = board_from_cells(cs2);
        assert_eq!(max_tile_in_corner(board2), 0.0);
    }

    #[test]
    fn merges_available_counts_adjacent_equal_pairs() {
        #[rustfmt::skip]
        let board = board_from_cells([
            2, 2, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
        ]);
        assert!((merges_available(board) - 1.0 / 24.0).abs() < 1e-9);
    }

    #[test]
    fn evaluate_is_the_dot_product_of_features_and_weights() {
        let board = board_from_cells([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        let enabled = [Feature::EmptyCells, Feature::MaxTileInCorner];
        let weights = [2.0, 3.0];
        let expected = empty_cells(board) * 2.0 + max_tile_in_corner(board) * 3.0;
        assert!((evaluate(board, &enabled, &weights) - expected).abs() < 1e-9);
    }
}
