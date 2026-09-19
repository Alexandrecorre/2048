//! 6c. Témoin "sans pistes" : au lieu de features conçues à la main
//! (chapitre 5), un modèle linéaire sur un encodage one-hot brut de
//! chaque case (16 cases × 16 valeurs d'exposant possibles = 256 poids).
//! C'est la référence "aucune connaissance donnée" : dégénérescence
//! minimale d'un réseau n-tuple (des 1-tuples, un par case).

pub const NUM_CELLS: usize = 16;
pub const NUM_EXPONENTS: usize = 16;
pub const DIM: usize = NUM_CELLS * NUM_EXPONENTS;

/// Indices (dans le vecteur de poids de taille [`DIM`]) actifs pour ce
/// plateau : un par case, correspondant à son exposant courant.
pub fn active_indices(board: u64) -> [usize; NUM_CELLS] {
    let mut idx = [0usize; NUM_CELLS];
    for (i, slot) in idx.iter_mut().enumerate() {
        let exponent = ((board >> (i * 4)) & 0xF) as usize;
        *slot = i * NUM_EXPONENTS + exponent;
    }
    idx
}

pub fn value(board: u64, weights: &[f64]) -> f64 {
    active_indices(board).iter().map(|&i| weights[i]).sum()
}

pub fn td_update(weights: &mut [f64], board: u64, error: f64, alpha: f64) {
    for &i in active_indices(board).iter() {
        weights[i] += alpha * error;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn value_sums_the_weight_of_each_cells_current_exponent() {
        let mut weights = vec![0.0; DIM];
        weights[0 * NUM_EXPONENTS + 3] = 2.0; // case 0, exposant 3
        weights[1 * NUM_EXPONENTS + 0] = 5.0; // case 1, exposant 0 (vide)
        let board: u64 = 3; // case 0 = exposant 3, reste vide
        assert_eq!(value(board, &weights), 2.0 + 5.0);
    }

    #[test]
    fn td_update_only_touches_active_weights() {
        let mut weights = vec![0.0; DIM];
        let board: u64 = 3; // case 0 = exposant 3, cases 1..16 vides (exposant 0)
        td_update(&mut weights, board, 1.0, 0.1);
        assert!((weights[0 * NUM_EXPONENTS + 3] - 0.1).abs() < 1e-9);
        // case 1 est vide (exposant 0, actif) : son poids "exposant 0" est
        // bien mis à jour, mais pas un exposant inactif comme 5.
        assert!((weights[1 * NUM_EXPONENTS + 0] - 0.1).abs() < 1e-9);
        assert_eq!(weights[1 * NUM_EXPONENTS + 5], 0.0);
    }
}
