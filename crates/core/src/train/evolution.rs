//! 6a. Optimisation évolutionnaire (algorithme génétique simple, μ+λ) :
//! chaque individu est un vecteur de poids, sa fitness est sa performance
//! moyenne sur K parties à seeds fixes. Simple, robuste, très
//! parallélisable (rayon).

use crate::features::{evaluate, Feature};
use crate::train::{evaluate_weights, LearningPoint};
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use rand_pcg::Pcg64Mcg;
use rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct EvolutionConfig {
    pub enabled: Vec<Feature>,
    pub population_size: usize,
    pub generations: usize,
    pub games_per_eval: usize,
    pub max_moves: usize,
    pub mutation_std: f64,
    pub elite_fraction: f64,
    pub seed: u64,
}

pub struct EvolutionResult {
    pub best_weights: Vec<f64>,
    /// Une entrée par génération : `steps` = numéro de génération.
    pub history: Vec<LearningPoint>,
}

fn fitness_of(weights: &[f64], enabled: &[Feature], config: &EvolutionConfig) -> f64 {
    let enabled = enabled.to_vec();
    let weights = weights.to_vec();
    evaluate_weights(
        move |board| evaluate(board, &enabled, &weights),
        config.games_per_eval,
        config.max_moves,
    )
}

pub fn run(config: &EvolutionConfig) -> EvolutionResult {
    assert!(config.population_size >= 2, "population trop petite");
    let dim = config.enabled.len();
    let mut rng = Pcg64Mcg::seed_from_u64(config.seed);

    let mut population: Vec<Vec<f64>> = (0..config.population_size)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect();

    let normal = Normal::new(0.0, config.mutation_std).expect("écart-type de mutation invalide");
    let mut history = Vec::with_capacity(config.generations);
    let mut best_overall = population[0].clone();
    let mut best_overall_fitness = f64::MIN;

    for generation in 0..config.generations {
        let fitnesses: Vec<f64> = population
            .par_iter()
            .map(|w| fitness_of(w, &config.enabled, config))
            .collect();

        let mut order: Vec<usize> = (0..population.len()).collect();
        order.sort_by(|&a, &b| fitnesses[b].total_cmp(&fitnesses[a]));

        let best_fitness = fitnesses[order[0]];
        if best_fitness > best_overall_fitness {
            best_overall_fitness = best_fitness;
            best_overall = population[order[0]].clone();
        }
        history.push(LearningPoint {
            steps: generation,
            mean_score: best_fitness,
        });

        let num_elites = ((population.len() as f64 * config.elite_fraction).round() as usize).max(1);
        let elites: Vec<Vec<f64>> = order[..num_elites]
            .iter()
            .map(|&i| population[i].clone())
            .collect();

        let mut next_generation = elites.clone();
        while next_generation.len() < population.len() {
            let parent = &elites[rng.gen_range(0..elites.len())];
            let child: Vec<f64> = parent.iter().map(|&g| g + normal.sample(&mut rng)).collect();
            next_generation.push(child);
        }
        population = next_generation;
    }

    EvolutionResult {
        best_weights: best_overall,
        history,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::Feature;

    #[test]
    fn evolution_improves_or_matches_fitness_over_generations() {
        let config = EvolutionConfig {
            enabled: vec![Feature::EmptyCells, Feature::MaxTileInCorner],
            population_size: 8,
            generations: 3,
            games_per_eval: 4,
            max_moves: 200,
            mutation_std: 0.3,
            elite_fraction: 0.25,
            seed: 1,
        };
        let result = run(&config);
        assert_eq!(result.history.len(), 3);
        assert_eq!(result.best_weights.len(), 2);
        // La meilleure fitness ne peut pas régresser d'une génération à
        // l'autre (les élites sont toujours conservées).
        for w in result.history.windows(2) {
            assert!(w[1].mean_score >= w[0].mean_score - 1e-9);
        }
    }
}
