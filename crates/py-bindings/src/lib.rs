use g2048_core::runner::{direction_to_u8, u8_to_direction};
use g2048_core::GameState;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyfunction]
fn hello() -> PyResult<String> {
    Ok(g2048_core::hello())
}

fn result_to_dict<'py>(py: Python<'py>, g: g2048_core::runner::GameResult) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("seed", g.seed)?;
    dict.set_item("score", g.score)?;
    dict.set_item("max_tile", g.max_tile)?;
    dict.set_item("num_moves", g.num_moves)?;
    dict.set_item("duration_ms", g.duration_ms)?;
    dict.set_item("moves", g.moves)?;
    Ok(dict)
}

/// API de haut niveau (chapitre 3) : exécute un lot de parties en parallèle
/// à partir d'une config TOML et retourne une liste de résultats.
#[pyfunction]
fn run_batch(py: Python<'_>, toml_text: &str) -> PyResult<Vec<PyObject>> {
    let config = g2048_core::runner::parse_config(toml_text)
        .map_err(|e| PyValueError::new_err(format!("config TOML invalide: {e}")))?;
    let results = g2048_core::runner::run_batch(&config).map_err(PyValueError::new_err)?;
    results
        .into_iter()
        .map(|g| result_to_dict(py, g).map(|d| d.into()))
        .collect()
}

/// Rejoue une partie à partir de sa seed et de sa séquence de coups, pour
/// vérifier la reproductibilité d'une expérience.
#[pyfunction]
fn replay(py: Python<'_>, seed: u64, moves: Vec<u8>) -> PyResult<PyObject> {
    let g = g2048_core::runner::replay(seed, &moves).map_err(PyValueError::new_err)?;
    result_to_dict(py, g).map(|d| d.into())
}

/// API coup par coup type Gymnasium, plus lente mais utile pour le
/// débogage et le deep RL (chapitre 8).
#[pyclass]
struct Env {
    state: GameState,
}

#[pymethods]
impl Env {
    #[new]
    fn new(seed: u64) -> Self {
        Env {
            state: GameState::new(seed),
        }
    }

    fn reset(&mut self, seed: u64) -> Vec<u32> {
        self.state = GameState::new(seed);
        self.observation()
    }

    /// Retourne (observation, reward, terminated, truncated, info).
    fn step(&mut self, py: Python<'_>, action: u8) -> PyResult<(Vec<u32>, f64, bool, bool, PyObject)> {
        let dir = u8_to_direction(action);
        let prev_score = self.state.score();
        let moved = self.state.apply_move(dir);
        if moved {
            self.state.spawn();
        }
        let reward = (self.state.score() - prev_score) as f64;
        let terminated = self.state.is_terminal();
        let info = PyDict::new_bound(py);
        info.set_item("invalid_move", !moved)?;
        Ok((self.observation(), reward, terminated, false, info.into()))
    }

    fn legal_moves(&self) -> Vec<u8> {
        self.state
            .legal_moves()
            .into_iter()
            .map(direction_to_u8)
            .collect()
    }

    fn observation(&self) -> Vec<u32> {
        let board = self.state.board();
        (0..16).map(|i| ((board >> (i * 4)) & 0xF) as u32).collect()
    }

    fn score(&self) -> u64 {
        self.state.score()
    }

    /// Vecteur de features (chapitre 5) du plateau courant, pour
    /// l'entrée "features" du deep RL (chapitre 8) plutôt que la grille
    /// brute one-hot.
    fn features(&self, enabled: Vec<String>) -> PyResult<Vec<f64>> {
        let enabled = parse_features(enabled)?;
        Ok(g2048_core::features::compute(self.state.board(), &enabled))
    }
}

fn parse_features(names: Vec<String>) -> PyResult<Vec<g2048_core::features::Feature>> {
    names
        .into_iter()
        .map(|n| {
            g2048_core::features::Feature::from_name(&n)
                .ok_or_else(|| PyValueError::new_err(format!("feature inconnue: '{n}'")))
        })
        .collect()
}

fn history_to_pylist(
    py: Python<'_>,
    history: Vec<g2048_core::train::LearningPoint>,
) -> PyResult<Vec<PyObject>> {
    history
        .into_iter()
        .map(|p| {
            let d = PyDict::new_bound(py);
            d.set_item("steps", p.steps)?;
            d.set_item("mean_score", p.mean_score)?;
            Ok(d.into())
        })
        .collect()
}

/// 6a. Optimisation évolutionnaire : `enabled` liste les noms de features
/// (chapitre 5) à activer, ex. `["empty_cells", "monotonicity"]`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn train_evolution(
    py: Python<'_>,
    enabled: Vec<String>,
    population_size: usize,
    generations: usize,
    games_per_eval: usize,
    max_moves: usize,
    mutation_std: f64,
    elite_fraction: f64,
    seed: u64,
) -> PyResult<PyObject> {
    let enabled = parse_features(enabled)?;
    let config = g2048_core::train::evolution::EvolutionConfig {
        enabled,
        population_size,
        generations,
        games_per_eval,
        max_moves,
        mutation_std,
        elite_fraction,
        seed,
    };
    let result = g2048_core::train::evolution::run(&config);
    let dict = PyDict::new_bound(py);
    dict.set_item("best_weights", result.best_weights)?;
    dict.set_item("history", history_to_pylist(py, result.history)?)?;
    Ok(dict.into())
}

/// 6b. TD(0) sur afterstate avec les features du chapitre 5.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn train_td_features(
    py: Python<'_>,
    enabled: Vec<String>,
    games: usize,
    max_moves: usize,
    alpha: f64,
    seed: u64,
    eval_every: usize,
    eval_games: usize,
) -> PyResult<PyObject> {
    let enabled = parse_features(enabled)?;
    let model = g2048_core::train::td::FeatureModel { enabled };
    let config = g2048_core::train::td::TdConfig {
        games,
        max_moves,
        alpha,
        seed,
        eval_every,
        eval_games,
    };
    let (weights, history) = g2048_core::train::td::run(&model, &config);
    let dict = PyDict::new_bound(py);
    dict.set_item("weights", weights)?;
    dict.set_item("history", history_to_pylist(py, history)?)?;
    Ok(dict.into())
}

/// 6c. TD(0) sur afterstate avec le témoin n-tuple (grille brute, sans
/// connaissance du jeu).
#[pyfunction]
fn train_td_ntuple(
    py: Python<'_>,
    games: usize,
    max_moves: usize,
    alpha: f64,
    seed: u64,
    eval_every: usize,
    eval_games: usize,
) -> PyResult<PyObject> {
    let model = g2048_core::train::td::NTupleModel;
    let config = g2048_core::train::td::TdConfig {
        games,
        max_moves,
        alpha,
        seed,
        eval_every,
        eval_games,
    };
    let (weights, history) = g2048_core::train::td::run(&model, &config);
    let dict = PyDict::new_bound(py);
    dict.set_item("weights", weights)?;
    dict.set_item("history", history_to_pylist(py, history)?)?;
    Ok(dict.into())
}

/// Score individuel de chaque partie jouée par un agent à profondeur 1
/// avec ces poids de features, utile pour calculer un intervalle de
/// confiance côté analyse Python. `seed_offset` permet un échantillon de
/// test disjoint de l'entraînement.
#[pyfunction]
fn scores_of_feature_weights(
    enabled: Vec<String>,
    weights: Vec<f64>,
    num_games: usize,
    max_moves: usize,
    seed_offset: u64,
) -> PyResult<Vec<u64>> {
    let enabled = parse_features(enabled)?;
    Ok(g2048_core::train::scores_of(
        move |board| g2048_core::features::evaluate(board, &enabled, &weights),
        num_games,
        max_moves,
        seed_offset,
    ))
}

/// Équivalent de [`scores_of_feature_weights`] pour le témoin n-tuple.
#[pyfunction]
fn scores_of_ntuple_weights(
    weights: Vec<f64>,
    num_games: usize,
    max_moves: usize,
    seed_offset: u64,
) -> Vec<u64> {
    g2048_core::train::scores_of(
        move |board| g2048_core::train::ntuple::value(board, &weights),
        num_games,
        max_moves,
        seed_offset,
    )
}

/// Chapitre 7 : joue un lot de parties avec un agent expectimax (poids de
/// features du chapitre 5/6) et retourne, par partie, score/tuile max/
/// nombre de coups/durée — pour tracer performance et temps par coup en
/// fonction de la profondeur de recherche.
#[pyfunction]
fn run_expectimax(
    py: Python<'_>,
    enabled: Vec<String>,
    weights: Vec<f64>,
    depth: u8,
    num_games: usize,
    max_moves: usize,
    seed_offset: u64,
) -> PyResult<Vec<PyObject>> {
    let enabled = parse_features(enabled)?;
    let results =
        g2048_core::search::run_batch(&enabled, &weights, depth, num_games, max_moves, seed_offset);
    results
        .into_iter()
        .map(|r| {
            let d = PyDict::new_bound(py);
            d.set_item("seed", r.seed)?;
            d.set_item("score", r.score)?;
            d.set_item("max_tile", r.max_tile)?;
            d.set_item("num_moves", r.num_moves)?;
            d.set_item("duration_ms", r.duration_ms)?;
            Ok(d.into())
        })
        .collect()
}

#[pymodule]
fn _g2048(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello, m)?)?;
    m.add_function(wrap_pyfunction!(run_batch, m)?)?;
    m.add_function(wrap_pyfunction!(replay, m)?)?;
    m.add_function(wrap_pyfunction!(train_evolution, m)?)?;
    m.add_function(wrap_pyfunction!(train_td_features, m)?)?;
    m.add_function(wrap_pyfunction!(train_td_ntuple, m)?)?;
    m.add_function(wrap_pyfunction!(scores_of_feature_weights, m)?)?;
    m.add_function(wrap_pyfunction!(scores_of_ntuple_weights, m)?)?;
    m.add_function(wrap_pyfunction!(run_expectimax, m)?)?;
    m.add_class::<Env>()?;
    Ok(())
}
