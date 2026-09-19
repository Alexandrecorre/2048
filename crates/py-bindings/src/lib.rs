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
}

#[pymodule]
fn _g2048(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello, m)?)?;
    m.add_function(wrap_pyfunction!(run_batch, m)?)?;
    m.add_function(wrap_pyfunction!(replay, m)?)?;
    m.add_class::<Env>()?;
    Ok(())
}
