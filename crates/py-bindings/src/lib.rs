use pyo3::prelude::*;

#[pyfunction]
fn hello() -> PyResult<String> {
    Ok(g2048_core::hello())
}

#[pymodule]
fn _g2048(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello, m)?)?;
    Ok(())
}
