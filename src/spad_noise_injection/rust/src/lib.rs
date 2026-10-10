mod dead_time_lib;
mod histogram_lib;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

#[pymodule]
fn _synthetic_sensor(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(histogram_lib::form_histogram_image, m)?)?;
    m.add_function(wrap_pyfunction!(dead_time_lib::dead_time_simulation, m)?)?;
    Ok(())
}
