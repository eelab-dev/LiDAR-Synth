use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

#[derive(Clone, Copy)]
struct Photon {
    time_ps: f64,
    collision_count: i32,
    emmision_delay_ps: f64,
    relaxation_delay_ps: f64,
}

struct PixelResult {
    y: usize,
    x: usize,
    accepted_entries: Vec<Photon>,
    discard: i64,
    accept: i64,
}

fn extract_photon(entry: &Bound<'_, PyAny>) -> PyResult<Photon> {
    Ok(Photon {
        time_ps: entry.get_item("time_ps")?.extract::<f64>()?,
        collision_count: entry.get_item("collision_count")?.extract::<i32>()?,
        emmision_delay_ps: entry.get_item("emmision_delay_ps")?.extract::<f64>()?,
        relaxation_delay_ps: entry.get_item("relaxation_delay_ps")?.extract::<f64>()?,
    })
}

fn extract_pixel_entries(entries: &Bound<'_, PyAny>) -> PyResult<Vec<Photon>> {
    let photon_number = entries.len()?;
    let mut photons = Vec::with_capacity(photon_number);
    for idx in 0..photon_number {
        photons.push(extract_photon(&entries.get_item(idx)?)?);
    }
    Ok(photons)
}

fn extract_pixels(
    pixels: &Bound<'_, PyAny>,
) -> PyResult<(usize, usize, Vec<Vec<Photon>>)> {
    let shape: (usize, usize) = pixels.getattr("shape")?.extract()?;
    let (image_height, image_width) = shape;
    let mut extracted = Vec::with_capacity(image_height * image_width);
    for y in 0..image_height {
        for x in 0..image_width {
            extracted.push(extract_pixel_entries(&pixels.get_item((y, x))?)?);
        }
    }
    Ok((image_height, image_width, extracted))
}

fn process_pixel(
    y: usize,
    x: usize,
    image_width: usize,
    entries: &[Photon],
    pluse_train_num: usize,
    dead_time: f64,
    gap_time: f64,
    active_cruch_time: bool,
    random_seed: Option<i64>,
) -> PixelResult {
    let mut pixel_seed = y * image_width + x;
    if let Some(seed) = random_seed {
        pixel_seed = seed as usize * 1_000_003 + pixel_seed;
    }
    let mut rng = StdRng::seed_from_u64(pixel_seed as u64);

    if entries.is_empty() {
        return PixelResult {
            y,
            x,
            accepted_entries: Vec::new(),
            discard: 0,
            accept: 0,
        };
    }

    let mut photon_records = Vec::with_capacity(entries.len());
    for entry in entries {
        let arrival_time =
            entry.time_ps * 2.0 + entry.emmision_delay_ps + entry.relaxation_delay_ps;
        let pulse_index = rng.gen_range(0..pluse_train_num) as f64;
        let absolute_time = pulse_index * gap_time + arrival_time;
        photon_records.push((absolute_time, *entry));
    }

    photon_records.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut accepted_entries = Vec::new();
    let mut discard = 0_i64;
    let mut accept = 0_i64;
    let mut last_accepted_time: Option<f64> = None;

    for (absolute_time, entry) in photon_records {
        if let Some(last_time) = last_accepted_time {
            if absolute_time - last_time < dead_time {
                discard += 1;
                if !active_cruch_time {
                    last_accepted_time = Some(absolute_time);
                }
                continue;
            }
        }

        accepted_entries.push(entry);
        accept += 1;
        last_accepted_time = Some(absolute_time);
    }

    accepted_entries.sort_by(|a, b| a.time_ps.total_cmp(&b.time_ps));

    PixelResult {
        y,
        x,
        accepted_entries,
        discard,
        accept,
    }
}

fn build_photon_dtype<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    let np = PyModule::import_bound(py, "numpy")?;
    np.getattr("dtype")?.call1((
        vec![
            ("time_ps", "float32"),
            ("collision_count", "int32"),
            ("emmision_delay_ps", "float32"),
            ("relaxation_delay_ps", "float32"),
        ],
    ))
}

fn build_pixel_output<'py>(
    py: Python<'py>,
    entries: &[Photon],
) -> PyResult<PyObject> {
    let np = PyModule::import_bound(py, "numpy")?;
    let dtype = build_photon_dtype(py)?;
    let pixel_output = np.getattr("empty")?.call1((entries.len(), dtype))?;

    let time_values = entries
        .iter()
        .map(|entry| entry.time_ps as f32)
        .collect::<Vec<_>>();
    let collision_values = entries
        .iter()
        .map(|entry| entry.collision_count)
        .collect::<Vec<_>>();
    let jitter_values = entries
        .iter()
        .map(|entry| entry.emmision_delay_ps as f32)
        .collect::<Vec<_>>();
    let relaxation_values = entries
        .iter()
        .map(|entry| entry.relaxation_delay_ps as f32)
        .collect::<Vec<_>>();

    pixel_output.set_item("time_ps", np.getattr("asarray")?.call1((time_values,))?)?;
    pixel_output.set_item(
        "collision_count",
        np.getattr("asarray")?.call1((collision_values,))?,
    )?;
    pixel_output.set_item(
        "emmision_delay_ps",
        np.getattr("asarray")?.call1((jitter_values,))?,
    )?;
    pixel_output.set_item(
        "relaxation_delay_ps",
        np.getattr("asarray")?.call1((relaxation_values,))?,
    )?;

    Ok(pixel_output.into_any().unbind())
}

#[pyfunction(signature = (pluse_train_num=50000, dead_time=13000.0, input_pixelized_timeStamp=None, gap_time=1000000.0, max_workers=None, active_cruch_time=true, random_seed=None))]
pub fn dead_time_simulation(
    py: Python<'_>,
    pluse_train_num: usize,
    dead_time: f64,
    input_pixelized_timeStamp: Option<PyObject>,
    gap_time: f64,
    max_workers: Option<usize>,
    active_cruch_time: bool,
    random_seed: Option<i64>,
) -> PyResult<(PyObject, i64, i64)> {
    let Some(input_pixelized_timeStamp) = input_pixelized_timeStamp else {
        return Err(PyValueError::new_err("simulated_photon must be provided"));
    };

    let pixels = input_pixelized_timeStamp.bind(py);
    let (image_height, image_width, extracted_pixels) = extract_pixels(pixels)?;

    let worker_count = max_workers.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|n| n.get().min(32))
            .unwrap_or(1)
    });

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(worker_count.max(1))
        .build()
        .map_err(|err| PyValueError::new_err(err.to_string()))?;

    let pixel_results = py.allow_threads(move || {
        pool.install(|| {
            extracted_pixels
                .par_iter()
                .enumerate()
                .map(|(index, entries)| {
                    let y = index / image_width;
                    let x = index % image_width;
                    process_pixel(
                        y,
                        x,
                        image_width,
                        entries,
                        pluse_train_num,
                        dead_time,
                        gap_time,
                        active_cruch_time,
                        random_seed,
                    )
                })
                .collect::<Vec<_>>()
        })
    });

    let np = PyModule::import_bound(py, "numpy")?;
    let object_dtype = np.getattr("dtype")?.call1(("O",))?;
    let pixel_output_array = np
        .getattr("empty")?
        .call1(((image_height, image_width), object_dtype))?;

    let mut discard_count = 0_i64;
    let mut accept_count = 0_i64;
    for pixel_result in pixel_results {
        pixel_output_array.set_item(
            (pixel_result.y, pixel_result.x),
            build_pixel_output(py, &pixel_result.accepted_entries)?,
        )?;
        discard_count += pixel_result.discard;
        accept_count += pixel_result.accept;
    }

    Ok((pixel_output_array.into_any().unbind(), discard_count, accept_count))
}
