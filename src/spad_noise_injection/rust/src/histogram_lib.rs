use numpy::PyArray3;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
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
    illegal_entries: Vec<(f64, i32)>,
    stamped_pixel: Vec<i64>,
    error_pixel: Vec<i64>,
    multi_path_pixel: Vec<i64>,
    true_pixel: Vec<i64>,
}

fn compute_bin_edges_native(
    range_min: f64,
    range_max: f64,
    bin_number: usize,
    dnl_row: &[f64],
) -> Vec<f64> {
    let ideal_width = (range_max - range_min) / bin_number as f64;
    let mut edges = vec![0.0_f64; bin_number + 1];
    edges[0] = range_min;
    let mut cumulative = range_min;
    for (index, dnl) in dnl_row.iter().enumerate() {
        cumulative += ideal_width * (1.0 + *dnl);
        edges[index + 1] = cumulative;
    }
    edges
}

fn searchsorted(edges: &[f64], distance: f64, bin_number: usize) -> usize {
    let raw_index = edges.partition_point(|edge| *edge <= distance);
    raw_index.saturating_sub(1).min(bin_number.saturating_sub(1))
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
    let (image_width, image_height) = shape;
    let mut extracted = Vec::with_capacity(image_width * image_height);
    for y in 0..image_width {
        for x in 0..image_height {
            extracted.push(extract_pixel_entries(&pixels.get_item((y, x))?)?);
        }
    }
    Ok((image_width, image_height, extracted))
}

fn extract_dnl_matrix(
    dnl_matrix: Option<&Bound<'_, PyAny>>,
    image_width: usize,
    image_height: usize,
    bin_number: usize,
) -> PyResult<Option<Vec<Vec<f64>>>> {
    let Some(dnl_matrix) = dnl_matrix else {
        return Ok(None);
    };

    let shape: (usize, usize, usize) = dnl_matrix.getattr("shape")?.extract()?;
    if shape != (image_width, image_height, bin_number) {
        return Err(PyValueError::new_err(format!(
            "dnl_matrix shape mismatch: expected ({}, {}, {}), got {:?}",
            image_width, image_height, bin_number, shape
        )));
    }

    let mut extracted = Vec::with_capacity(image_width * image_height);
    for y in 0..image_width {
        for x in 0..image_height {
            let mut dnl_row = Vec::with_capacity(bin_number);
            for bin_index in 0..bin_number {
                dnl_row.push(dnl_matrix.get_item((y, x, bin_index))?.extract::<f64>()?);
            }
            extracted.push(dnl_row);
        }
    }
    Ok(Some(extracted))
}

fn process_pixel(
    y: usize,
    x: usize,
    entries: &[Photon],
    bin_number: usize,
    range_min: f64,
    range_max: f64,
    ideal_edges: Option<&[f64]>,
    dnl_row: Option<&[f64]>,
) -> PixelResult {
    let edges = match dnl_row {
        Some(row) => compute_bin_edges_native(range_min, range_max, bin_number, row),
        None => ideal_edges.unwrap().to_vec(),
    };

    let mut illegal_entries = Vec::new();
    let mut stamped_pixel = vec![0_i64; bin_number];
    let mut error_pixel = vec![0_i64; bin_number];
    let mut multi_path_pixel = vec![0_i64; bin_number];
    let mut true_pixel = vec![0_i64; bin_number];

    for entry in entries {
        let distance =
            entry.time_ps + entry.emmision_delay_ps / 2.0 + entry.relaxation_delay_ps / 2.0;
        let collosion = entry.collision_count;

        if range_min > distance || range_max < distance {
            illegal_entries.push((distance, collosion));
            continue;
        }

        let bin_index = searchsorted(&edges, distance, bin_number);
        stamped_pixel[bin_index] += 1;

        if collosion < 0 {
            error_pixel[bin_index] += 1;
        }
        if collosion > 2 {
            multi_path_pixel[bin_index] += 1;
        }
        if collosion == 2 {
            true_pixel[bin_index] += 1;
        }
    }

    PixelResult {
        y,
        x,
        illegal_entries,
        stamped_pixel,
        error_pixel,
        multi_path_pixel,
        true_pixel,
    }
}

#[pyfunction(signature = (pixels, bin_number=25, range_distance=vec![1000.0, 2500.0], peak_func=None, dnl_matrix=None, num_workers=64))]
pub fn form_histogram_image(
    py: Python<'_>,
    pixels: PyObject,
    bin_number: usize,
    range_distance: Vec<f64>,
    peak_func: Option<PyObject>,
    dnl_matrix: Option<PyObject>,
    num_workers: usize,
) -> PyResult<(PyObject, PyObject, PyObject)> {
    let _ = peak_func;
    let pixels = pixels.bind(py);
    if range_distance.len() < 2 {
        return Err(PyValueError::new_err(
            "range_distance must contain at least [min, max]",
        ));
    }

    let (image_width, image_height, extracted_pixels) = extract_pixels(pixels)?;
    let extracted_dnl = extract_dnl_matrix(
        dnl_matrix.as_ref().map(|matrix| matrix.bind(py)),
        image_width,
        image_height,
        bin_number,
    )?;

    let range_min = range_distance[0];
    let range_max = range_distance[1];
    let ideal_edges = if extracted_dnl.is_none() {
        Some(
            (0..=bin_number)
                .map(|index| {
                    range_min + (range_max - range_min) * index as f64 / bin_number as f64
                })
                .collect::<Vec<_>>(),
        )
    } else {
        None
    };

    let worker_count = if num_workers == 0 { 1 } else { num_workers };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(worker_count)
        .build()
        .map_err(|err| PyValueError::new_err(err.to_string()))?;

    let pixel_results = py.allow_threads(move || {
        pool.install(|| {
            extracted_pixels
                .par_iter()
                .enumerate()
                .map(|(index, entries)| {
                    let y = index / image_height;
                    let x = index % image_height;
                    process_pixel(
                        y,
                        x,
                        entries,
                        bin_number,
                        range_min,
                        range_max,
                        ideal_edges.as_deref(),
                        extracted_dnl
                            .as_ref()
                            .map(|matrix| matrix[index].as_slice()),
                    )
                })
                .collect::<Vec<_>>()
        })
    });

    let np = PyModule::import_bound(py, "numpy")?;
    let object_dtype = np.getattr("dtype")?.call1(("O",))?;
    let illegal_photon = np.getattr("empty")?.call1(((image_width, image_height), object_dtype))?;
    for y in 0..image_width {
        for x in 0..image_height {
            illegal_photon.set_item((y, x), PyList::empty_bound(py))?;
        }
    }

    let total_len = image_width * image_height * bin_number;
    let mut stamped_histogram = vec![0_i64; total_len];
    let mut error_histogram = vec![0_i64; total_len];
    let mut multi_path_histogram = vec![0_i64; total_len];
    let mut true_histogram = vec![0_i64; total_len];

    let offset = |y: usize, x: usize, bin_index: usize| -> usize {
        (y * image_height + x) * bin_number + bin_index
    };

    for pixel_result in pixel_results {
        let illegal_list = PyList::empty_bound(py);
        for (distance, collosion) in pixel_result.illegal_entries {
            illegal_list.append((distance, collosion))?;
        }
        illegal_photon.set_item((pixel_result.y, pixel_result.x), illegal_list)?;

        for bin_index in 0..bin_number {
            let idx = offset(pixel_result.y, pixel_result.x, bin_index);
            stamped_histogram[idx] = pixel_result.stamped_pixel[bin_index];
            error_histogram[idx] = pixel_result.error_pixel[bin_index];
            multi_path_histogram[idx] = pixel_result.multi_path_pixel[bin_index];
            true_histogram[idx] = pixel_result.true_pixel[bin_index];
        }
    }

    let stamped_histogram = PyArray3::from_vec3_bound(
        py,
        &(0..image_width)
            .map(|y| {
                (0..image_height)
                    .map(|x| {
                        (0..bin_number)
                            .map(|bin_index| stamped_histogram[offset(y, x, bin_index)])
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>(),
    )?;
    let error_histogram = PyArray3::from_vec3_bound(
        py,
        &(0..image_width)
            .map(|y| {
                (0..image_height)
                    .map(|x| {
                        (0..bin_number)
                            .map(|bin_index| error_histogram[offset(y, x, bin_index)])
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>(),
    )?;
    let multi_path_histogram = PyArray3::from_vec3_bound(
        py,
        &(0..image_width)
            .map(|y| {
                (0..image_height)
                    .map(|x| {
                        (0..bin_number)
                            .map(|bin_index| multi_path_histogram[offset(y, x, bin_index)])
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>(),
    )?;
    let true_histogram = PyArray3::from_vec3_bound(
        py,
        &(0..image_width)
            .map(|y| {
                (0..image_height)
                    .map(|x| {
                        (0..bin_number)
                            .map(|bin_index| true_histogram[offset(y, x, bin_index)])
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>(),
    )?;

    let illegal_information = PyTuple::new_bound(
        py,
        [
            illegal_photon.into_any().unbind(),
            error_histogram.into_any().unbind(),
            multi_path_histogram.into_any().unbind(),
        ],
    );

    Ok((
        illegal_information.into_any().unbind(),
        stamped_histogram.into_any().unbind(),
        true_histogram.into_any().unbind(),
    ))
}
