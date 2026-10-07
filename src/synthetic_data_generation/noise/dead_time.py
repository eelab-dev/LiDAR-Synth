import numpy as np
import random
import importlib
import warnings
from concurrent.futures import ThreadPoolExecutor
import os

from .timestamps import (
    PHOTON_DTYPE,
    extract_jitter_time_ps,
    extract_relaxation_time_ps,
    extract_time_ps,
    is_empty_pixel,
)

try:
    from _synthetic_sensor import dead_time_simulation as rust_dead_time_simulation
except ImportError:
    rust_dead_time_simulation = None

_RUST_DEAD_TIME_WARNING_EMITTED = False


def _load_rust_dead_time_simulation():
    global rust_dead_time_simulation

    if rust_dead_time_simulation is not None:
        return rust_dead_time_simulation

    import_errors = []
    module_specs = (
        ("_synthetic_sensor", "dead_time_simulation"),
    )

    for module_name, attr_name in module_specs:
        try:
            module = importlib.import_module(module_name)
            rust_dead_time_simulation = getattr(module, attr_name)
            return rust_dead_time_simulation
        except Exception as exc:
            import_errors.append(f"{module_name}: {exc}")

    raise ImportError(
        "_synthetic_sensor is not available. Tried imports: "
        + " | ".join(import_errors)
    )



def _build_pixel_output(entries, accepted_entries):
    entries_array = np.asarray(entries)

    if entries_array.dtype.names:
        if len(accepted_entries) == 0:
            return np.empty(0, dtype=entries_array.dtype)

        pixel_output = np.array(accepted_entries, dtype=entries_array.dtype)
    else:
        if len(accepted_entries) == 0:
            return np.empty(0, dtype=PHOTON_DTYPE)

        pixel_output = np.empty(len(accepted_entries), dtype=PHOTON_DTYPE)
        pixel_output["time_ps"] = np.asarray(
            [extract_time_ps(entry) for entry in accepted_entries],
            dtype=np.float32,
        )
        pixel_output["collision_count"] = np.asarray(
            [entry[1] for entry in accepted_entries],
            dtype=np.int32,
        )
        pixel_output["emmision_delay_ps"] = np.asarray(
            [extract_jitter_time_ps(entry) for entry in accepted_entries],
            dtype=np.float32,
        )
        pixel_output["relaxation_delay_ps"] = np.asarray(
            [extract_relaxation_time_ps(entry) for entry in accepted_entries],
            dtype=np.float32,
        )

    if len(pixel_output) > 0:
        order = np.argsort(pixel_output["time_ps"])
        pixel_output = pixel_output[order]

    return pixel_output


def dead_time_simulation(
    pluse_train_num = 50000,
    dead_time=13000,
    input_pixelized_timeStamp = None,
    gap_time = 1000000,
    max_workers = None,
    active_cruch_time = True,
    random_seed: int | None = None):

    if input_pixelized_timeStamp is None:
        raise ValueError("simulated_photon must be provided")

    image_height, image_width = input_pixelized_timeStamp.shape

    def process_pixel(y, x, entries):
        pixel_seed = y * image_width + x
        if random_seed is not None:
            pixel_seed = int(random_seed) * 1_000_003 + pixel_seed
        rng = random.Random(pixel_seed)
        accepted_entries = []
        discard = 0
        accept = 0

        if is_empty_pixel(entries):
            return (y, x, _build_pixel_output(entries, []), discard, accept)

        photon_records = []

        # Build absolute photon time directly from the per-photon time in ps.
        for entry in entries:
            arrival_time = extract_time_ps(entry) * 2
            jitter_time = extract_jitter_time_ps(entry)
            relaxation_time = extract_relaxation_time_ps(entry)
            arrival_time = arrival_time + jitter_time + relaxation_time
            pulse_index = rng.randrange(1,pluse_train_num,1)
            absolute_time = pulse_index * gap_time + arrival_time
            photon_records.append((absolute_time,  arrival_time, entry))

        # Sort by absolute time
        photon_records.sort(key=lambda x: x[0])

        last_accepted_time = None

        for absolute_time, arrival_time, entry in photon_records:
            if last_accepted_time is None:
                accepted_entries.append(entry)
                accept += 1
                last_accepted_time = absolute_time
            elif absolute_time - last_accepted_time < dead_time:
                discard += 1
                if active_cruch_time == False:
                    last_accepted_time = absolute_time
            else:
                accepted_entries.append(entry)
                accept += 1
                last_accepted_time = absolute_time

        return (y, x, _build_pixel_output(entries, accepted_entries), discard, accept)

    tasks = []
    for y, row in enumerate(input_pixelized_timeStamp):
        for x, entries in enumerate(row):
            tasks.append((y, x, entries))

    if max_workers is None:
        max_workers = min(32, os.cpu_count() or 1)

    pixel_output_array = np.empty((image_height, image_width), dtype=object)
    discard_count = 0
    accept_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for y, x, out, d, a in ex.map(lambda args: process_pixel(*args), tasks):
            pixel_output_array[y][x] = out
            discard_count += d
            accept_count += a

    return pixel_output_array, discard_count, accept_count


def dead_time_simulation_rust(
    pluse_train_num=50000,
    dead_time=13000,
    input_pixelized_timeStamp=None,
    gap_time=1000000,
    max_workers=None,
    active_cruch_time=True,
    random_seed: int | None = None,
):
    global _RUST_DEAD_TIME_WARNING_EMITTED

    try:
        rust_impl = _load_rust_dead_time_simulation()
    except ImportError as exc:
        if not _RUST_DEAD_TIME_WARNING_EMITTED:
            warnings.warn(
                "Rust dead-time simulation is unavailable; falling back to the "
                f"Python implementation. Details: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            _RUST_DEAD_TIME_WARNING_EMITTED = True
        return dead_time_simulation(
            pluse_train_num=pluse_train_num,
            dead_time=dead_time,
            input_pixelized_timeStamp=input_pixelized_timeStamp,
            gap_time=gap_time,
            max_workers=max_workers,
            active_cruch_time=active_cruch_time,
            random_seed=random_seed,
        )

    return rust_impl(
        pluse_train_num=pluse_train_num,
        dead_time=dead_time,
        input_pixelized_timeStamp=input_pixelized_timeStamp,
        gap_time=gap_time,
        max_workers=max_workers,
        active_cruch_time=active_cruch_time,
        random_seed=random_seed,
    )
