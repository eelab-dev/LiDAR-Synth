from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np

from spad_noise_injection.ambient import create_ambient_timeStamp
from spad_noise_injection.dark_count import (
    add_darkCount_spatial_ratio_map,
    add_dark_histogram,
    read_darkcout_histogram,
    read_spatial_ratio_map,
)
from spad_noise_injection.dead_time import dead_time_simulation_rust
from spad_noise_injection.depth import form_depth_from_histogram, save_depth_image_png
from spad_noise_injection.diffuse_map import add_percentage_noise, read_diffuse_from_h5
from spad_noise_injection.dnl import read_dnl_matrix
from spad_noise_injection.histogram import form_histogram_image_rust, save_histogram_to_h5
from spad_noise_injection.relaxation_delay import create_relaxation_delay
from spad_noise_injection.time_jitter import create_emmision_jitter
from spad_noise_injection.timestamps import (
    get_timeStamp_count_sum,
    merge_pixelized_timeStamp,
    read_pixllized_timeStamp_from_h5,
)


def _add_option(parser: argparse.ArgumentParser, old_name: str, **kwargs):
    aliases = [old_name.replace("_", "-"), old_name]
    parser.add_argument(*dict.fromkeys(aliases), dest=old_name.lstrip("-").replace("-", "_"), **kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="synthetic-data inject-noise", description="Inject sensor noise into pixelized photon data.")
    _add_option(parser, "--input_pixelized_file", help="Path to pixelized photon .h5 file")
    _add_option(parser, "--input_diffuse_file", help="Path to diffuse map .h5 file")
    _add_option(parser, "--input_dnl_file", help="Optional path to DNL matrix .h5 file", default=None)
    _add_option(
        parser,
        "--input_dark_histogram_file",
        help="Optional path to a measured dark-count histogram .h5 file",
        default=None,
    )
    _add_option(
        parser,
        "--input_spatial_ratio_map",
        help="Optional path to a spatial_ratio_map .h5/.npy file",
        default=None,
    )
    _add_option(parser, "--dark_count_scale", type=float, default=0.0)
    _add_option(parser, "--spatial_jitter_sigma", type=float, default=0.08)
    _add_option(parser, "--diffuse_noise_level", type=float)
    _add_option(parser, "--time_jitter_sigma", type=float)
    _add_option(parser, "--relaxation_tau", type=float, default=0.0)
    parser.add_argument("--emission-mean", "--emmission-mean", "--emmission_mean", dest="emmission_mean", type=int, help="Mean emission delay in ps")
    parser.add_argument("--seed", "--seeds", dest="seeds", type=int, help="Seed for the six sensor noise streams")
    parser.add_argument("-o", "--output_file", "--output-file", dest="output_file")
    _add_option(parser, "--ambient_level", type=float)
    parser.add_argument("--pulse-train-num", "--pluse-train-num", "--pluse_train_num", dest="pluse_train_num", type=int, help="Number of pulses for dead-time simulation")
    _add_option(parser, "--dead_time", type=float, default=13000.0)
    _add_option(parser, "--worker_threads", type=int, default=None)
    _add_option(parser, "--min_range", type=float)
    _add_option(parser, "--bin_width", type=float)
    _add_option(parser, "--bin_number", type=int)
    _add_option(parser, "--output_histogram_file", help="Path to output histogram .h5 file")
    _add_option(parser, "--output_depth_image_path", help="Optional depth image PNG path")
    return parser


def _optional_path(value):
    if value in (None, "", "None", "none"):
        return None
    return Path(value)


def main(argv: list[str] | None = None) -> int:
    start_time = perf_counter()
    parser = build_parser()
    args = parser.parse_args(argv)

    required_args = {
        "input_pixelized_file": args.input_pixelized_file,
        "input_diffuse_file": args.input_diffuse_file,
        "diffuse_noise_level": args.diffuse_noise_level,
        "time_jitter_sigma": args.time_jitter_sigma,
        "seeds": args.seeds,
        "output_file": args.output_file,
        "ambient_level": args.ambient_level,
        "pluse_train_num": args.pluse_train_num,
        "min_range": args.min_range,
        "bin_width": args.bin_width,
        "bin_number": args.bin_number,
        "output_histogram_file": args.output_histogram_file,
    }
    missing_args = [f"--{name}" for name, value in required_args.items() if value is None]
    if missing_args:
        parser.error("Missing required input(s): " + ", ".join(missing_args))

    input_files = {
        "--input_pixelized_file": Path(args.input_pixelized_file),
        "--input_diffuse_file": Path(args.input_diffuse_file),
    }
    input_dnl_file = _optional_path(args.input_dnl_file)
    if input_dnl_file is not None:
        input_files["--input_dnl_file"] = input_dnl_file

    missing_files = [name for name, path in input_files.items() if not path.is_file()]
    if missing_files:
        parser.error("Input file(s) not found for: " + ", ".join(missing_files))

    if args.diffuse_noise_level < 0:
        parser.error("--diffuse_noise_level must be non-negative")
    if args.time_jitter_sigma < 0:
        parser.error("--time_jitter_sigma must be non-negative")
    if args.relaxation_tau < 0:
        parser.error("--relaxation_tau must be non-negative")
    if args.ambient_level < 0:
        parser.error("--ambient_level must be non-negative")
    if args.pluse_train_num <= 0:
        parser.error("--pluse_train_num must be greater than 0")
    if args.dead_time <= 0:
        parser.error("--dead_time must be greater than 0")
    if args.worker_threads is not None and args.worker_threads <= 0:
        parser.error("--worker_threads must be greater than 0")
    if args.dark_count_scale < 0:
        parser.error("--dark_count_scale must be non-negative")
    if args.spatial_jitter_sigma < 0:
        parser.error("--spatial_jitter_sigma must be non-negative")
    if args.bin_width <= 0:
        parser.error("--bin_width must be greater than 0")
    if args.bin_number <= 0:
        parser.error("--bin_number must be greater than 0")
    if args.emmission_mean < 0:
        parser.error("--emisson_mean must be greate than 0")

    seed_sequence = np.random.SeedSequence(args.seeds)
    stage_seed_sequences = seed_sequence.spawn(6)
    (
        diffuse_seed,
        jitter_seed,
        relaxation_seed,
        ambient_seed,
        dead_time_seed,
        dark_count_seed,
    ) = [int(stage_seed.generate_state(1, dtype=np.uint32)[0]) for stage_seed in stage_seed_sequences]

    input_dark_histogram_file = _optional_path(args.input_dark_histogram_file)
    if input_dark_histogram_file is not None and not input_dark_histogram_file.is_file():
        parser.error(f"Input dark-count histogram not found: {input_dark_histogram_file}")
    if input_dark_histogram_file is None and args.dark_count_scale != 0:
        parser.error("--input_dark_histogram_file is required when --dark_count_scale is non-zero")

    input_spatial_ratio_map_file = _optional_path(args.input_spatial_ratio_map)
    if input_spatial_ratio_map_file is not None and not input_spatial_ratio_map_file.is_file():
        parser.error(f"Input spatial ratio map not found: {input_spatial_ratio_map_file}")

    print(args.input_pixelized_file)
    input_stamped_num = get_timeStamp_count_sum(pixelized_timestamp=args.input_pixelized_file)
    input_pixelized_timestamp = read_pixllized_timeStamp_from_h5(args.input_pixelized_file)

    diffuse_input = read_diffuse_from_h5(args.input_diffuse_file)
    diffuse_input = add_percentage_noise(args.diffuse_noise_level, diffuse_input, random_seed=diffuse_seed)

    input_pixelized_timestamp = create_emmision_jitter(
        time_stamp=input_pixelized_timestamp,
        seeds_number=jitter_seed,
        jitter_sigma=args.time_jitter_sigma,
        emission_mean=args.emmission_mean,
    )
    input_pixelized_timestamp = create_relaxation_delay(
        time_stamp=input_pixelized_timestamp,
        seeds_number=relaxation_seed,
        relaxation_tau=args.relaxation_tau,
    )

    input_dnl_matrix = read_dnl_matrix(input_dnl_file) if input_dnl_file is not None else None
    input_min_range = args.min_range
    input_bin_num = args.bin_number
    input_bin_width = args.bin_width
    input_max_range = input_min_range + input_bin_num * input_bin_width
    ambient_timeStamp_num = int(args.ambient_level / 100 * input_stamped_num)
    ambient_timeStamp = create_ambient_timeStamp(
        diffuse_map_matrix=diffuse_input,
        num_timestamps=ambient_timeStamp_num,
        min_range=input_min_range,
        max_range=input_max_range,
        random_seed=ambient_seed,
    )

    total_incoming_timeStamp = merge_pixelized_timeStamp(input_pixelized_timestamp, ambient_timeStamp)
    dead_time_pixelized, discard_num, accept_num = dead_time_simulation_rust(
        input_pixelized_timeStamp=total_incoming_timeStamp,
        pluse_train_num=args.pluse_train_num,
        dead_time=args.dead_time,
        max_workers=args.worker_threads,
        random_seed=dead_time_seed,
    )
    accept_rate = accept_num / (discard_num + accept_num)
    print("The accept ratio is " + str(accept_rate))

    _, histogram, _ = form_histogram_image_rust(
        pixels=dead_time_pixelized,
        bin_number=input_bin_num,
        range_distance=[input_min_range, input_max_range],
        dnl_matrix=input_dnl_matrix,
        num_workers=args.worker_threads or 64,
    )

    if input_dark_histogram_file is not None and args.dark_count_scale != 0:
        dark_histogram = read_darkcout_histogram(input_dark_histogram_file)
        if input_spatial_ratio_map_file is not None:
            spatial_ratio_map = read_spatial_ratio_map(input_spatial_ratio_map_file)
            histogram = add_darkCount_spatial_ratio_map(
                source_histogram=histogram,
                darkcount_histogram=dark_histogram,
                spatial_ratio_map=spatial_ratio_map,
                percenter_value=args.dark_count_scale,
                seed=dark_count_seed,
                spatial_jitter_sigma=args.spatial_jitter_sigma,
            )
        else:
            histogram = add_dark_histogram(histogram, dark_histogram, args.dark_count_scale)

    save_histogram_to_h5(filename=args.output_histogram_file, stamped_histogram=histogram)

    if args.output_depth_image_path:
        depth = form_depth_from_histogram(histogram)
        save_depth_image_png(depth, save_path=args.output_depth_image_path, range_max=input_bin_num - 1)

    execution_time_seconds = perf_counter() - start_time
    print(f"Execution time: {execution_time_seconds:.3f} seconds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
