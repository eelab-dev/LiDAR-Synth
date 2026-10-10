from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

from spad_noise_injection.histogram import form_histogram_image_rust, save_histogram_to_h5
from spad_noise_injection.relaxation_delay import add_zero_relaxation_delay
from spad_noise_injection.time_jitter import add_zero_jitter
from spad_noise_injection.timestamps import (
    get_timeStamp_count_sum,
    read_pixllized_timeStamp_from_h5,
)


def _add_option(parser: argparse.ArgumentParser, old_name: str, **kwargs):
    aliases = [old_name.replace("_", "-"), old_name]
    parser.add_argument(*dict.fromkeys(aliases), dest=old_name.lstrip("-").replace("-", "_"), **kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert pixelized photon data into a clean histogram.")
    _add_option(parser, "--input_pixelized_file", help="Path to pixelized photon .h5 file")
    _add_option(parser, "--min_range", type=float, help="the mininum detecting range for the lidar")
    _add_option(parser, "--bin_width", type=float, help="the bin width of the lidar")
    _add_option(parser, "--bin_number", type=int, help="the bin number of the lidar")
    parser.add_argument("--emission-mean", "--emmission-mean", "--emmission_mean", dest="emmission_mean", type=float, default=0.0, help="Constant emission delay in ps")
    _add_option(parser, "--output_histogram_file", help="Path to output histogram .h5 file")
    return parser


def main(argv: list[str] | None = None) -> int:
    start_time = perf_counter()
    parser = build_parser()
    args = parser.parse_args(argv)

    required_args = {
        "input_pixelized_file": args.input_pixelized_file,
        "min_range": args.min_range,
        "bin_width": args.bin_width,
        "bin_number": args.bin_number,
        "output_histogram_file": args.output_histogram_file,
    }
    missing_args = [f"--{name}" for name, value in required_args.items() if value is None]
    if missing_args:
        parser.error("Missing required input(s): " + ", ".join(missing_args))

    input_files = {"--input_pixelized_file": Path(args.input_pixelized_file)}
    missing_files = [name for name, path in input_files.items() if not path.is_file()]
    if missing_files:
        parser.error("Input file(s) not found for: " + ", ".join(missing_files))

    if args.bin_width <= 0:
        parser.error("--bin_width must be greater than 0")
    if args.bin_number <= 0:
        parser.error("--bin_number must be greater than 0")
    if args.emmission_mean < 0:
        parser.error("--emmission_mean must be non-negative")

    get_timeStamp_count_sum(pixelized_timestamp=args.input_pixelized_file)
    input_pixelized_timestamp = read_pixllized_timeStamp_from_h5(args.input_pixelized_file)

    input_pixelized_timestamp = add_zero_jitter(
        input_pixelized_timestamp,
        emission_delay=args.emmission_mean,
    )
    input_pixelized_timestamp = add_zero_relaxation_delay(input_pixelized_timestamp)

    input_min_range = args.min_range
    input_bin_num = args.bin_number
    input_bin_width = args.bin_width
    input_max_range = input_min_range + input_bin_num * input_bin_width
    _, histogram, _ = form_histogram_image_rust(
        pixels=input_pixelized_timestamp,
        bin_number=input_bin_num,
        range_distance=[input_min_range, input_max_range],
        dnl_matrix=None,
    )
    save_histogram_to_h5(filename=args.output_histogram_file, stamped_histogram=histogram)

    execution_time_seconds = perf_counter() - start_time
    print(f"Execution time: {execution_time_seconds:.3f} seconds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
