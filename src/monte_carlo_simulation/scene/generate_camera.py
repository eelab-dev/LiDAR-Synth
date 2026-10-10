from __future__ import annotations

import argparse
import math
import random
from collections.abc import Sequence


DEFAULT_BASELINE_OFFSET = [5.0, 0.0, 0.0]


def sample_camera_position(
    location_min: Sequence[float],
    location_max: Sequence[float],
    rng: random.Random | None = None,
) -> list[float]:
    """Sample each camera coordinate uniformly between inclusive bounds."""
    if len(location_min) != 3 or len(location_max) != 3:
        raise ValueError(
            "camera location bounds must each contain exactly three coordinates"
        )

    lower = [float(value) for value in location_min]
    upper = [float(value) for value in location_max]
    for axis, (minimum, maximum) in zip(
        "xyz", zip(lower, upper, strict=True), strict=True
    ):
        if minimum > maximum:
            raise ValueError(
                f"camera location minimum exceeds maximum on {axis}: {minimum} > {maximum}"
            )

    generator = rng if rng is not None else random
    return [
        generator.uniform(minimum, maximum) for minimum, maximum in zip(lower, upper)
    ]


def sample_baseline_offset(
    baseline_offset_min: Sequence[float] | None,
    baseline_offset_max: Sequence[float] | None,
    rng: random.Random | None = None,
) -> list[float]:
    """Sample a three-dimensional camera-local detector offset in millimetres."""
    if baseline_offset_min is None and baseline_offset_max is None:
        return DEFAULT_BASELINE_OFFSET.copy()
    if baseline_offset_min is None or baseline_offset_max is None:
        raise ValueError(
            "baseline_offset_min and baseline_offset_max must be provided together"
        )
    if len(baseline_offset_min) != 3 or len(baseline_offset_max) != 3:
        raise ValueError(
            "baseline offset bounds must each contain exactly three coordinates"
        )

    lower = [float(value) for value in baseline_offset_min]
    upper = [float(value) for value in baseline_offset_max]
    for axis, (minimum, maximum) in zip(
        "xyz", zip(lower, upper, strict=True), strict=True
    ):
        if minimum > maximum:
            raise ValueError(
                f"baseline offset minimum exceeds maximum on {axis}: "
                f"{minimum} > {maximum}"
            )

    generator = rng if rng is not None else random
    return [
        generator.uniform(minimum, maximum) for minimum, maximum in zip(lower, upper)
    ]


def look_at_from_yaw_pitch(
    camera_position: Sequence[float], yaw: float, pitch: float
) -> list[float]:
    """Return a look-at point for yaw/pitch angles expressed in degrees."""
    if len(camera_position) != 3:
        raise ValueError("camera_position must contain exactly three coordinates")

    yaw_radians = math.radians(yaw)
    pitch_radians = math.radians(pitch)
    cos_pitch = math.cos(pitch_radians)
    direction = [
        cos_pitch * math.sin(yaw_radians),
        math.sin(pitch_radians),
        -cos_pitch * math.cos(yaw_radians),
    ]
    return [
        float(position) + offset
        for position, offset in zip(camera_position, direction, strict=True)
    ]


def resolve_camera_pose(
    camera_location_min: Sequence[float],
    camera_location_max: Sequence[float],
    yaw: float,
    pitch: float,
    rng: random.Random | None = None,
) -> tuple[list[float], list[float]]:
    """Sample a camera position and derive its look-at point from yaw/pitch."""
    camera_position = sample_camera_position(
        camera_location_min, camera_location_max, rng
    )
    return camera_position, look_at_from_yaw_pitch(camera_position, yaw, pitch)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="create camera")
    parser.add_argument(
        "--generate_camera",
        "--generate-camera",
        action="store_true",
        help="Flag to indicate camera generation",
    )
    parser.add_argument(
        "--camera_location_min",
        "--camera-location-min",
        nargs=3,
        type=float,
        required=True,
        help="Minimum camera location coordinates (x y z)",
    )
    parser.add_argument(
        "--camera_location_max",
        "--camera-location-max",
        nargs=3,
        type=float,
        required=True,
        help="Maximum camera location coordinates (x y z)",
    )
    parser.add_argument(
        "--yaw",
        type=float,
        required=True,
        help="Camera yaw angle in degrees; zero points along -Z",
    )
    parser.add_argument(
        "--pitch",
        type=float,
        required=True,
        help="Camera pitch angle in degrees; positive points upward",
    )
    parser.add_argument(
        "--camera_file",
        "--camera-file",
        required=True,
        help="Path to generated camera JSON file",
    )
    parser.add_argument("--detector_width", "--detector-width", type=float, help="Detector width in mm")
    parser.add_argument("--detector_height", "--detector-height", type=float, help="Detector height in mm")
    parser.add_argument(
        "--baseline_offset_min",
        "--baseline-offset-min",
        nargs=3,
        type=float,
        help="Minimum detector offset in camera-local x y z coordinates (mm)",
    )
    parser.add_argument(
        "--baseline_offset_max",
        "--baseline-offset-max",
        nargs=3,
        type=float,
        help="Maximum detector offset in camera-local x y z coordinates (mm)",
    )
    parser.add_argument("--up", nargs=3, type=float, help="Camera up direction")
    parser.add_argument("--delay_mean", "--delay-mean", type=float, help="The mean of delay laser")
    parser.add_argument(
        "--delay_std",
        "--delay-std",
        type=float,
        help="The standard deviation of laser delay",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        camera_location, look_at = resolve_camera_pose(
            camera_location_min=args.camera_location_min,
            camera_location_max=args.camera_location_max,
            yaw=args.yaw,
            pitch=args.pitch,
        )
        baseline_offset = sample_baseline_offset(
            args.baseline_offset_min,
            args.baseline_offset_max,
        )
    except ValueError as error:
        parser.error(str(error))

    input_up = args.up if args.up is not None else [0.0, 1.0, 0.0]

    # Import lazily so the pure sampling functions can be tested without the USD runtime.
    from .scene_generation_lib import generate_camera_json

    generate_camera_json(
        up=input_up,
        camera_position=camera_location,
        look_at_point=look_at,
        filename=args.camera_file,
        detector_width=args.detector_width,
        detector_height=args.detector_height,
        baseline_offset=baseline_offset,
        delay_mean=args.delay_mean,
        delay_std=args.delay_std,
    )


if __name__ == "__main__":
    main()
