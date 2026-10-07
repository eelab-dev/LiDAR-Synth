"""Portable public commands around preserved simulation implementations."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

NATIVE_TARGETS = {
    'simulate': 'LiDARSimulation', 'diffuse-map': 'LiDARSceneDiffuseMap',
    'ground-truth': 'LiDARGroundtruth', 'pixelize': 'lidar_pixelation',
}
SCENE_MODULES = {'scene': 'generate_scene', 'camera': 'generate_camera', 'remove-object': 'remove_object'}
COMMANDS = (*NATIVE_TARGETS, *SCENE_MODULES, 'inject-noise', 'histogram', 'run', 'demo')
PATH_OPTIONS = {
    'scene', 'model', 'input', 'output', 'input_file', 'output_file', 'camera_file',
    'geometry_file', 'metadata_file', 'output_metadata_file', 'input_pixelized_file',
    'input_diffuse_file', 'input_dnl_file', 'input_dark_histogram_file', 'input_spatial_ratio_map',
    'output_histogram_file', 'output_depth_image_path',
}


def native_executable(command):
    target = NATIVE_TARGETS[command]
    root = Path(__file__).resolve().parents[3]
    override = os.environ.get('SYNTHETIC_DATA_NATIVE_DIR')
    candidates = [Path(override) / target] if override else []
    directory = 'pixelization/target/release' if command == 'pixelize' else 'simulator/build'
    candidates.append(root / 'native' / directory / target)
    for path in candidates:
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    found = shutil.which(target)
    if found:
        return found
    raise FileNotFoundError(f'{target} is not built. Run ./setup.sh --full, or set SYNTHETIC_DATA_NATIVE_DIR.')


def native_command(command, argv):
    parser = argparse.ArgumentParser(prog=f'synthetic-data {command}')
    if command == 'pixelize':
        parser.add_argument('--input', '--input-file', '--input_file', required=True)
        parser.add_argument('--output', '--output-file', '--output_file', '-o', required=True)
        mapping = {'width': 'image_width', 'height': 'image_height', 'fov_x': 'fov_x', 'fov_y': 'fov_y', 'fov': 'fov'}
    else:
        parser.add_argument('--scene', '--model', required=True)
        parser.add_argument('--output', '-o', required=True)
        mapping = {
            'width': 'width', 'height': 'height', 'fov_x': 'fov_x', 'fov_y': 'fov_y',
            'samples_per_ray': 'ssp', 'seed': 'seed', 'detector_width': 'detectorWidth',
            'detector_height': 'detectorHeight', 'camera_position': 'cameraPosition',
            'look_at': 'lookAt', 'up': 'up', 'baseline_offset': 'baselineOffset',
        }
        if command != 'diffuse-map':
            mapping['detector_distance'] = 'detectorDistance'
        if command == 'ground-truth':
            for unsupported in ('detector_width', 'detector_height', 'baseline_offset', 'detector_distance'):
                del mapping[unsupported]
    vectors = {'camera_position', 'look_at', 'up', 'baseline_offset'}
    for key, original in mapping.items():
        flags = list(dict.fromkeys(['--' + key.replace('_', '-'), '--' + original]))
        kwargs = {'nargs': 3} if key in vectors else {}
        parser.add_argument(*flags, dest=key, type=int if key in {'width', 'height', 'samples_per_ray', 'seed'} else float, **kwargs)
    args = parser.parse_args(argv)
    source = Path(args.input if command == 'pixelize' else args.scene).expanduser().resolve()
    if not source.is_file():
        parser.error(f'Input file does not exist: {source}')
    for key in ('width', 'height', 'samples_per_ray'):
        value = getattr(args, key, None)
        if value is not None and value <= 0:
            parser.error(f'--{key.replace("_", "-")} must be positive')
    output = Path(args.output).expanduser().resolve()
    executable = native_executable(command)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = [executable, '--input_file' if command == 'pixelize' else '--model', str(source), '--output_file' if command == 'pixelize' else '--output', str(output)]
    for key, flag in mapping.items():
        value = getattr(args, key)
        if value is not None:
            result.append('--' + flag)
            result.extend(str(item) for item in (value if isinstance(value, list) else [value]))
    return result


def options_to_args(options):
    result = []
    for key, value in options.items():
        if value is None or value is False:
            continue
        result.append('--' + key.replace('_', '-'))
        if value is not True:
            result.extend(str(item) for item in (value if isinstance(value, list) else [value]))
    return result


def pipeline_commands(config_path, output_override=None):
    config_path = Path(config_path).expanduser().resolve()
    config = json.loads(config_path.read_text())
    if not isinstance(config, dict) or not isinstance(config.get('stages'), list) or not config['stages']:
        raise ValueError("Pipeline config must contain a nonempty 'stages' list")
    base = config_path.parent
    output = Path(output_override).expanduser().resolve() if output_override else Path(config.get('output_dir', 'outputs')).expanduser()
    if not output.is_absolute():
        output = (base / output).resolve()
    commands = []
    for stage in config['stages']:
        if not isinstance(stage, dict):
            raise ValueError('Each stage must be an object')
        command = stage.get('command')
        if command not in COMMANDS or command in {'run', 'demo'}:
            raise ValueError(f'Unsupported pipeline stage: {command!r}')
        if not isinstance(stage.get('options', {}), dict):
            raise ValueError('Stage options must be an object')
        options = {}
        for key, value in stage.get('options', {}).items():
            key = key.replace('-', '_')
            if isinstance(value, str):
                value = value.replace('{output_dir}', str(output))
                if key in PATH_OPTIONS and value.lower() not in {'none', ''}:
                    path = Path(value).expanduser()
                    value = str(path if path.is_absolute() else (base / path).resolve())
            options[key] = value
        commands.append((command, options_to_args(options)))
    return commands


def run_pipeline(argv):
    parser = argparse.ArgumentParser(prog='synthetic-data run')
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output-dir')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without executing')
    args = parser.parse_args(argv)
    for command, stage_args in pipeline_commands(args.config, args.output_dir):
        if args.dry_run:
            print(json.dumps(['synthetic-data', command, *stage_args]))
            continue
        for index, option in enumerate(stage_args[:-1]):
            if option in {'--output', '--output-file', '--camera-file', '--geometry-file', '--output-metadata-file', '--output-histogram-file', '--output-depth-image-path'}:
                Path(stage_args[index + 1]).parent.mkdir(parents=True, exist_ok=True)
        print(f'Running {command}', flush=True)
        dispatch(command, stage_args)


def dispatch(command, argv):
    if command in NATIVE_TARGETS:
        subprocess.run(native_command(command, argv), check=True)
    elif command in SCENE_MODULES:
        subprocess.run([sys.executable, '-m', 'synthetic_data_generation.scene.' + SCENE_MODULES[command], *argv], check=True)
    elif command == 'inject-noise':
        from .noise import main
        main(argv)
    elif command == 'histogram':
        from .clean import main
        main(argv)
    elif command == 'run':
        run_pipeline(argv)
    elif command == 'demo':
        from .demo import main
        main(argv)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Generate synthetic LiDAR photons and noisy sensor histograms.')
    parser.add_argument('command', choices=COMMANDS, help='Use COMMAND --help for stage options')
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(arguments[:1])
    try:
        dispatch(args.command, arguments[1:])
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
