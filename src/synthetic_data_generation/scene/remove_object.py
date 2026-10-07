import sys
import argparse
import json
from pathlib import Path


def mark_geometry_removed(input_file, output_file, geometry_name, scene_file):
    input_path = Path(input_file)
    output_path = Path(output_file)
    with input_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    try:
        geometry = metadata["objects"][geometry_name]
    except KeyError as error:
        raise KeyError(f"Geometry metadata does not contain objects.{geometry_name}") from error

    geometry["present"] = False
    metadata["scene_file"] = Path(scene_file).name
    processing = metadata.setdefault("processing", {})
    removed_geometries = processing.setdefault("removed_geometries", [])
    if geometry_name not in removed_geometries:
        removed_geometries.append(geometry_name)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=4)
        handle.write("\n")
    temporary_path.replace(output_path)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Remove the short box from an OpenUSD scene and update its geometry metadata."
    )

    parser.add_argument("--input_file", "--input-file", "-i", required=True, help="Path to the input OpenUSD file")
    parser.add_argument("--output_file", "--output-file", "-o", required=True, help="Path to the output OpenUSD file")
    parser.add_argument("--metadata_file", "--metadata-file", help="Input geometry metadata JSON file")
    parser.add_argument("--output_metadata_file", "--output-metadata-file", help="Output geometry metadata JSON file")
    return parser


def main(argv=None):
    parser = build_parser()

    input_file = None
    output_file = None
    args = parser.parse_args(argv)

    if args.input_file:
        input_file = args.input_file

    if args.output_file:
        output_file = args.output_file

    if input_file is None or output_file is None:
        print("⚠️  Error: Both --input_file and --output_file must be provided.")
        parser.print_help()
        sys.exit(1)
    obj_to_remove = "short_box"

    from .scene_generation_lib import scene

    myScene = scene.from_usd_file(input_file)
    myScene.remove_geometry_by_name(obj_to_remove)
    myScene.export_scene(output_file)

    if args.metadata_file:
        metadata_output = args.output_metadata_file or args.metadata_file
        mark_geometry_removed(
            args.metadata_file,
            metadata_output,
            obj_to_remove,
            output_file,
        )


if __name__ == "__main__":
    main()
