import argparse
import json
import random
from pathlib import Path


def write_geometry_metadata(file_name, metadata):
    output_path = Path(file_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=4)
        handle.write("\n")
    temporary_path.replace(output_path)


def build_parser():
    parser = argparse.ArgumentParser(description="Export a Cornell Box scene to OpenUSD format.")
    parser.add_argument("-o", "--output_file", "--output-file", type=str, default="cornell_box.usda", help="Output OpenUSD file path")
    parser.add_argument("--detectorDistance", type=float, default = 1000 , help = "distance from detector to the ")
    parser.add_argument("--generate_camera", "--generate-camera", action="store_true", help="Flag to indicate camera generation")
    parser.add_argument("--camera_location", "--camera-location", nargs=3, type=float, help="Camera location coordinates (x y z)")
    parser.add_argument("--look_at", "--look-at", nargs=3, type=float, help="Camera look-at coordinates (x y z)")
    parser.add_argument("--camera_file", "--camera-file", type=str, help="Path to generated camera JSON file")
    parser.add_argument("--short_box_rotation", "--short-box-rotation", type=float, default=0, help="Short box rotation around the y axis in degrees")
    parser.add_argument(
        "--short_box_shape",
        "--short-box-shape",
        choices=("box",),
        default="box",
        help="Short-object geometry type; currently supported: box",
    )
    parser.add_argument("--geometry_file", "--geometry-file", type=str, help="Output JSON path for generated geometry metadata")
    return parser


def main(argv=None):
    parser = build_parser()
    output_file = "cornell_box.usda"
    detector_distance = 1000
    args = parser.parse_args(argv)
    from .scene_generation_lib import generate_camera_json, material_info, scene

    if args.output_file:
        output_file = args.output_file
    if args.generate_camera:
        camera_output_file = args.camera_file
        camear_location = args.camera_location
        camear_look = args.look_at
        generate_camera_json(camera_position=camear_location,look_at_point=camear_look,filename=camera_output_file)



    materials = [
        material_info(material_type="lambert", material_name="white",  properties={"reflectivity":0.8}),
        material_info(material_type="lambert", material_name="red",    properties={"reflectivity":0.5}),
        material_info(material_type="lambert", material_name="green",  properties={"reflectivity":0.3}),
    ]
    material_names = [m._material_name for m in materials]

    def random_material():
        return random.choice(material_names)

    # dims
    box_width, box_height, box_depth = 535, 540, 516
    wall_width = 14

    # Create scene object
    scene_obj = scene(materials)

    floor_size = [box_width, wall_width, box_depth]
    floor_position = [0, 0.5*wall_width, 0]
    ceiling_size = [box_width, wall_width, box_depth]
    ceiling_position = [0, box_height + 0.5*wall_width, 0]
    back_wall_size = [box_width, box_height + wall_width, wall_width]
    back_wall_position = [0, box_height/2 + 0.5*wall_width, -box_depth/2]
    left_wall_size = [wall_width, box_height + wall_width, box_depth]
    left_wall_position = [-box_width/2, box_height/2 + 0.5*wall_width, 0]
    right_wall_size = [wall_width, box_height + wall_width, box_depth]
    right_wall_position = [box_width/2, box_height/2 + 0.5*wall_width, 0]
    ground_size = [box_width*5, 1, box_depth*5]
    ground_position = [0, 0, 0]
    outer_wall_size = [box_width*5, box_height*5, 1]
    outer_wall_position = [0, (box_height*5)/2, -box_depth/2 - 200]

    # Floor
    scene_obj.create_box_with_material(
        floor_size,
        floor_position,
        random_material(),
        "floor"
    )

    # Ceiling
    scene_obj.create_box_with_material(
        ceiling_size,
        ceiling_position,
        random_material(),
        "ceiling"
    )

    # Back wall
    scene_obj.create_box_with_material(
        back_wall_size,
        back_wall_position,
        random_material(),
        "back_wall"
    )

    # Left wall (Red)
    scene_obj.create_box_with_material(
        left_wall_size,
        left_wall_position,
        random_material(),
        "left_wall"
    )

    # Right wall (Green)
    scene_obj.create_box_with_material(
        right_wall_size,
        right_wall_position,
        random_material(),
        "right_wall"
    )

    # Ground plate
    scene_obj.create_box_with_material(
        ground_size,
        ground_position,
        random_material(),
        "ground"
    )

    # Wall plate
    scene_obj.create_box_with_material(
        outer_wall_size,
        outer_wall_position,
        random_material(),
        "wall"
    )


    short_block_width, short_block_depth, short_block_height = 160, 165, 160
    # trangular_short = np.sqrt(((short_block_width/2)**2 + (short_block_depth/2)**2)) * 2
    # tall_block_width = trangular_short + random.random() * 0.3 * trangular_short
    # tall_block_depth = short_block_depth + random.uniform(-0.8,0.5) * 0.2 * short_block_depth

    # tall_block_height = 330 + random.uniform(-1, 1) * 0.1 * 330
    # distance = 320 + random.uniform(-1, 1) * 0.2 * 320
    # block_x = random.uniform(-box_width / 2 + tall_block_width / 2, box_width / 2 - tall_block_width / 2)


    # short_block_z = random.uniform(-box_depth/2+0.5*trangular_short-20 , box_depth/2 - tall_block_depth -0.5*trangular_short - 10)
    # tall_block_z = random.uniform(short_block_z + 0.5*trangular_short+ 10 + 0.5 * tall_block_depth, box_depth/2 - 0.5*tall_block_depth)

    tall_block_width = short_block_width + random.random() * 0.3 * short_block_width
    tall_block_depth = short_block_depth + random.random() * 0.2 * short_block_depth
    tall_block_height = 330 + random.uniform(-1, 1) * 0.1 * 330
    distance = 300 + random.uniform(-1, 1) * 0.2 * 300
    block_x = random.uniform(-box_width / 2 + tall_block_width / 2, box_width / 2 - tall_block_width / 2)
    short_block_z = random.uniform(-box_depth/2 +0.5*short_block_depth, box_depth/2 - tall_block_depth -0.5*short_block_depth)
    tall_block_z = random.uniform(short_block_z + 0.5*short_block_depth + 5 + 0.5 * tall_block_depth, 265 - 0.5*tall_block_depth)
    tall_box_size = [tall_block_width, tall_block_height, tall_block_depth]
    tall_box_position = [block_x, tall_block_height/2 + 0.5*wall_width, tall_block_z]
    short_box_size = [short_block_width, short_block_height, short_block_depth]
    short_box_position = [block_x, short_block_height/2 + 0.5*wall_width, short_block_z]
    # Tall block
    scene_obj.create_box_with_material(
        tall_box_size,
        tall_box_position,
        random_material(),
        "tall_box"
    )

    # Short object. Keep the geometry name stable so downstream removal continues to work.
    if args.short_box_shape == "box":
        scene_obj.create_box_with_material(
            short_box_size,
            short_box_position,
            random_material(),
            "short_box"
        )

    scene_obj.rotate_geometry_along_y(geo_name="short_box", angle_degrees=args.short_box_rotation)

    # Print scene parameters
    print(f"📌 Box dimensions: {box_width}, {box_height}, {box_depth}")
    print(f"📌 Short block: {short_block_width}, {short_block_height}, {short_block_depth}")
    print(f"📌 Tall block: {tall_block_width}, {tall_block_height}, {tall_block_depth}")
    print(f"📌 Distance between blocks: {tall_block_z - short_block_z}")
    print(f"📌 Block X position: {block_x}")

    # scene_obj.save()
    scene_obj.export_scene(file_name=output_file)
    geometry_file = args.geometry_file or str(Path(output_file).with_suffix(".json"))
    geometry_metadata = {
        "schema_version": 1,
        "units": "mm",
        "scene_file": Path(output_file).name,
        "coordinate_system": {
            "vertical_axis": "y",
            "depth_axis": "z"
        },
        "room": {
            "size": [box_width, box_height, box_depth],
            "wall_width": wall_width
        },
        "environment_geometry": {
            "floor": {"size": floor_size, "position": floor_position},
            "ceiling": {"size": ceiling_size, "position": ceiling_position},
            "back_wall": {"size": back_wall_size, "position": back_wall_position},
            "left_wall": {"size": left_wall_size, "position": left_wall_position},
            "right_wall": {"size": right_wall_size, "position": right_wall_position},
            "ground": {"size": ground_size, "position": ground_position},
            "wall": {"size": outer_wall_size, "position": outer_wall_position}
        },
        "objects": {
            "tall_box": {
                "present": True,
                "size": tall_box_size,
                "position": tall_box_position,
                "rotation": {"axis": "y", "angle_degrees": 0.0}
            },
            "short_box": {
                "present": True,
                "shape": args.short_box_shape,
                "size": short_box_size,
                "position": short_box_position,
                "rotation": {
                    "axis": "y",
                    "angle_degrees": float(args.short_box_rotation)
                }
            }
        }
    }
    write_geometry_metadata(geometry_file, geometry_metadata)
    print(f"Wrote scene: {output_file}")
    print(f"Wrote geometry metadata: {geometry_file}")


if __name__ == "__main__":
    main()
