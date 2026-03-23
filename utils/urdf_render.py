import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from jinja2 import Environment, FileSystemLoader

ROOT_DIR = str(Path(__file__).resolve().parents[1])
sys.path.append(ROOT_DIR)


def main(args):
    dir_path = f"{ROOT_DIR}/assets/canonical_extended" if args.extended else f"{ROOT_DIR}/assets/canonical"
    hand_params = json.load(open(os.path.join(dir_path, f"json/{args.robot_name}.json")))
    env = Environment(
        loader=FileSystemLoader(dir_path),
        trim_blocks=True,
        lstrip_blocks=True
    )
    urdf_template = env.get_template('dex_template.jinja2')

    FLOAT_DIGITS = 4
    def get_little_relative_xyz(little_extra_origin, finger_xyz):  # compute the relative position of the little finger knuckle to the extra origin for URDF
        extra_matrix = R.from_euler('xyz', little_extra_origin[3:]).as_matrix()
        relative_xyz = extra_matrix.T @ (np.array(finger_xyz) - np.array(little_extra_origin[:3]))
        return [0.0 if x == -0.0 else round(x, FLOAT_DIGITS) for x in relative_xyz.tolist()]
    env.globals['get_little_relative_xyz'] = get_little_relative_xyz

    output_path = os.path.join(dir_path, f"urdf/{args.robot_name}.urdf")
    print("Canonical URDF will be saved to:", output_path)
    if input("Write to file? (y/n):").strip().lower() == "y":
        with open(output_path, "w") as f:
            f.write(urdf_template.render(hand_params))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_name", type=str, required=True, help="Name of the robot")
    parser.add_argument("--extended", action="store_true", help="Whether to visualize the extended canonical hand model")
    args = parser.parse_args()

    main(args)
