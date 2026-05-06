import os
import sys
import json
import numpy as np
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from scipy.spatial.transform import Rotation as R

ROOT_DIR = str(Path(__file__).resolve().parents[2])
sys.path.append(ROOT_DIR)

FLOAT_DIGITS = 4


def urdf_render(hand_params, output_path):
    env = Environment(
        loader=FileSystemLoader(f"{ROOT_DIR}/grasp_zeroshot/assets"),
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

    with open(output_path, "w") as f:
        f.write(urdf_template.render(hand_params))
    print("Canonical URDF is saved to:", output_path)


if __name__ == "__main__":
    json_dir = os.path.join(ROOT_DIR, "grasp_zeroshot/assets/json/")
    urdf_dir = os.path.join(ROOT_DIR, "grasp_zeroshot/assets/urdf/")
    os.makedirs(urdf_dir, exist_ok=True)

    for fname in os.listdir(json_dir):
        if not fname.endswith(".json"):
            continue
        robot_name = os.path.splitext(fname)[0]
        hand_params = json.load(open(os.path.join(json_dir, fname)))
        urdf_render(hand_params, os.path.join(urdf_dir, f"{robot_name}.urdf"))
