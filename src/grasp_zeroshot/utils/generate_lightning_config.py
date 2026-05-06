import os
import sys
from pathlib import Path
from itertools import product

ROOT_DIR = str(Path(__file__).resolve().parents[2])
sys.path.append(ROOT_DIR)

OUTPUT_DIR = f"{ROOT_DIR}/third_party/lightning-grasp/lygra/robot/canonical/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FINGERS = ["thumb", "index", "middle", "little"]

LINK_SUFFIX = {
    3: "distal",
    2: "middle",
    1: "proximal"
}

FINGER_JOINTS = {
    "thumb": ["thumb_joint1", "thumb_joint2", "thumb_joint4", "thumb_joint5"],
    "index": ["index_joint1", "index_joint2", "index_joint3", "index_joint4"],
    "middle": ["middle_joint1", "middle_joint2", "middle_joint3", "middle_joint4"],
    "little": ["little_joint1", "little_joint2", "little_joint3", "little_joint4"]
}

JOINT_COUNT = {
    3: 4,
    2: 3,
    1: 2,
    0: 0
}

TEMPLATE = '''import os
import sys
import numpy as np
from pathlib import Path

ROOT_DIR = str(Path(__file__).resolve().parents[5])
REPO_DIR = str(Path(__file__).resolve().parents[3])
sys.path.append(REPO_DIR)

from lygra.robot.base import RobotInterface


class Leap{suffix}(RobotInterface):
    def get_canonical_space(self):
        box_min = np.array([0.08, -0.03, 0.08], dtype=np.float32)
        box_max = np.array([0.12, 0.03, 0.13], dtype=np.float32)
        return box_min, box_max 

    def get_default_urdf_path(self):
        return f'{{ROOT_DIR}}/grasp_zeroshot/assets/urdf/leap_hand_{suffix}.urdf'

    def get_contact_field_config(self):
        config = {{
            "type": "v1",
            "movable_link": {{}},
            "static_link": {{}}
        }}

        for link in [
{contact_links}
        ]:
            config["movable_link"][link] = {{
                "disabled_normal": [
                    (np.array([0.0, 0.0, -1.0]), 1.57)
                ]
            }}

        config["static_link"]["base_link"] = {{
            "allowed_normal": [
                (np.array([[1.0, 0.0, 0.0]]), 3.1416 * 0.25)
            ]
        }}

        return config

    def get_active_joints(self):
        return {active_joints}

    def get_base_link(self):
        return "world"

    def get_static_links(self):
        return ["palm"]

    def get_mesh_scale(self):
        return 1.0
'''


def main():
    for combo in product([0, 1, 2, 3], repeat=4):
        suffix = ''.join(map(str, combo))

        # -------- contact links --------
        contact_links = []
        for finger, num in zip(FINGERS, combo):
            if num == 0:
                continue
            contact_links.append(f'"{finger}_{LINK_SUFFIX[num]}"')

        contact_links_str = ""
        for link in contact_links:
            contact_links_str += f"            {link},\n"

        # -------- active joints --------
        joints_lines = []
        for finger, num in zip(FINGERS, combo):
            count = JOINT_COUNT[num]
            if count == 0:
                continue
            finger_joints = FINGER_JOINTS[finger][:count]
            line = "            " + ", ".join(f"'{j}'" for j in finger_joints)
            joints_lines.append(line)

        active_joints_str = "[\n" + ",\n".join(joints_lines) + "\n        ]"

        code = TEMPLATE.format(
            suffix=suffix,
            contact_links=contact_links_str.rstrip(),
            active_joints=active_joints_str
        )

        file_path = os.path.join(OUTPUT_DIR, f"leap_hand_{suffix}.py")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(code)
        print(f"    - Generated leap_hand_{suffix}.py")


if __name__ == "__main__":
    main()
