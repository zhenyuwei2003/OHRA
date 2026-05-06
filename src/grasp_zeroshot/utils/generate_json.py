import os
import sys
import re
import json
import copy
import itertools
from pathlib import Path

ROOT_DIR = str(Path(__file__).resolve().parents[2])
sys.path.append(ROOT_DIR)


TEMPLATE_FILE = f"{ROOT_DIR}/grasp_zeroshot/assets/leap_hand_template.json"
OUTPUT_DIR = f"{ROOT_DIR}/grasp_zeroshot/assets/json/"

FINGER_MAP = {
    0: 0,  # thumb
    1: 1,  # index
    2: 2,  # middle
    3: 4,  # little
}

JOINT_LIMIT_INDEX_MAP = {
    0: {
        1: [0, 1],
        2: [0, 1, 3],
        3: [0, 1, 3, 4],
    },
    1: {
        1: [0],
        2: [0, 1, 2],
        3: [0, 1, 2, 3],
    },
    2: {
        1: [0],
        2: [0, 1, 2],
        3: [0, 1, 2, 3],
    },
    4: {
        1: [1],
        2: [1, 2, 3],
        3: [1, 2, 3, 4],
    },
}


def zero_like(x):
    if isinstance(x, list):
        return [zero_like(v) for v in x]
    return 0.0


def process_finger(data, finger_idx, link_num):
    # finger_radii
    data["finger_radii"][finger_idx] = (
        template["finger_radii"][finger_idx] if link_num > 0 else 0.0
    )

    # finger_lengths
    lengths = template["finger_lengths"][finger_idx]
    data["finger_lengths"][finger_idx] = [lengths[i] if i < link_num else 0.0 for i in range(3)]

    # joint_origins
    origins = template["joint_origins"][finger_idx]
    if link_num == 0:
        data["joint_origins"][finger_idx] = zero_like(origins)
    if link_num == 1 and finger_idx == 0:
        data["joint_origins"][finger_idx] = [
            origins[i] if i < 2 else zero_like(origins[i])
            for i in range(len(origins))
        ]

    # joint_axes
    axes = template["joint_axes"][finger_idx]
    if link_num == 0:
        data["joint_axes"][finger_idx] = zero_like(axes)

    # joint_lowers / uppers
    def process_joint_limits(template_limits, finger_idx, link_num):
        if link_num == 0:
            return [0.0] * len(template_limits)
        keep_indices = JOINT_LIMIT_INDEX_MAP[finger_idx].get(link_num, [])
        return [
            template_limits[i] if i in keep_indices else 0.0
            for i in range(len(template_limits))
        ]
    data["joint_lowers"][finger_idx] = process_joint_limits(template["joint_lowers"][finger_idx], finger_idx, link_num)
    data["joint_uppers"][finger_idx] = process_joint_limits(template["joint_uppers"][finger_idx], finger_idx, link_num)


def main():
    global template
    with open(TEMPLATE_FILE, "r") as f:
        template = json.load(f)

    for links in itertools.product([0, 1, 2, 3], repeat=4):
        new_json_data = copy.deepcopy(template)

        for i, link_num in enumerate(links):
            finger_idx = FINGER_MAP[i]
            process_finger(new_json_data, finger_idx, link_num)

        json_name = f"leap_hand_{''.join(map(str, links))}.json"
        json_path = os.path.join(OUTPUT_DIR, json_name)

        with open(json_path, "w") as f:
            json.dump(new_json_data, f, indent=4)

        # save json file
        def format_json_string(json_str):  # remove extra newlines and spaces in lists
            def replacer(match):
                content = match.group(1).replace('\n', '').replace('  ', '').replace(',', ', ')
                return f"[{content}]"

            json_str = re.sub(r'\[\s*([^\[\]\{\}]*?)\s*\]', replacer, json_str)
            return json_str

        with open(json_path, 'w', encoding='utf-8') as f:
            f.write(format_json_string(json.dumps(new_json_data, indent=4, ensure_ascii=False)))
        print(f"    - Generated {json_name}")

    print("Generation completed.")


if __name__ == "__main__":
    main()
