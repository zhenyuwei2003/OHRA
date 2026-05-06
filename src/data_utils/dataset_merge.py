import os
import sys
import torch
import argparse
from pathlib import Path
from collections import defaultdict

ROOT_DIR = str(Path(__file__).resolve().parents[1])
sys.path.append(ROOT_DIR)


def main(args):
    dataset_path = os.path.join(ROOT_DIR, "data", "filtered_tmp")
    output_path = os.path.join(ROOT_DIR, args.output_path)

    robot_names = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])

    info = {}
    metadata = []

    for robot_name in robot_names:
        robot_dir = os.path.join(dataset_path, robot_name)

        num_per_object = defaultdict(int)
        num_total = 0
        num_upper_object = 0

        pt_files = sorted([f for f in os.listdir(robot_dir) if f.startswith("filtered_") and f.endswith(".pt")])

        if len(pt_files) == 0:
            print(f"[WARNING] No filtered data found in: {robot_dir}")
            info[robot_name] = {
                "robot_name": robot_name,
                "num_total": 0,
                "num_upper_object": 0,
                "num_per_object": {},
            }
            continue

        for file_name in pt_files:
            file_path = os.path.join(robot_dir, file_name)
            file_name = file_name.replace(".pt", "")[len("filtered_"):]  # {object_name}_{number}
            object_name, number = file_name.rsplit("_", 1)
            expected_num = int(number)

            batch_q = torch.load(file_path, map_location="cpu")
            actual_num = batch_q.shape[0]
            assert actual_num == expected_num, f"Data number mismatch in {file_name}: expected {expected_num}, got {actual_num}!"

            for i in range(actual_num):
                q = batch_q[i]
                metadata.append((q, object_name, robot_name))

            num_total += actual_num
            num_upper_object = max(num_upper_object, actual_num)
            num_per_object[object_name] += actual_num

        info[robot_name] = {
            "robot_name": robot_name,
            "num_total": num_total,
            "num_upper_object": num_upper_object,
            "num_per_object": dict(sorted(num_per_object.items())),
        }

    merged_dataset = {
        "info": info,
        "metadata": metadata,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(merged_dataset, output_path)

    print("=" * 80)
    print(f"Saved merged dataset to: {output_path}")
    print(f"Total metadata length: {len(metadata)}")
    print("Info summary:")
    for robot_name, robot_info in info.items():
        print(
            f"    - {robot_name}: "
            f"num_total={robot_info['num_total']}, "
            f"num_upper_object={robot_info['num_upper_object']}"
        )
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the merged dataset")
    args = parser.parse_args()

    main(args)
