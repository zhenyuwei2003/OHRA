import os
import sys
import torch
import itertools
from pathlib import Path

ROOT_DIR = str(Path(__file__).resolve().parents[2])
sys.path.append(ROOT_DIR)


dataset_name = "dataset_geq8_max200"
dataset_path = f"{ROOT_DIR}/data/grasp_zeroshot/{dataset_name}.pt"

metadata = torch.load(dataset_path, weights_only=True)
robot_names = sorted(list(set([m[2] for m in metadata])))

print(f"{'=' * 100} Dataset Info {'=' * 100}")
print(f"Dataset Name: {dataset_name}")
print("Total Grasp Num:", len(metadata))
count = 0
for combo in itertools.product("0123", repeat=4):
    suffix = "".join(combo)
    robot_name = f"leap_hand_{suffix}"
    num = len([m for m in metadata if m[2] == robot_name])
    print(f"leap_{robot_name.split('_')[-1]}: {num:5d} grasps", end="    ")
    count += 1
    if count % 8 == 0:
        print()
print(f"{'=' * 100} Dataset Info {'=' * 100}")
