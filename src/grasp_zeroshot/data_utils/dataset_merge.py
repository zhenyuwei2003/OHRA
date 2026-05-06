import os
import sys
import torch
import itertools
from pathlib import Path

ROOT_DIR = str(Path(__file__).resolve().parents[2])
sys.path.append(ROOT_DIR)


data_dir = f"{ROOT_DIR}/data/grasp_zeroshot/tmp/"
metadata = []

for combo in itertools.product("0123", repeat=4):
    suffix = "".join(combo)
    robot_name = f"leap_hand_{suffix}"
    folder_path = os.path.join(data_dir, robot_name)
    if not os.path.isdir(folder_path):
        continue

    if int(combo[0]) + int(combo[1]) + int(combo[2]) + int(combo[3]) < 8:  # or robot_name == "leap_hand_3333"
        continue

    for fname in sorted(os.listdir(folder_path)):
        if not fname.endswith(".pt") or not fname.startswith("filtered"):
            continue

        data = torch.load(os.path.join(folder_path, fname), weights_only=True)[:200]
        object_name = os.path.splitext(fname)[0].split("_", 1)[1].rsplit("_", 1)[0]
        for item in data:
            metadata.append((item, object_name, robot_name))

print("Total grasps in merged dataset:", len(metadata))
torch.save(metadata, f"{ROOT_DIR}/data/grasp_zeroshot/dataset_geq8_max200.pt")
