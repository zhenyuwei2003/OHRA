import os
import sys
import math
import json
import torch
import random
import trimesh
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

ROOT_DIR = str(Path(__file__).resolve().parents[1])
sys.path.append(ROOT_DIR)


def params_dict_to_data(params_dict):
    def flatten(lst):
        for item in lst:
            if isinstance(item, list):
                yield from flatten(item)
            else:
                yield item

    params_list = []
    for value in params_dict.values():
        if isinstance(value, float):
            params_list.append(value)
        else:
            params_list.extend(list(flatten(value)))
    params_data = params_list[:-44] + [1.0 if params_list[-44 + i] != params_list[-22 + i] else 0.0 for i in range(22)] + [0.0]
    return params_data


class CMapDataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
        batch_size: int,
        robot_names: list = None,
        is_train: bool = True,
        num_points: int = 512,
    ):
        self.batch_size = batch_size
        self.is_train = is_train
        self.num_points = num_points

        self.metadata = torch.load(os.path.join(ROOT_DIR, f"data/grasp_zeroshot/{dataset_name}.pt"), weights_only=True)
        self.robot_names = robot_names if robot_names is not None else sorted(list(set([m[2] for m in self.metadata])))
        self.object_names = ["apple", "baseball", "camera", "cylinder_medium", "door_knob", "pear", "potted_meat_can", "rubber_duck", "tomato_soup_can", "water_bottle"]

        self.hands_params = {}
        for robot_name in self.robot_names:
            params_dict = json.load(open(f"{ROOT_DIR}/grasp_zeroshot/assets/json/{robot_name}.json"))
            params_data = torch.tensor(params_dict_to_data(params_dict), dtype=torch.float32)
            self.hands_params[robot_name] = params_data

        self.metadata_subset = {}
        for robot_name in self.robot_names:
            self.metadata_subset[robot_name] = [(m[0], m[1]) for m in self.metadata if m[2] == robot_name]

        if not self.is_train:
            self.combination = []
            for robot_name in self.robot_names:
                for object_name in self.object_names:
                    self.combination.append((robot_name, object_name))
            self.combination = sorted(self.combination)

        self.object_pcs = {}
        for object_name in self.object_names:
            mesh_path = os.path.join(ROOT_DIR, f"data/grasp_zeroshot/object/{object_name}.stl")
            mesh = trimesh.load_mesh(mesh_path)
            object_pc, _ = mesh.sample(65536, return_index=True)
            self.object_pcs[object_name] = torch.tensor(object_pc, dtype=torch.float32)

    def __getitem__(self, index):
        """
        Train: sample a batch of data
        Validate: get (robot, object) from index, sample a batch of data
        """
        if self.is_train:
            robot_name_batch = []
            object_name_batch = []
            object_pc_batch = []
            hand_params_batch = []
            q_batch = []

            for idx in range(self.batch_size):
                robot_name = random.choice(self.robot_names)
                q, object_name = random.choice(self.metadata_subset[robot_name])

                indices = torch.randperm(65536)[:self.num_points]
                object_pc = self.object_pcs[object_name][indices]
                object_pc += torch.randn(object_pc.shape) * 0.002

                robot_name_batch.append(robot_name)
                object_name_batch.append(object_name)
                object_pc_batch.append(object_pc)
                hand_params_batch.append(self.hands_params[robot_name])
                q_batch.append(q)

            object_pc_batch = torch.stack(object_pc_batch)
            hand_params_batch = torch.stack(hand_params_batch)
            q_batch = torch.stack(q_batch)

            B, N = self.batch_size, self.num_points
            assert object_pc_batch.shape == (B, N, 3), f"Expected: {(B, N, 3)}, Actual: {object_pc_batch.shape}"
            assert hand_params_batch.shape == (B, 152), f"Expected: {(B, 152)}, Actual: {hand_params_batch.shape}"
            assert q_batch.shape == (B, 31), f"Expected: {(B, 31)}, Actual: {q_batch.shape[0]}"

            return {
                'robot_name': robot_name_batch,
                'object_name': object_name_batch,
                'object_pc': object_pc_batch,
                'hand_params': hand_params_batch,
                'q': q_batch
            }
        else:  # validate
            B, N = self.batch_size, self.num_points
            robot_name, object_name = self.combination[index]

            mesh_path = os.path.join(ROOT_DIR, f"data/grasp_zeroshot/object/{object_name}.stl")
            mesh = trimesh.load_mesh(mesh_path)
            object_pc, _ = mesh.sample(65536, return_index=True)
            indices = torch.randperm(65536)[:self.num_points]
            object_pc = torch.tensor(object_pc, dtype=torch.float32)[indices]
            object_pc_batch = object_pc.unsqueeze(0).repeat(B, 1, 1)

            hand_params = self.hands_params[robot_name]
            hand_params_batch = hand_params.unsqueeze(0).repeat(B, 1)

            assert object_pc_batch.shape == (B, N, 3), f"Expected: {(B, N, 3)}, Actual: {object_pc_batch.shape}"
            assert hand_params_batch.shape == (B, 152), f"Expected: {(B, 152)}, Actual: {hand_params_batch.shape}"

            return {
                'robot_name': robot_name,
                'object_name': object_name,
                'object_pc': object_pc_batch,
                'hand_params': hand_params_batch,
            }

    def __len__(self):
        if self.is_train:
            return math.ceil(len(self.metadata) / self.batch_size)
        else:
            return len(self.combination)


def custom_collate_fn(batch):
    return batch[0]


def create_dataloader(cfg, is_train):
    dataset = CMapDataset(
        dataset_name=cfg.dataset_name,
        batch_size=cfg.batch_size,
        robot_names=cfg.robot_names,
        is_train=is_train,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=custom_collate_fn,
        num_workers=cfg.num_workers,
        shuffle=is_train
    )
    return dataloader
