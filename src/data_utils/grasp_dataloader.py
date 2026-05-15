import os
import sys
import json
import math
import torch
import random
import trimesh
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

ROOT_DIR = str(Path(__file__).resolve().parents[1])
sys.path.append(ROOT_DIR)

from utils.hand_model import HandModel
from data_utils.sample_utils import *


class CMapDataset(Dataset):
    def __init__(
        self,
        batch_size: int,
        robot_names: list = None,
        is_train: bool = True,
        debug_object_names: list = None,
        num_points: int = 512,
        object_pc_type: str = "random"
    ):
        self.batch_size = batch_size
        self.robot_names = robot_names if robot_names is not None else ["allegro_hand_left", "barrett_hand", "shadow_hand_right"]
        self.is_train = is_train
        self.num_points = num_points
        self.object_pc_type = object_pc_type

        self.hands_meta_info = {}
        self.hands = {}
        self.hands_params = {}
        for robot_name in self.robot_names:
            self.hands_meta_info[robot_name] = json.load(open(os.path.join(ROOT_DIR, f"assets/meta_infos/base/{robot_name}.json")))
            self.hands[robot_name] = HandModel(
                robot_name=robot_name,
                urdf_type="path",
                urdf_file=self.hands_meta_info[robot_name]["urdf_path"]["canonical"],
                is_canonical=True,
                use_collision=False,
                device=torch.device("cpu")
            )
            params_dict = json.load(open(os.path.join(ROOT_DIR, f"assets/canonical/json/{robot_name}.json")))
            params_list = params_dict_to_list(params_dict)
            params_data = torch.tensor(params_list_to_data(params_list), dtype=torch.float32)
            self.hands_params[robot_name] = params_data

        split_json_path = os.path.join(ROOT_DIR, f"data/CMapDataset/split_train_validate_objects.json")
        dataset_split = json.load(open(split_json_path))
        self.object_names = dataset_split["train"] if is_train else dataset_split["validate"]
        if debug_object_names is not None:
            print("[WARNING] Using debug objects!")
            self.object_names = debug_object_names

        dataset = torch.load(os.path.join(ROOT_DIR, f"data/CMapDataset/canonical_filtered.pt"), weights_only=True)
        self.metadata = [m for m in dataset["metadata"] if m[1] in self.object_names and m[2] in self.robot_names]

        self.metadata_subset = {}
        for robot_name in self.robot_names:
            self.metadata_subset[robot_name] = [(m[0], m[1]) for m in self.metadata if m[2] == robot_name]

        if not self.is_train:
            self.combination = []
            for robot_name in self.robot_names:
                for object_name in self.object_names:
                    self.combination.append((robot_name, object_name))
            self.combination = sorted(self.combination)

        # print("Length of metadata:", len(self.metadata))
        # print("Length of combination:", len(self.combination))

        self.object_pcs = {}
        if self.object_pc_type != "fixed":
            for object_name in self.object_names:
                name = object_name.split("+")
                mesh_path = os.path.join(ROOT_DIR, f"data/object_mesh/{name[0]}/{name[1]}/{name[1]}.stl")
                mesh = trimesh.load_mesh(mesh_path)
                object_pc, _ = mesh.sample(65536, return_index=True)
                self.object_pcs[object_name] = torch.tensor(object_pc, dtype=torch.float32)
        else:
            print("[WARNING] Using fixed object pcs!")

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

                if self.object_pc_type == 'fixed':
                    name = object_name.split('+')
                    object_path = os.path.join(ROOT_DIR, f'data/object_pc/{name[0]}/{name[1]}.pt')
                    object_pc = torch.load(object_path, weights_only=True)[:, :3]
                elif self.object_pc_type == 'random':
                    indices = torch.randperm(65536)[:self.num_points]
                    object_pc = self.object_pcs[object_name][indices]
                    object_pc += torch.randn(object_pc.shape) * 0.002
                else:
                    raise NotImplementedError(f"Unsupported object_pc_type in training: {str(self.object_pc_type)}")

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
            assert hand_params_batch.shape == (B, 66), f"Expected: {(B, 66)}, Actual: {hand_params_batch.shape}"
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

            name = object_name.split('+')
            object_path = os.path.join(ROOT_DIR, f'data/object_pc/{name[0]}/{name[1]}.pt')
            object_pc = torch.load(object_path, weights_only=True)[:, :3].to(torch.float32)
            object_pc_batch = object_pc.unsqueeze(0).repeat(B, 1, 1)

            hand_params = self.hands_params[robot_name]
            hand_params_batch = hand_params.unsqueeze(0).repeat(B, 1)

            assert object_pc_batch.shape == (B, N, 3), f"Expected: {(B, N, 3)}, Actual: {object_pc_batch.shape}"
            assert hand_params_batch.shape == (B, 66), f"Expected: {(B, 66)}, Actual: {hand_params_batch.shape}"

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
        batch_size=cfg.batch_size,
        robot_names=cfg.robot_names,
        is_train=is_train,
        debug_object_names=cfg.debug_object_names,
        object_pc_type=cfg.object_pc_type
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=custom_collate_fn,
        num_workers=cfg.num_workers,
        shuffle=is_train
    )
    return dataloader
