import os
import sys
import argparse
import warnings
from pathlib import Path
from termcolor import cprint
from scipy.spatial.transform import Rotation as R

ROOT_DIR = str(Path(__file__).resolve().parents[1])
sys.path.append(ROOT_DIR)

from IsaacGym.isaac_validator import IsaacValidator
from utils.hand_model import HandModel
from utils.rotation import rot6d_to_matrix

import torch  # IsaacGym must be imported before PyTorch


def isaac_main(
    mode: str,
    robot_name: str,
    object_name: str,
    batch_size: int,
    is_extended: bool,
    q_batch: torch.Tensor = None,
    gpu: int = 0,
    use_gui: bool = False
):
    """
    For filtering dataset and validating grasps.

    :param mode: str, 'filter' or 'validation'
    :param robot_name: str
    :param object_name: str
    :param batch_size: int, number of grasps in IsaacGym Gym simultaneously
    :param is_extended: bool, whether to use the extended version of canonical URDF
    :param q_batch: torch.Tensor (validation only)
    :param gpu: int, specify the GPU device used by IsaacGym Gym
    :param use_gui: bool, whether to visualize IsaacGym Gym simulation process
    :return: success: (batch_size,), bool, whether each grasp is successful in IsaacGym Gym;
             q_isaac: (success_num, DOF), torch.float32, successful joint values after the grasp phase
    """
    if use_gui:  # for unknown reason otherwise will segmentation fault :(
        gpu = 0

    urdf_path = os.path.join(ROOT_DIR, 'assets', 'canonical_extended' if is_extended else 'canonical', 'urdf')
    hand = HandModel(
        robot_name=robot_name,
        urdf_type='path',
        urdf_file=os.path.join(urdf_path, f"{robot_name}.urdf"),
        is_canonical=True,
        is_extended=is_extended
    )

    simulator = IsaacValidator(
        robot_name=robot_name,
        hand=hand,
        batch_size=batch_size,
        gpu=gpu,
        is_filter=(mode == 'filter'),
        use_gui=use_gui
    )
    print("[IsaacGym] IsaacValidator is created.")

    object_name_split = object_name.split('+')
    simulator.set_asset(
        robot_path=urdf_path,
        robot_file=f'{robot_name}.urdf',
        object_path=os.path.join(ROOT_DIR, 'data', 'object_mesh', object_name_split[0], object_name_split[1]),
        object_file=f'{object_name_split[1]}.urdf'
    )
    simulator.create_envs()
    print("[IsaacGym] IsaacValidator preparation is done.")

    if mode == 'filter':
        q = torch.load(os.path.join(ROOT_DIR, 'data', 'filtered_tmp', robot_name, f'{object_name}_{batch_size}.pt'))

        q_pos, q_6d, q_joint = q[:, :3], q[:, 3:9], q[:, 9:]
        q_xyzw = torch.from_numpy(R.from_matrix(rot6d_to_matrix(q_6d)).as_quat())
        q_batch = torch.cat([q_pos, q_xyzw, q_joint], dim=-1).to(dtype=torch.float32, device=torch.device('cpu'))

    simulator.set_actor_pose_dof(q_batch.to(torch.device('cpu')))
    success, q_isaac = simulator.run_sim()
    simulator.destroy()

    return success, q_isaac


# for Python scripts subprocess call to avoid IsaacGym Gym GPU memory leak problem
if __name__ == '__main__':
    print("[INFO] Start isaac_main.py ...")
    warnings.simplefilter(action='ignore', category=FutureWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--robot_name', type=str, required=True)
    parser.add_argument('--object_name', type=str, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--is_extended', action='store_true')
    parser.add_argument('--q_file', type=str)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--use_gui', action='store_true')
    args = parser.parse_args()

    print(f'GPU: {args.gpu}')
    assert args.mode in ['filter', 'validation'], f"Unknown mode: {args.mode}!"
    q_batch = torch.load(args.q_file, map_location=f'cpu') if args.q_file is not None else None
    success, q_isaac = isaac_main(
        mode=args.mode,
        robot_name=args.robot_name,
        object_name=args.object_name,
        batch_size=args.batch_size,
        is_extended=args.is_extended,
        q_batch=q_batch,
        gpu=args.gpu,
        use_gui=args.use_gui
    )

    success_num = success.sum().item()
    print(q_isaac.shape)
    if args.mode == 'filter':
        print(f"<{args.robot_name}/{args.object_name}> before: {args.batch_size}, after: {success_num}")
        if success_num > 0:
            q_filtered = q_isaac[success]
            save_dir = str(os.path.join(ROOT_DIR, 'data', 'filtered_tmp', args.robot_name))
            os.makedirs(save_dir, exist_ok=True)
            torch.save(q_filtered, os.path.join(save_dir, f'filtered_{args.object_name}_{success_num}.pt'))
    elif args.mode == 'validation':
        cprint(f"[{args.robot_name}/{args.object_name}] Result: {success_num}/{args.batch_size}", 'green')
        save_data = {
            'success': success,
            'q_isaac': q_isaac
        }
        os.makedirs(os.path.join(ROOT_DIR, 'tmp'), exist_ok=True)
        torch.save(save_data, os.path.join(ROOT_DIR, f'tmp/isaac_main_ret_{args.gpu}.pt'))
