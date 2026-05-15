import os
import sys
import time
import hydra
import torch
import warnings
import subprocess
import numpy as np
from pathlib import Path
from termcolor import cprint
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R

ROOT_DIR = str(Path(__file__).resolve().parents[1])
sys.path.append(ROOT_DIR)

from utils.rotation import matrix_to_rot6d
from model.grasp_zeroshot.pl_module import TrainingModule
from data_utils.grasp_zeroshot_dataloader import create_dataloader


def validate_isaac(robot_name, object_name, q_batch, gpu: int = 0, use_gui: bool = False):
    """
    Wrap function for subprocess call (isaac_main.py) to avoid IsaacGym Gym GPU memory leak problem.

    :param robot_name: str
    :param object_name: str
    :param q_batch: torch.Tensor, joint values to validate
    :param gpu: int
    :return: (list<bool>, list<float>), success list & info list
    """
    os.makedirs(os.path.join(ROOT_DIR, 'tmp'), exist_ok=True)
    q_file_path = str(os.path.join(ROOT_DIR, f'tmp/q_list_validate_{gpu}.pt'))
    torch.save(q_batch, q_file_path)

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = env.get("CONDA_PREFIX") + "/lib:" + env.get("LD_LIBRARY_PATH")
    args = [
        "python",
        os.path.join(ROOT_DIR, "grasp_zeroshot/utils/isaac_main.py"),
        "--mode", "validation",
        "--robot_name", robot_name,
        "--object_name", object_name,
        "--batch_size", str(q_batch.shape[0]),
        "--q_file", q_file_path,
        "--gpu", str(gpu),
    ]
    if use_gui:
        args.append("--use_gui")
    ret = subprocess.run(args, capture_output=True, text=True, env=env)

    try:
        ret_file_path = os.path.join(ROOT_DIR, f'tmp/isaac_main_ret_{gpu}.pt')
        save_data = torch.load(ret_file_path)
        success = save_data['success']
        q_isaac = save_data['q_isaac']
        os.remove(q_file_path)
        os.remove(ret_file_path)
    except FileNotFoundError as e:
        cprint(f"Caught a ValueError: {e}", "yellow")
        cprint("[stdout]", "blue")
        cprint(ret.stdout.strip(), "blue")
        cprint("[stderr]", "red")
        cprint(ret.stderr.strip(), "red")
        exit()
    return success, q_isaac


@hydra.main(version_base="1.2", config_path="../configs/grasp_zeroshot", config_name="val")
def main(val_cfg):
    train_cfg = OmegaConf.load(f"{ROOT_DIR}/output/{val_cfg.ckpt_name}/log/hydra/.hydra/config.yaml")

    assert len(val_cfg.gpu) == 1, "For multi-GPU validation, please run val_grasp_mp.py to accelerate the process."
    device = torch.device(f'cuda:{val_cfg.gpu[0]}')

    os.makedirs(os.path.join(ROOT_DIR, 'validate_output'), exist_ok=True)
    log_file_name = os.path.join(ROOT_DIR, f'validate_output/{val_cfg.ckpt_name}.log')
    print('Log file:', log_file_name)

    for validate_epoch in val_cfg.validate_epochs:
        print(f"\n************************ Validating epoch {validate_epoch} ************************")
        with open(log_file_name, 'a') as f:
            print(f"\n************************ Validating epoch {validate_epoch} ************************", file=f)

        pl_module = TrainingModule.load_from_checkpoint(
            checkpoint_path=f"{ROOT_DIR}/output/{val_cfg.ckpt_name}/state_dict/epoch={validate_epoch}.ckpt",
            map_location=device,
            cfg=train_cfg,
        )
        pl_module.eval()

        dataset_cfg = train_cfg.dataset
        dataset_cfg.batch_size = val_cfg.total_batch_size
        dataset_cfg.robot_names = val_cfg.robot_names
        dataloader = create_dataloader(dataset_cfg, is_train=False)

        global_robot_name = None
        all_success_q = []
        time_list = []
        success_num = 0
        total_num = 0
        vis_info = []
        for i, data in enumerate(dataloader):
            robot_name = data['robot_name']
            object_name = data['object_name']

            if robot_name != global_robot_name:
                if global_robot_name is not None:
                    all_success_q = torch.cat(all_success_q, dim=0)
                    diversity_std = torch.std(all_success_q, dim=0).mean()
                    times = np.array(time_list)
                    time_mean = np.mean(times)
                    time_std = np.std(times)

                    success_rate = success_num / total_num * 100
                    cprint(f"[{global_robot_name}]", 'magenta', end=' ')
                    cprint(f"Result: {success_num}/{total_num}({success_rate:.2f}%)", 'yellow', end=' ')
                    cprint(f"Std: {diversity_std:.3f}", 'cyan', end=' ')
                    cprint(f"Time: (mean) {time_mean:.2f} s, (std) {time_std:.2f} s", 'blue')
                    with open(log_file_name, 'a') as f:
                        cprint(f"[{global_robot_name}]", 'magenta', end=' ', file=f)
                        cprint(f"Result: {success_num}/{total_num}({success_rate:.2f}%)", 'yellow', end=' ', file=f)
                        cprint(f"Std: {diversity_std:.3f}", 'cyan', end=' ', file=f)
                        cprint(f"Time: (mean) {time_mean:.2f} s, (std) {time_std:.2f} s", 'blue', file=f)

                    all_success_q = []
                    time_list = []
                    success_num = 0
                    total_num = 0
                global_robot_name = robot_name

            data_count = 0
            predict_q_list = []
            object_pc_list = []
            while data_count != val_cfg.total_batch_size:
                split_num = min(val_cfg.total_batch_size - data_count, val_cfg.split_batch_size)
                object_pc = data['object_pc'][data_count: data_count + split_num].to(device)
                hand_params = data['hand_params'][data_count: data_count + split_num].to(device)
                data_count += split_num

                # random sample initial wrist rotation
                rot_matrix = torch.tensor(R.random(split_num).as_matrix(), dtype=torch.float32)  # (B, 3, 3)
                rot_6d = matrix_to_rot6d(rot_matrix).to(device)
                with torch.no_grad():
                    start_time = time.time()
                    predict_q = pl_module.inference(object_pc, hand_params, rot_6d)
                    end_time = time.time()
                    time_list.append(end_time - start_time)

                object_pc_list.append(object_pc)
                predict_q_list.append(predict_q)
            predict_q_batch = torch.cat(predict_q_list, dim=0)
            object_pc_batch = torch.cat(object_pc_list, dim=0)

            success, isaac_q = validate_isaac(
                robot_name=robot_name,
                object_name=object_name,
                q_batch=predict_q_batch,
                gpu=val_cfg.gpu[0],
                use_gui=val_cfg.use_gui
            )
            succ_num = success.sum().item() if success is not None else -1
            success_q = predict_q_batch[success]
            all_success_q.append(success_q)

            vis_info.append({
                'robot_name': robot_name,
                'object_name': object_name,
                'predict_q': predict_q_batch,
                'object_pc': object_pc_batch,
                'success': success,
                'isaac_q': isaac_q
            })

            cprint(f"[{robot_name}/{object_name}]", 'light_blue', end=' ')
            cprint(f"Result: {succ_num}/{val_cfg.total_batch_size}({succ_num / val_cfg.total_batch_size * 100:.2f}%)", 'green')
            with open(log_file_name, 'a') as f:
                cprint(f"[{robot_name}/{object_name}]", 'light_blue', end=' ', file=f)
                cprint(f"Result: {succ_num}/{val_cfg.total_batch_size}({succ_num / val_cfg.total_batch_size * 100:.2f}%)", 'green', file=f)
            success_num += succ_num
            total_num += val_cfg.total_batch_size

        all_success_q = torch.cat(all_success_q, dim=0)
        diversity_std = torch.std(all_success_q, dim=0).mean()

        times = np.array(time_list)
        time_mean = np.mean(times)
        time_std = np.std(times)

        success_rate = success_num / total_num * 100
        cprint(f"[{global_robot_name}]", 'magenta', end=' ')
        cprint(f"Result: {success_num}/{total_num}({success_rate:.2f}%)", 'yellow', end=' ')
        cprint(f"Std: {diversity_std:.3f}", 'cyan', end=' ')
        cprint(f"Time: (mean) {time_mean:.2f} s, (std) {time_std:.2f} s", 'blue')
        with open(log_file_name, 'a') as f:
            cprint(f"[{global_robot_name}]", 'magenta', end=' ', file=f)
            cprint(f"Result: {success_num}/{total_num}({success_rate:.2f}%)", 'yellow', end=' ', file=f)
            cprint(f"Std: {diversity_std:.3f}", 'cyan', end=' ', file=f)
            cprint(f"Time: (mean) {time_mean:.2f} s, (std) {time_std:.2f} s", 'blue', file=f)

        vis_info_file = f'{val_cfg.ckpt_name}_epoch{validate_epoch}'
        os.makedirs(os.path.join(ROOT_DIR, 'vis_info'), exist_ok=True)
        torch.save(vis_info, os.path.join(ROOT_DIR, f'vis_info/{vis_info_file}.pt'))


if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)
    torch.set_num_threads(8)
    main()
