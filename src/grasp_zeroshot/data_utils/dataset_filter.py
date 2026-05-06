import os
import sys
import time
import torch
import argparse
import warnings
import subprocess
import multiprocessing as mp
from pathlib import Path
from termcolor import cprint

ROOT_DIR = str(Path(__file__).resolve().parents[2])
sys.path.append(ROOT_DIR)


def worker(robot_name, object_name, batch_size, gpu):
    args = [
        'python',
        os.path.join(ROOT_DIR, 'grasp_zeroshot/utils/isaac_main.py'),
        '--mode', 'filter',
        '--robot_name', robot_name,
        '--object_name', object_name,
        '--batch_size', str(batch_size),
        '--gpu', gpu
    ]
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = env.get("CONDA_PREFIX") + "/lib:" + env.get("LD_LIBRARY_PATH")
    start_time = time.time()
    ret = subprocess.run(args=args, env=env, capture_output=True, text=True)
    end_time = time.time()
    info = ret.stdout.strip().splitlines()[-1]
    cprint(f'{info:80}', 'light_blue', end=' ')
    cprint(f'time: {end_time - start_time:.2f} s', 'yellow')
    if not info.startswith('<'):
        cprint(ret.stderr.strip(), 'red')


def filter_dataset(args):
    dataset_dir = os.path.join(ROOT_DIR, args.dataset_dir)
    gpu_list = args.gpu_list

    robot_names = os.listdir(dataset_dir)

    pool = mp.Pool(processes=len(gpu_list))
    gpu_index = 0
    for robot_name in sorted(robot_names):
        file_names = os.listdir(os.path.join(dataset_dir, robot_name))
        file_names = [name for name in file_names if not name.startswith('filtered')]
        for file_name in sorted(file_names):
            object_name = file_name.split('.')[0].rsplit("_", 1)[0]
            batch_size = int(file_name.split('.')[0].rsplit("_", 1)[1])
            gpu = gpu_list[gpu_index % len(gpu_list)]

            pool.apply_async(worker, args=(robot_name, object_name, batch_size, gpu))
            gpu_index += 1
    pool.close()
    pool.join()


if __name__ == '__main__':
    warnings.simplefilter(action='ignore', category=FutureWarning)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument('--gpu_list', type=lambda string: string.split(','), required=True, help="GPU list for parallel processing")  # input format like '--gpu_list 0,1,2,3,4,5,6,7'
    args = parser.parse_args()

    filter_dataset(args)
