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

ROOT_DIR = str(Path(__file__).resolve().parents[1])
sys.path.append(ROOT_DIR)


def worker(robot_name, object_name, batch_size, gpu, is_extended):
    args = [
        'python',
        os.path.join(ROOT_DIR, 'IsaacGym/isaac_main.py'),
        '--mode', 'filter',
        '--robot_name', robot_name,
        '--object_name', object_name,
        '--batch_size', str(batch_size),
        '--gpu', gpu
    ]
    if is_extended:
        args.append('--is_extended')
    # for arg in args:
    #     print(arg, end=' ')
    # print('\n')
    start_time = time.time()
    ret = subprocess.run(args, capture_output=True, text=True)
    end_time = time.time()
    info = ret.stdout.strip().splitlines()[-1]
    cprint(f'{info:80}', 'light_blue', end=' ')
    cprint(f'time: {end_time - start_time:.2f} s', 'yellow')
    if not info.startswith('<'):
        cprint(ret.stderr.strip(), 'red')


def filter_dataset(args):
    dataset_path = os.path.join(ROOT_DIR, args.dataset_path)
    gpu_list = args.gpu_list

    dataset = torch.load(dataset_path, map_location=torch.device('cpu'), weights_only=True)
    info, metadata = dataset['info'], dataset['metadata']

    robot_names = sorted(list(info.keys()))
    object_names = []
    for robot_name in robot_names:
        object_names.extend(list(info[robot_name]['num_per_object'].keys()))
    object_names = sorted(list(set(object_names)))

    pool = mp.Pool(processes=len(gpu_list))
    gpu_index = 0
    for robot_name in robot_names:
        for object_name in object_names:
            q_list = [m[0] for m in metadata if m[1] == object_name and m[2] == robot_name]
            batch_size = len(q_list)
            if batch_size == 0:
                continue
            gpu = gpu_list[gpu_index % len(gpu_list)]

            q_batch = torch.stack(q_list, dim=0)
            save_dir = str(os.path.join(ROOT_DIR, 'data', 'filtered_tmp', robot_name))
            os.makedirs(save_dir, exist_ok=True)
            torch.save(q_batch, os.path.join(save_dir, f'{object_name}_{batch_size}.pt'))

            pool.apply_async(worker, args=(robot_name, object_name, batch_size, gpu, args.is_extended))
            gpu_index += 1
    pool.close()
    pool.join()


if __name__ == '__main__':
    warnings.simplefilter(action='ignore', category=FutureWarning)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument('--gpu_list', type=lambda string: string.split(','), required=True, help="GPU list for parallel processing")  # input format like '--gpu_list 0,1,2,3,4,5,6,7'
    parser.add_argument("--is_extended", action="store_true", help="Whether to use the extended canonical hand model")
    args = parser.parse_args()

    filter_dataset(args)
