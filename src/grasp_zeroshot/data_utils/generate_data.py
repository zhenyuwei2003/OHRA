import os
import sys
import argparse
import itertools
import subprocess
import multiprocessing as mp
from pathlib import Path

ROOT_DIR = str(Path(__file__).resolve().parents[2])
sys.path.append(ROOT_DIR)

object_list = [
    "apple.stl",
    "baseball.stl",
    "camera.stl",
    "cylinder_medium.stl",
    "door_knob.stl",
    "pear.stl",
    "potted_meat_can.stl",
    "rubber_duck.stl",
    "tomato_soup_can.stl",
    "water_bottle.stl",
]
REPEAT_TIMES = 4


def worker(robot_id, object_name, gpu):
    python = sys.executable
    args = [
        python,
        f"{ROOT_DIR}/third_party/lightning-grasp/demo.py",
        "--robot", f"leap_{robot_id}",
        "--n_contact", "4",
        "--object_mesh_path", f"{ROOT_DIR}/data/grasp_zeroshot/object/{object_name}",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    ret = subprocess.run(
        args=args,
        env=env,
        capture_output=True,
        text=True
    )
    print(f"[Robot: leap_hand_{robot_id} / Object: {object_name[:-4]}] {ret.stdout.strip().splitlines()[-1]}")


def main(gpu_list):
    pool = mp.Pool(processes=len(gpu_list))
    gpu_index = 0

    for finger_link_nums in itertools.product([3, 2, 1, 0], repeat=4):
        finger_link_nums = list(finger_link_nums)
        robot_id = ''.join(str(link_num) for link_num in finger_link_nums)

        for object_name in object_list:
            for _ in range(REPEAT_TIMES):  # run multiple times for each robot-object pair to get more grasps
                gpu = gpu_list[gpu_index % len(gpu_list)]
                pool.apply_async(worker, args=(robot_id, object_name, gpu))
                gpu_index += 1

    pool.close()
    pool.join()


if __name__ == "__main__":
    print("[WARNING] Please ensure to use Lightning Grasp environment!")

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_list', type=lambda string: string.split(','), required=True, help="GPU list for parallel processing")  # input format like '--gpu_list 0,1,2,3,4,5,6,7'
    args = parser.parse_args()

    main(args.gpu_list)
