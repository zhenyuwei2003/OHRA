from lygra.robot import build_robot
from lygra.utils.robot_visualizer import RobotVisualizer
from lygra.utils.vis_utils import get_box_lineset_visual
import numpy as np 
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot', default='allegro', type=str)
    args = parser.parse_args()

    robot_name = args.robot
    robot = build_robot(robot_name)

    box_min, box_max = robot.get_canonical_space()

    print("min", box_min)
    print("max", box_max)

    robot_tree = robot.get_kinematics_tree()
    
    lower, upper = robot_tree.get_active_joint_limit()
    q = (lower + upper) / 2

    box = get_box_lineset_visual(box_min, box_max)

    viewer = RobotVisualizer(robot)
    robot_mesh = viewer.get_mesh_fk(q, visual=False)
    viewer.show(robot_mesh + [box])
