import argparse
import numpy as np
from omni.isaac.kit import SimulationApp
from omni.isaac.core import World
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from helper import add_robot_to_scene

parser = argparse.ArgumentParser()
parser.add_argument("--robot", type=str, default="franka.yml", help="robot configuration to load")
args = parser.parse_args()

simulation_app = SimulationApp({"headless": False, "width": "1920", "height": "1080"})


my_world = World(stage_units_in_meters=1.0)
robot_cfg_path = get_robot_configs_path()
robot_cfg = load_yaml(join_path(robot_cfg_path, args.robot))["robot_cfg"]

robot, robot_prim_path = add_robot_to_scene(robot_cfg, my_world)
