#!/usr/bin/env python3
"""
Simplified Motion Generation Example for Isaac Sim with CuRobo
简化的Isaac Sim与CuRobo运动生成示例

This script demonstrates basic motion planning with a Franka robot.
本脚本展示使用Franka机器人的基本运动规划。

Usage: omni_python simple_motion_gen_reacher.py
"""

try:
    import isaacsim
except ImportError:
    pass

import torch
a = torch.zeros(4, device="cuda:0")

from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({
    "headless": False,  # Set to True for headless mode
    "width": "1920",
    "height": "1080",
})

import numpy as np
import carb
from helper import add_extensions, add_robot_to_scene
from omni.isaac.core import World
from omni.isaac.core.objects import cuboid, sphere
from omni.isaac.core.utils.types import ArticulationAction

# CuRobo imports
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig, Mesh
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.types.state import JointState
from curobo.util.logger import log_error, setup_curobo_logger
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import (
    get_assets_path,
    get_robot_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
)
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
    PoseCostMetric,
)

############################################################

def main():
    """Main function to run the motion generation example"""
    
    # Initialize world and stage
    my_world = World(stage_units_in_meters=1.0)
    stage = my_world.stage

    # Setup basic scene structure
    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)
    stage.DefinePrim("/curobo", "Xform")

    # Create target cube for robot to reach
    target = cuboid.VisualCuboid(
        "/World/target",
        position=np.array([0.5, 0, 0.5]),
        orientation=np.array([0, 1, 0, 0]),
        color=np.array([1.0, 0, 0]),  # Red color
        size=0.05,
    )

    # Setup logging and variables
    setup_curobo_logger("warn")
    past_pose = None
    n_obstacle_cuboids = 30
    n_obstacle_mesh = 100
    usd_help = UsdHelper()
    target_pose = None
    tensor_args = TensorDeviceType()
    
    # Load robot configuration
    robot_cfg_path = get_robot_configs_path()
    robot_cfg = load_yaml(join_path(robot_cfg_path, "franka.yml"))["robot_cfg"]
    j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
    default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]

    robot, robot_prim_path = add_robot_to_scene(robot_cfg, my_world)
    articulation_controller = None

    # Load world configuration (table only)
    world_cfg_table = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    )
    world_cfg_table.cuboid[0].pose[2] -= 0.02  # Lower table height by 0.02m

    # Create mesh version of the table for collision detection
    world_cfg1 = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    ).get_mesh_world()
    world_cfg1.mesh[0].name += "_mesh"
    world_cfg1.mesh[0].pose[2] = -10.5  # Move mesh table far below

    # Combine cuboid and mesh configurations
    world_cfg = WorldConfig(cuboid=world_cfg_table.cuboid, mesh=world_cfg1.mesh)

    # Path to arm_l.obj file
    abs_mesh_path = join_path(get_assets_path(), "scene/arm_l.obj")
    
    # Debug: Check if file exists
    import os
    if os.path.exists(abs_mesh_path):
        print(f"✓ Found arm_l.obj at: {abs_mesh_path}")
    else:
        print(f"✗ arm_l.obj not found at: {abs_mesh_path}")

    # Motion generation configuration
    trajopt_dt = None
    optimize_dt = True
    trajopt_tsteps = 32
    trim_steps = None
    max_attempts = 4
    interpolation_dt = 0.05
    enable_finetune_trajopt = True

    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        tensor_args,
        collision_checker_type=CollisionCheckerType.MESH,
        num_trajopt_seeds=12,
        num_graph_seeds=12,
        interpolation_dt=interpolation_dt,
        collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
        optimize_dt=optimize_dt,
        trajopt_dt=trajopt_dt,
        trajopt_tsteps=trajopt_tsteps,
        trim_steps=trim_steps,
    )

    motion_gen = MotionGen(motion_gen_config)

    # Add world configuration to stage
    print("Adding world configuration to stage...")
    usd_help.load_stage(my_world.stage)
    usd_help.add_world_to_stage(world_cfg, base_frame="/World")
    print("✓ World configuration added successfully")

    # Add arm_l.obj mesh for visualization (WORKING method)
    print(f"\nAdding arm_l.obj mesh for visualization...")
    try:
        # Create mesh object
        arm_mesh = Mesh(
            name="arm_l_mesh",
            pose=[0.6, 0.3, 0.2, 1, 0, 0, 0],  # Position: x=0.6m, y=0.3m, z=0.2m
            file_path=abs_mesh_path,
            scale=[1.0, 1.0, 1.0],  # No scaling needed (file is correct size)
            color=[0.0, 1.0, 0.0, 1.0]  # Green color
        )
        
        # Add mesh to stage using direct method (proven working)
        mesh_path = usd_help.add_mesh_to_stage(arm_mesh, base_frame="/World")
        print(f"✓ arm_l.obj added successfully!")
        print(f"  USD Path: {mesh_path}")
        print(f"  Position: [0.6, 0.3, 0.2] (GREEN)")
        print(f"  File size: 12.3cm max dimension")
        
    except Exception as e:
        print(f"✗ Failed to add arm_l.obj mesh: {e}")

    # Perform Motion Generation warmup
    print("\nWarming up Motion Generation...")
    motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False)
    print("✓ CuRobo is ready!")

    add_extensions(simulation_app, None)

    plan_config = MotionGenPlanConfig(
        enable_graph=False,
        enable_graph_attempt=2,
        max_attempts=max_attempts,
        enable_finetune_trajopt=enable_finetune_trajopt,
        time_dilation_factor=0.5,
    )

    # Main simulation loop
    cmd_plan = None
    cmd_idx = 0
    my_world.scene.add_default_ground_plane()
    i = 0
    past_cmd = None
    target_orientation = None
    past_orientation = None
    pose_metric = None
    
    while simulation_app.is_running():
        my_world.step(render=True)
        if not my_world.is_playing():
            if i % 100 == 0:
                print("**** Click Play to start simulation *****")
            i += 1
            continue

        step_index = my_world.current_time_step_index
        
        if articulation_controller is None:
            articulation_controller = robot.get_articulation_controller()

        # Initialize robot joints
        if step_index < 10:
            robot._articulation_view.initialize()
            idx_list = [robot.get_dof_index(x) for x in j_names]
            robot.set_joint_positions(default_config, idx_list)
            robot._articulation_view.set_max_efforts(
                values=np.array([5000 for i in range(len(idx_list))]), joint_indices=idx_list
            )
            
        if step_index < 20:
            continue

        # Update world obstacles periodically
        if step_index == 50 or step_index % 1000 == 0.0:
            print("Updating world obstacles...")
            obstacles = usd_help.get_obstacles_from_stage(
                only_paths=["/World"],
                reference_prim_path=robot_prim_path,
                ignore_substring=[
                    robot_prim_path,
                    "/World/target",
                    "/World/defaultGroundPlane",
                    "/curobo",
                ],
            ).get_collision_check_world()
            print(f"Found {len(obstacles.objects)} obstacles")
            motion_gen.update_world(obstacles)
            print("✓ World updated")

        # Get target cube position and orientation
        cube_position, cube_orientation = target.get_world_pose()

        # Initialize pose tracking variables
        if past_pose is None:
            past_pose = cube_position
        if target_pose is None:
            target_pose = cube_position
        if target_orientation is None:
            target_orientation = cube_orientation
        if past_orientation is None:
            past_orientation = cube_orientation

        # Get current robot joint state
        sim_js = robot.get_joints_state()
        if sim_js is None:
            print("Warning: sim_js is None")
            continue
            
        sim_js_names = robot.dof_names
        if np.any(np.isnan(sim_js.positions)):
            log_error("Isaac Sim returned NaN joint position values")
            
        # Convert to CuRobo joint state format
        cu_js = JointState(
            position=tensor_args.to_device(sim_js.positions),
            velocity=tensor_args.to_device(sim_js.velocities),
            acceleration=tensor_args.to_device(sim_js.velocities) * 0.0,
            jerk=tensor_args.to_device(sim_js.velocities) * 0.0,
            joint_names=sim_js_names,
        )
        cu_js.velocity *= 0.0
        cu_js.acceleration *= 0.0
        cu_js = cu_js.get_ordered_joint_state(motion_gen.kinematics.joint_names)

        # Check if robot is static and target has moved
        robot_static = (np.max(np.abs(sim_js.velocities)) < 0.5)
        target_moved = (
            (np.linalg.norm(cube_position - target_pose) > 1e-3 or 
             np.linalg.norm(cube_orientation - target_orientation) > 1e-3) and
            np.linalg.norm(past_pose - cube_position) == 0.0 and
            np.linalg.norm(past_orientation - cube_orientation) == 0.0
        )
        
        if target_moved and robot_static:
            # Plan motion to new target position
            print(f"Planning motion to target: {cube_position}")
            
            ik_goal = Pose(
                position=tensor_args.to_device(cube_position),
                quaternion=tensor_args.to_device(cube_orientation),
            )
            
            plan_config.pose_cost_metric = pose_metric
            result = motion_gen.plan_single(cu_js.unsqueeze(0), ik_goal, plan_config)
            
            if result.success.item():
                print("✓ Motion plan successful")
                cmd_plan = result.get_interpolated_plan()
                cmd_plan = motion_gen.get_full_js(cmd_plan)
                
                # Get common joint names between simulation and plan
                idx_list = []
                common_js_names = []
                for x in sim_js_names:
                    if x in cmd_plan.joint_names:
                        idx_list.append(robot.get_dof_index(x))
                        common_js_names.append(x)
                
                cmd_plan = cmd_plan.get_ordered_joint_state(common_js_names)
                cmd_idx = 0
            else:
                print(f"✗ Motion planning failed: {result.status}")
                
            target_pose = cube_position
            target_orientation = cube_orientation
            
        past_pose = cube_position
        past_orientation = cube_orientation
        
        # Execute planned motion
        if cmd_plan is not None:
            cmd_state = cmd_plan[cmd_idx]
            past_cmd = cmd_state.clone()
            
            # Apply joint commands to robot
            art_action = ArticulationAction(
                cmd_state.position.cpu().numpy(),
                cmd_state.velocity.cpu().numpy(),
                joint_indices=idx_list,
            )
            articulation_controller.apply_action(art_action)
            cmd_idx += 1
            
            # Additional simulation steps for smoother motion
            for _ in range(2):
                my_world.step(render=False)
                
            # Check if plan execution is complete
            if cmd_idx >= len(cmd_plan.position):
                cmd_idx = 0
                cmd_plan = None
                past_cmd = None
                print("✓ Motion execution complete")

    simulation_app.close()

if __name__ == "__main__":
    main() 