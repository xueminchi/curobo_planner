#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

try:
    # Third Party
    import isaacsim
except ImportError:
    pass

# Third Party
import torch

a = torch.zeros(4, device="cuda:0")

############################################################

# Third Party
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": False,
        "width": "1920",
        "height": "1080",
    }
)

# Standard Library
from typing import Dict

# Third Party
import carb  # 用来打印日志
import numpy as np
from helper import add_extensions, add_robot_to_scene
from omni.isaac.core import World
from omni.isaac.core.objects import cuboid

########### OV #################
from omni.isaac.core.utils.types import ArticulationAction

# CuRobo
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.types.state import JointState
from curobo.util.logger import log_error, setup_curobo_logger
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import (
    get_assets_path,
    get_filename,
    get_path_of_dir,
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
    # 创建curobo运动生成实例
    num_targets = 0
    
    # 创建世界
    my_world = World(stage_units_in_meters=1.0)  # 世界里面的单位是米
    stage = my_world.stage  # 获取世界stage，stage是usd的根节点

    xform = stage.DefinePrim("/World", "Xform")  # 在stage里面创建一个根节点，名字叫World
    stage.SetDefaultPrim(xform)  # 设置默认的prim为xform，Prim 是usd的基类，Xform是prim的子类，Xform是用来表示变换的
    stage.DefinePrim("/curobo", "Xform")  # 在stage里面创建一个节点，名字叫curobo，类型为Xform

    # 创建目标立方体
    target = cuboid.VisualCuboid(
        "/World/target",  # 目标立方体是World的子节点，名字叫target
        position=np.array([0.5, 0, 0.5]),
        orientation=np.array([0, 1, 0, 0]),
        color=np.array([1.0, 0, 0]),  # 颜色是红色
        size=0.05,
    )

    setup_curobo_logger("warn")
    past_pose = None
    n_obstacle_cuboids = 30
    n_obstacle_mesh = 100

    # 初始化curobo实例
    usd_help = UsdHelper()
    target_pose = None

    tensor_args = TensorDeviceType()
    
    # 默认加载franka机器人
    robot_cfg_path = get_robot_configs_path()
    robot_cfg = load_yaml(join_path(robot_cfg_path, "franka.yml"))["robot_cfg"]  # 通过yaml文件来加载机器人配置
    
    j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]  # 关节名称
    default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]  # 默认配置

    robot, robot_prim_path = add_robot_to_scene(robot_cfg, my_world)  # 添加机器人到世界

    articulation_controller = None

    # 配置世界碰撞检测
    world_cfg_table = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    )
    world_cfg_table.cuboid[0].pose[2] -= 0.02  #  桌子的高度减去0.02米
    world_cfg1 = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    ).get_mesh_world()
    world_cfg1.mesh[0].name += "_mesh"  # 桌子的高度减去10.5米，并添加后缀_mesh
    # world_cfg1.mesh[0].pose[2] = 0
    # 调整一下桌子的size，太大了
    world_cfg1.mesh[0].pose[:3] = [2.65, 2.65, 0.5]
    world_cfg = WorldConfig(cuboid=world_cfg_table.cuboid, mesh=world_cfg1.mesh)  # 创建世界配置

    # 运动生成配置
    trajopt_dt = None
    optimize_dt = True
    trajopt_tsteps = 32
    trim_steps = None
    max_attempts = 4
    interpolation_dt = 0.05
    enable_finetune_trajopt = True
    
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg,  # 添加机器人配置文件，里面有机器人的关节名称，关节限制，关节类型，关节位置，关节速度，关节加速度，关节力矩等
        world_cfg,  # 添加世界配置文件，里面有障碍物的位置，障碍物的形状，障碍物的尺寸等
        tensor_args,  # 添加tensor配置文件，里面有tensor的设备类型，tensor的精度等  
        collision_checker_type=CollisionCheckerType.MESH,  # 添加碰撞检测类型，这里使用的是mesh碰撞检测
        num_trajopt_seeds=12,  # 添加trajopt种子数，这里使用的是12个种子
        num_graph_seeds=12,  # 添加graph种子数，这里使用的是12个种子
        interpolation_dt=interpolation_dt,  # 添加插值时间，这里使用的是0.05秒
        collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},  # 添加碰撞缓存，这里使用的是10个障碍物
        optimize_dt=optimize_dt,
        trajopt_dt=trajopt_dt,
        trajopt_tsteps=trajopt_tsteps,
        trim_steps=trim_steps,
    )
    
    motion_gen = MotionGen(motion_gen_config)
    
    print("warming up...")
    motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False)

    print("Curobo is Ready")

    add_extensions(simulation_app, None)

    plan_config = MotionGenPlanConfig(
        enable_graph=False,
        enable_graph_attempt=2,
        max_attempts=max_attempts,
        enable_finetune_trajopt=enable_finetune_trajopt,
        time_dilation_factor=0.5,
    )

    usd_help.load_stage(my_world.stage)
    usd_help.add_world_to_stage(world_cfg, base_frame="/World")

    cmd_plan = None
    cmd_idx = 0
    my_world.scene.add_default_ground_plane()
    i = 0
    spheres = None
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
        if step_index < 10:
            robot._articulation_view.initialize()
            idx_list = [robot.get_dof_index(x) for x in j_names]
            robot.set_joint_positions(default_config, idx_list)

            robot._articulation_view.set_max_efforts(
                values=np.array([5000 for i in range(len(idx_list))]), joint_indices=idx_list
            )
        if step_index < 20:
            continue

        if step_index == 50 or step_index % 1000 == 0.0:
            print("Updating world, reading w.r.t.", robot_prim_path)
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
            print(len(obstacles.objects))

            motion_gen.update_world(obstacles)
            print("Updated World")
            carb.log_info("Synced CuRobo world from stage.")

        # 获取目标立方体的位置和方向
        cube_position, cube_orientation = target.get_world_pose()

        if past_pose is None:
            past_pose = cube_position
        if target_pose is None:
            target_pose = cube_position
        if target_orientation is None:
            target_orientation = cube_orientation
        if past_orientation is None:
            past_orientation = cube_orientation

        sim_js = robot.get_joints_state()
        if sim_js is None:
            print("sim_js is None")
            continue
        sim_js_names = robot.dof_names
        if np.any(np.isnan(sim_js.positions)):
            log_error("isaac sim has returned NAN joint position values.")
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

        # 检查机器人是否静止
        robot_static = False
        if np.max(np.abs(sim_js.velocities)) < 0.5:
            robot_static = True
            
        # 检查是否需要规划新路径
        if (
            (
                np.linalg.norm(cube_position - target_pose) > 1e-3
                or np.linalg.norm(cube_orientation - target_orientation) > 1e-3
            )
            and np.linalg.norm(past_pose - cube_position) == 0.0
            and np.linalg.norm(past_orientation - cube_orientation) == 0.0
            and robot_static
        ):
            # 设置末端执行器目标
            ee_translation_goal = cube_position
            ee_orientation_teleop_goal = cube_orientation

            # 计算curobo解
            ik_goal = Pose(
                position=tensor_args.to_device(ee_translation_goal),
                quaternion=tensor_args.to_device(ee_orientation_teleop_goal),
            )
            plan_config.pose_cost_metric = pose_metric
            result = motion_gen.plan_single(cu_js.unsqueeze(0), ik_goal, plan_config)

            succ = result.success.item()
            if succ:
                num_targets += 1
                cmd_plan = result.get_interpolated_plan()
                cmd_plan = motion_gen.get_full_js(cmd_plan)
                
                # 获取共同的关节名称
                idx_list = []
                common_js_names = []
                for x in sim_js_names:
                    if x in cmd_plan.joint_names:
                        idx_list.append(robot.get_dof_index(x))
                        common_js_names.append(x)

                cmd_plan = cmd_plan.get_ordered_joint_state(common_js_names)
                cmd_idx = 0
            else:
                carb.log_warn("Plan did not converge to a solution: " + str(result.status))
            target_pose = cube_position
            target_orientation = cube_orientation
            
        past_pose = cube_position
        past_orientation = cube_orientation
        
        # 执行规划的命令
        if cmd_plan is not None:
            cmd_state = cmd_plan[cmd_idx]
            past_cmd = cmd_state.clone()
            
            # 获取完整的自由度状态
            art_action = ArticulationAction(
                cmd_state.position.cpu().numpy(),
                cmd_state.velocity.cpu().numpy(),
                joint_indices=idx_list,
            )
            # 设置从IK获得的期望关节角度
            articulation_controller.apply_action(art_action)
            cmd_idx += 1
            for _ in range(2):
                my_world.step(render=False)
            if cmd_idx >= len(cmd_plan.position):
                cmd_idx = 0
                cmd_plan = None
                past_cmd = None
                
    simulation_app.close()

if __name__ == "__main__":
    main() 