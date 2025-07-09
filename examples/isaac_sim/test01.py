# 尝试导入isaacsim模块（如果可用）
try:
    # Third Party
    import isaacsim
except ImportError:
    pass

# Third Party
import torch

# 创建一个4维的零张量，放在CUDA设备上
a = torch.zeros(4, device="cuda:0")

# Standard Library
import argparse

# 创建命令行参数解析器
parser = argparse.ArgumentParser()
parser.add_argument(
    "--headless_mode",
    type=str,
    default=None,
    help="To run headless, use one of [native, websocket], webrtc might not work.",
)
parser.add_argument("--robot", type=str, default="franka.yml", help="robot configuration to load")
parser.add_argument(
    "--external_asset_path",
    type=str,
    default=None,
    help="Path to external assets when loading an externally located robot",
)
parser.add_argument(
    "--external_robot_configs_path",
    type=str,
    default=None,
    help="Path to external robot config when loading an external robot",
)

parser.add_argument(
    "--visualize_spheres",
    action="store_true",
    help="When True, visualizes robot spheres",
    default=False,
)
parser.add_argument(
    "--reactive",
    action="store_true",
    help="When True, runs in reactive mode",
    default=False,
)

parser.add_argument(
    "--constrain_grasp_approach",
    action="store_true",
    help="When True, approaches grasp with fixed orientation and motion only along z axis.",
    default=False,
)

# 解析命令行参数
args = parser.parse_args()

############################################################

# Third Party
from omni.isaac.kit import SimulationApp

# 初始化Isaac Sim仿真应用
simulation_app = SimulationApp(
    {
        "headless": args.headless_mode is not None,  # 如果指定了headless模式则无头运行
        "width": "1920",
        "height": "1080",
    }
)
# Standard Library
from typing import Dict

# Third Party
import carb
import numpy as np
from helper import add_extensions, add_robot_to_scene
from omni.isaac.core import World
from omni.isaac.core.objects import cuboid, sphere

########### OV #################
from omni.isaac.core.utils.types import ArticulationAction

# CuRobo相关导入
# from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
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


########### OV #################;;;;;


def main():
    # 创建CuRobo运动规划实例的计数器
    num_targets = 0
    # 假设障碍物在objects_path中
    my_world = World(stage_units_in_meters=1.0)  # 创建世界，设置单位为米
    stage = my_world.stage

    # 定义世界根节点
    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)
    stage.DefinePrim("/curobo", "Xform")
    # my_world.stage.SetDefaultPrim(my_world.stage.GetPrimAtPath("/World"))
    stage = my_world.stage
    # stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))

    # 创建一个目标立方体用于跟随
    target = cuboid.VisualCuboid(
        "/World/target",
        position=np.array([0.5, 0, 0.5]),  # 位置
        orientation=np.array([0, 1, 0, 0]),  # 方向（四元数）
        color=np.array([1.0, 0, 0]),  # 红色
        size=0.05,  # 大小
    )

    # 设置CuRobo日志级别
    setup_curobo_logger("warn")
    past_pose = None
    n_obstacle_cuboids = 30  # 障碍物立方体数量
    n_obstacle_mesh = 100    # 障碍物网格数量

    # 预热CuRobo实例
    usd_help = UsdHelper()
    target_pose = None

    # 设置张量设备类型（GPU/CPU）
    tensor_args = TensorDeviceType()
    robot_cfg_path = get_robot_configs_path()
    robot_cfg = load_yaml(join_path(robot_cfg_path, args.robot))["robot_cfg"]

    # 如果指定了外部资源路径，则更新配置
    if args.external_asset_path is not None:
        robot_cfg["kinematics"]["external_asset_path"] = args.external_asset_path
    j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]  # 关节名称
    default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]  # 默认配置

    # 将机器人添加到场景中
    robot, robot_prim_path = add_robot_to_scene(robot_cfg, my_world)

    articulation_controller = None

    # 加载世界配置（碰撞检测用）
    world_cfg_table = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    )
    world_cfg_table.cuboid[0].pose[2] -= 0.02  # 调整桌子高度
    world_cfg1 = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    ).get_mesh_world()
    world_cfg1.mesh[0].name += "_mesh"
    world_cfg1.mesh[0].pose[2] = -10.5  # 将网格放在地下

    # 合并世界配置
    world_cfg = WorldConfig(cuboid=world_cfg_table.cuboid, mesh=world_cfg1.mesh)

    # 轨迹优化参数设置
    trajopt_dt = None
    optimize_dt = True
    trajopt_tsteps = 32
    trim_steps = None
    max_attempts = 4
    interpolation_dt = 0.05
    enable_finetune_trajopt = True
    
    # 如果是反应模式，调整参数
    if args.reactive:
        trajopt_tsteps = 40
        trajopt_dt = 0.04
        optimize_dt = False
        max_attempts = 1
        trim_steps = [1, None]
        interpolation_dt = trajopt_dt
        enable_finetune_trajopt = False
    
    # 创建运动规划配置
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        tensor_args,
        collision_checker_type=CollisionCheckerType.MESH,  # 使用网格碰撞检测
        num_trajopt_seeds=12,  # 轨迹优化种子数
        num_graph_seeds=12,    # 图搜索种子数
        interpolation_dt=interpolation_dt,
        collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
        optimize_dt=optimize_dt,
        trajopt_dt=trajopt_dt,
        trajopt_tsteps=trajopt_tsteps,
        trim_steps=trim_steps,
    )
    motion_gen = MotionGen(motion_gen_config)
    
    # 如果不是反应模式，进行预热
    if not args.reactive:
        print("warming up...")
        motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False)

    print("Curobo is Ready")

    # 添加扩展
    add_extensions(simulation_app, args.headless_mode)

    # 创建规划配置
    plan_config = MotionGenPlanConfig(
        enable_graph=False,
        enable_graph_attempt=2,
        max_attempts=max_attempts,
        enable_finetune_trajopt=enable_finetune_trajopt,
        time_dilation_factor=0.5 if not args.reactive else 1.0,
    )

    # 加载舞台并添加世界到舞台
    usd_help.load_stage(my_world.stage)
    usd_help.add_world_to_stage(world_cfg, base_frame="/World")

    # 初始化变量
    cmd_plan = None
    cmd_idx = 0
    my_world.scene.add_default_ground_plane()  # 添加默认地面
    i = 0
    spheres = None
    past_cmd = None
    target_orientation = None
    past_orientation = None
    pose_metric = None
    
    # 主仿真循环
    while simulation_app.is_running():
        my_world.step(render=True)  # 仿真步进
        
        # 如果仿真未开始播放，等待用户点击播放
        if not my_world.is_playing():
            if i % 100 == 0:
                print("**** Click Play to start simulation *****")
            i += 1
            # if step_index == 0:
            #    my_world.play()
            continue

        step_index = my_world.current_time_step_index
        
        # 获取关节控制器
        if articulation_controller is None:
            articulation_controller = robot.get_articulation_controller()
            
        # 初始化机器人关节
        if step_index < 10:
            robot._articulation_view.initialize()
            idx_list = [robot.get_dof_index(x) for x in j_names]
            robot.set_joint_positions(default_config, idx_list)

            robot._articulation_view.set_max_efforts(
                values=np.array([5000 for i in range(len(idx_list))]), joint_indices=idx_list
            )
        if step_index < 20:
            continue

        # 定期更新世界中的障碍物
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

            motion_gen.update_world(obstacles)  # 更新运动规划器的世界模型
            print("Updated World")
            carb.log_info("Synced CuRobo world from stage.")

        # 获取目标立方体的位置和方向
        cube_position, cube_orientation = target.get_world_pose()

        # 初始化历史位置和方向
        if past_pose is None:
            past_pose = cube_position
        if target_pose is None:
            target_pose = cube_position
        if target_orientation is None:
            target_orientation = cube_orientation
        if past_orientation is None:
            past_orientation = cube_orientation

        # 获取仿真中的关节状态
        sim_js = robot.get_joints_state()
        if sim_js is None:
            print("sim_js is None")
            continue
        sim_js_names = robot.dof_names
        
        # 检查是否有NaN值
        if np.any(np.isnan(sim_js.positions)):
            log_error("isaac sim has returned NAN joint position values.")
            
        # 创建CuRobo关节状态对象
        cu_js = JointState(
            position=tensor_args.to_device(sim_js.positions),
            velocity=tensor_args.to_device(sim_js.velocities),  # * 0.0,
            acceleration=tensor_args.to_device(sim_js.velocities) * 0.0,
            jerk=tensor_args.to_device(sim_js.velocities) * 0.0,
            joint_names=sim_js_names,
        )

        # 如果不是反应模式，将速度和加速度设为零
        if not args.reactive:
            cu_js.velocity *= 0.0
            cu_js.acceleration *= 0.0

        # 如果是反应模式且有之前的命令，使用之前的命令
        if args.reactive and past_cmd is not None:
            cu_js.position[:] = past_cmd.position
            cu_js.velocity[:] = past_cmd.velocity
            cu_js.acceleration[:] = past_cmd.acceleration
            
        # 重新排序关节状态以匹配运动规划器的关节名称顺序
        cu_js = cu_js.get_ordered_joint_state(motion_gen.kinematics.joint_names)

        # 如果启用了球体可视化，创建和更新机器人的碰撞球体
        if args.visualize_spheres and step_index % 2 == 0:
            sph_list = motion_gen.kinematics.get_robot_as_spheres(cu_js.position)

            if spheres is None:
                spheres = []
                # 创建球体:

                for si, s in enumerate(sph_list[0]):
                    sp = sphere.VisualSphere(
                        prim_path="/curobo/robot_sphere_" + str(si),
                        position=np.ravel(s.position),
                        radius=float(s.radius),
                        color=np.array([0, 0.8, 0.2]),  # 绿色
                    )
                    spheres.append(sp)
            else:
                for si, s in enumerate(sph_list[0]):
                    if not np.isnan(s.position[0]):
                        spheres[si].set_world_pose(position=np.ravel(s.position))
                        spheres[si].set_radius(float(s.radius))

        # 检查机器人是否静止
        robot_static = False
        if (np.max(np.abs(sim_js.velocities)) < 0.5) or args.reactive:
            robot_static = True
            
        # 检查是否需要重新规划（目标位置或方向发生变化且机器人静止）
        if (
            (
                np.linalg.norm(cube_position - target_pose) > 1e-3
                or np.linalg.norm(cube_orientation - target_orientation) > 1e-3
            )
            and np.linalg.norm(past_pose - cube_position) == 0.0
            and np.linalg.norm(past_orientation - cube_orientation) == 0.0
            and robot_static
        ):
            # 设置末端执行器遥操作目标，使用立方体进行简单的非VR初始化
            ee_translation_goal = cube_position
            ee_orientation_teleop_goal = cube_orientation

            # 计算CuRobo解：
            ik_goal = Pose(
                position=tensor_args.to_device(ee_translation_goal),
                quaternion=tensor_args.to_device(ee_orientation_teleop_goal),
            )
            plan_config.pose_cost_metric = pose_metric
            result = motion_gen.plan_single(cu_js.unsqueeze(0), ik_goal, plan_config)
            # ik_result = ik_solver.solve_single(ik_goal, cu_js.position.view(1,-1), cu_js.position.view(1,1,-1))

            succ = result.success.item()  # ik_result.success.item()
            
            # 如果是第一个目标且启用了抓取约束，创建抓取接近度量
            if num_targets == 1:
                if args.constrain_grasp_approach:
                    pose_metric = PoseCostMetric.create_grasp_approach_metric()
                    
            if succ:
                num_targets += 1
                cmd_plan = result.get_interpolated_plan()  # 获取插值后的轨迹
                cmd_plan = motion_gen.get_full_js(cmd_plan)  # 获取完整的关节状态
                
                # 只获取两个系统中都存在的关节名称
                idx_list = []
                common_js_names = []
                for x in sim_js_names:
                    if x in cmd_plan.joint_names:
                        idx_list.append(robot.get_dof_index(x))
                        common_js_names.append(x)
                # idx_list = [robot.get_dof_index(x) for x in sim_js_names]

                cmd_plan = cmd_plan.get_ordered_joint_state(common_js_names)

                cmd_idx = 0

            else:
                carb.log_warn("Plan did not converge to a solution: " + str(result.status))
            target_pose = cube_position
            target_orientation = cube_orientation
            
        # 更新历史位置和方向
        past_pose = cube_position
        past_orientation = cube_orientation
        
        # 如果有命令计划，执行轨迹
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
            
            # 执行额外的仿真步进（不渲染）
            for _ in range(2):
                my_world.step(render=False)
                
            # 如果轨迹执行完毕，重置
            if cmd_idx >= len(cmd_plan.position):
                cmd_idx = 0
                cmd_plan = None
                past_cmd = None
                
    # 关闭仿真应用
    simulation_app.close()


if __name__ == "__main__":
    main()