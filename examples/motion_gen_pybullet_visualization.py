#!/usr/bin/env python3
"""
使用PyBullet可视化运动规划(Motion Generation)过程
"""

import time
import numpy as np
import pybullet as p
import pybullet_data

# Third Party
import torch

# CuRobo
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import Cuboid, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.util.logger import setup_curobo_logger
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

# Local
from pybullet_kinematics_visualization import PyBulletKinematicsVisualizer

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class MotionGenPyBulletVisualizer(PyBulletKinematicsVisualizer):
    """扩展PyBullet可视化器以支持运动规划可视化"""
    
    def __init__(self, robot_config_name="franka.yml", gui=True):
        super().__init__(robot_config_name, gui)
        self.start_markers = []
        self.goal_markers = []
        self.trajectory_markers = []
        
    def add_start_marker(self, position, orientation=None, size=0.05, color=[0, 1, 0, 0.8]):
        """添加起始位置标记
        
        Args:
            position: 起始位置 [x, y, z]
            orientation: 起始方向（四元数）
            size: 标记大小
            color: 标记颜色 [r, g, b, a] - 绿色
        """
        if orientation is None:
            orientation = [0, 0, 0, 1]
            
        # 创建一个绿色立方体作为起始标记
        visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[size, size, size],
            rgbaColor=color
        )
        
        marker_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_shape,
            basePosition=position,
            baseOrientation=orientation
        )
        
        self.start_markers.append(marker_id)
        return marker_id
    
    def add_goal_marker(self, position, orientation=None, size=0.05, color=[1, 0, 0, 0.8]):
        """添加目标位置标记
        
        Args:
            position: 目标位置 [x, y, z]
            orientation: 目标方向（四元数）
            size: 标记大小
            color: 标记颜色 [r, g, b, a] - 红色
        """
        if orientation is None:
            orientation = [0, 0, 0, 1]
            
        # 创建一个红色球体作为目标标记
        visual_shape = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=size,
            rgbaColor=color
        )
        
        marker_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_shape,
            basePosition=position,
            baseOrientation=orientation
        )
        
        self.goal_markers.append(marker_id)
        return marker_id
    
    def add_trajectory_point_marker(self, position, size=0.02, color=[0, 0, 1, 0.4]):
        """添加轨迹点标记
        
        Args:
            position: 轨迹点位置 [x, y, z]
            size: 标记大小
            color: 标记颜色 [r, g, b, a] - 半透明蓝色
        """
        visual_shape = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=size,
            rgbaColor=color
        )
        
        marker_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_shape,
            basePosition=position
        )
        
        self.trajectory_markers.append(marker_id)
        return marker_id
    
    def clear_all_markers(self):
        """清除所有标记"""
        all_markers = self.start_markers + self.goal_markers + self.trajectory_markers
        for marker_id in all_markers:
            try:
                p.removeBody(marker_id)
            except:
                pass  # 忽略已删除的物体
                
        self.start_markers.clear()
        self.goal_markers.clear()
        self.trajectory_markers.clear()
    
    def load_obstacles_from_world_config(self, world_cfg):
        """
        从WorldConfig加载障碍物并在PyBullet中显示
        
        Args:
            world_cfg: WorldConfig对象
            
        Returns:
            list: 创建的障碍物ID列表
        """
        obstacle_ids = []
        
        if hasattr(world_cfg, 'cuboid') and world_cfg.cuboid is not None and len(world_cfg.cuboid) > 0:
            print(f"\n加载 {len(world_cfg.cuboid)} 个立方体障碍物...")
            
            for cuboid_data in world_cfg.cuboid:
                # 获取尺寸和位置信息
                dims = cuboid_data.dims
                pose = cuboid_data.pose
                name = cuboid_data.name
                
                # 创建立方体几何体
                collision_shape = p.createCollisionShape(
                    p.GEOM_BOX, 
                    halfExtents=[dims[0]/2, dims[1]/2, dims[2]/2]
                )
                visual_shape = p.createVisualShape(
                    p.GEOM_BOX, 
                    halfExtents=[dims[0]/2, dims[1]/2, dims[2]/2],
                    rgbaColor=[0.8, 0.2, 0.2, 0.7]  # 半透明红色
                )
                
                # 位置和姿态
                position = [pose[0], pose[1], pose[2]]
                orientation = [pose[4], pose[5], pose[6], pose[3]]  # [x,y,z,w] -> [x,y,z,w]
                
                # 创建障碍物
                obstacle_id = p.createMultiBody(
                    baseMass=0,  # 静态障碍物
                    baseCollisionShapeIndex=collision_shape,
                    baseVisualShapeIndex=visual_shape,
                    basePosition=position,
                    baseOrientation=orientation
                )
                
                obstacle_ids.append(obstacle_id)
                print(f"  - {name}: 位置 {position}, 尺寸 {dims}")
        
        return obstacle_ids
    
    def visualize_trajectory(self, trajectory, start_state, goal_pose, 
                           interpolation_dt=0.02, playback_speed=1.0, 
                           show_trajectory_points=False):
        """可视化完整的运动轨迹
        
        Args:
            trajectory: 轨迹数据 (Joint states)
            start_state: 起始关节状态
            goal_pose: 目标姿态
            interpolation_dt: 插值时间步长
            playback_speed: 播放速度倍数
            show_trajectory_points: 是否显示轨迹点
        """
        print(f"\n开始可视化运动轨迹...")
        print(f"轨迹长度: {len(trajectory.position)} 个时间步")
        print(f"插值时间步长: {interpolation_dt}秒")
        
        # 清除之前的标记
        self.clear_all_markers()
        
        # 计算起始末端执行器位置
        if hasattr(start_state, 'position'):
            start_joints = start_state.position[0].cpu().numpy()
        else:
            start_joints = start_state
            
        extended_start = self._extend_joint_configuration(start_joints)
        self.set_joint_angles(extended_start)
        start_ee_pos, start_ee_quat = self.get_end_effector_pose()
        
        # 添加起始位置标记
        if start_ee_pos is not None:
            self.add_start_marker(start_ee_pos, start_ee_quat)
            print(f"起始位置: {start_ee_pos}")
        
        # 添加目标位置标记
        if hasattr(goal_pose, 'position'):
            goal_pos = goal_pose.position[0].cpu().numpy() if hasattr(goal_pose.position[0], 'cpu') else goal_pose.position[0]
            goal_quat = goal_pose.quaternion[0].cpu().numpy() if hasattr(goal_pose.quaternion[0], 'cpu') else goal_pose.quaternion[0]
        else:
            goal_pos, goal_quat = goal_pose[:3], goal_pose[3:7]
            
        self.add_goal_marker(goal_pos, goal_quat)
        print(f"目标位置: {goal_pos}")
        
        # 如果需要显示轨迹点，先计算所有轨迹点的末端执行器位置
        if show_trajectory_points:
            print("预计算轨迹点...")
            for i in range(0, len(trajectory.position), max(1, len(trajectory.position)//20)):  # 每20个点显示1个
                joint_config = trajectory.position[i].cpu().numpy()
                extended_config = self._extend_joint_configuration(joint_config)
                self.set_joint_angles(extended_config)
                ee_pos, _ = self.get_end_effector_pose()
                if ee_pos is not None:
                    self.add_trajectory_point_marker(ee_pos)
        
        # 播放轨迹
        print(f"\n开始播放轨迹，播放速度: {playback_speed}x")
        print("按 Ctrl+C 可以停止播放")
        
        try:
            for i, joint_positions in enumerate(trajectory.position):
                # 转换为numpy数组
                if hasattr(joint_positions, 'cpu'):
                    joint_config = joint_positions.cpu().numpy()
                else:
                    joint_config = joint_positions
                
                # 扩展关节配置以匹配PyBullet
                extended_config = self._extend_joint_configuration(joint_config)
                
                # 设置机器人姿态
                self.set_joint_angles(extended_config)
                
                # 更新仿真
                p.stepSimulation()
                
                # 控制播放速度
                time.sleep(interpolation_dt / playback_speed)
                
                # 每10步打印一次进度
                if i % 10 == 0:
                    progress = (i + 1) / len(trajectory.position) * 100
                    print(f"\r播放进度: {progress:.1f}%", end='', flush=True)
            
            print(f"\n轨迹播放完成！")
            
        except KeyboardInterrupt:
            print(f"\n轨迹播放被用户中断")
        
        # 获取最终位置
        final_ee_pos, final_ee_quat = self.get_end_effector_pose()
        if final_ee_pos is not None:
            final_error = np.linalg.norm(np.array(final_ee_pos) - np.array(goal_pos))
            print(f"最终位置: {final_ee_pos}")
            print(f"目标误差: {final_error:.6f}m")


def demo_motion_gen_simple_visualization():
    """简单运动规划可视化演示"""
    print("=== 简单运动规划可视化演示 ===")
    
    tensor_args = TensorDeviceType()
    
    # 创建世界配置
    world_config = {
        "cuboid": {
            "table": {
                "dims": [2.0, 2.0, 0.2],  # x, y, z
                "pose": [0.0, 0.0, -0.1, 1, 0, 0, 0.0],  # x, y, z, qw, qx, qy, qz
            },
        },
    }
    
    # 创建运动规划配置
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        "franka.yml",
        world_config,
        tensor_args,
        interpolation_dt=0.02,
        collision_checker_type=CollisionCheckerType.PRIMITIVE,
        use_cuda_graph=True,
    )
    motion_gen = MotionGen(motion_gen_config)
    motion_gen.warmup()
    
    # 创建可视化器
    visualizer = MotionGenPyBulletVisualizer(gui=True)
    
    try:
        # 加载障碍物
        world_cfg = WorldConfig.from_dict(world_config)
        obstacle_ids = visualizer.load_obstacles_from_world_config(world_cfg)
        print(f"加载了 {len(obstacle_ids)} 个障碍物")
        
        # 获取retract配置作为起始状态
        retract_cfg = motion_gen.get_retract_config()
        start_state = JointState.from_position(retract_cfg.view(1, -1))
        
        # 设置目标姿态
        goal_pose = Pose.from_list([0.4, 0.2, 0.4, 1.0, 0.0, 0.0, 0.0])  # x, y, z, qw, qx, qy, qz
        
        print(f"\n规划从起始位置到目标位置的轨迹...")
        
        # 规划轨迹
        result = motion_gen.plan_single(
            start_state, 
            goal_pose, 
            MotionGenPlanConfig(max_attempts=3)
        )
        
        if result.success is not None and (result.success.item() if hasattr(result.success, 'item') else result.success):
            print(f"轨迹规划成功！")
            print(f"规划时间: {result.solve_time:.4f}秒")
            print(f"轨迹时间: {result.motion_time:.4f}秒")
            
            # 获取插值轨迹
            interpolated_trajectory = result.get_interpolated_plan()
            
            # 可视化轨迹
            visualizer.visualize_trajectory(
                interpolated_trajectory, 
                start_state, 
                goal_pose,
                interpolation_dt=result.interpolation_dt,
                playback_speed=0.5,  # 慢速播放
                show_trajectory_points=True
            )
            
        else:
            print(f"轨迹规划失败！状态: {result.status}")
        
        print("\n演示完成！按回车键退出...")
        input()
        
    finally:
        visualizer.disconnect()


def demo_motion_gen_collision_avoidance():
    """避障运动规划可视化演示"""
    print("=== 避障运动规划可视化演示 ===")
    
    tensor_args = TensorDeviceType()
    world_file = "collision_table.yml"
    robot_file = "franka.yml"
    
    # 创建运动规划配置
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_file,
        world_file,
        tensor_args,
        interpolation_dt=0.01,
        collision_checker_type=CollisionCheckerType.PRIMITIVE,
        use_cuda_graph=True,
        num_trajopt_seeds=4,
        trajopt_tsteps=32,
    )
    motion_gen = MotionGen(motion_gen_config)
    motion_gen.warmup()
    
    # 创建可视化器
    visualizer = MotionGenPyBulletVisualizer(gui=True)
    
    try:
        # 加载障碍物
        world_cfg = WorldConfig.from_dict(
            load_yaml(join_path(get_world_configs_path(), world_file))
        )
        obstacle_ids = visualizer.load_obstacles_from_world_config(world_cfg)
        print(f"加载了 {len(obstacle_ids)} 个障碍物")
        
        # 获取retract配置和起始状态
        retract_cfg = motion_gen.get_retract_config()
        start_state = JointState.from_position(retract_cfg.view(1, -1))
        
        # 计算retract姿态作为目标
        state = motion_gen.compute_kinematics(start_state)
        # goal_pose = Pose(
        #     state.ee_pos_seq.squeeze(), 
        #     quaternion=state.ee_quat_seq.squeeze()
        # )

        goal_pose = Pose.from_list([0.4, 0.2, 0.4, 0.0, 1.0, 0.0, 0.0])  # x, y, z, qw, qx, qy, qz

        # 修改起始状态（添加偏移）
        # start_state.position[0, 0] += 0.3  # 修改第一个关节
        # start_state.position[0, 1] += 0.2  # 修改第二个关节
        # start_state.position[0, 2] += 0.2  # 修改第三个关节
        # start_state.position[0, 3] += 0.4  # 修改第四个关节
        # start_state.position[0, 4] += 0.2  # 修改第五个关节
        # start_state.position[0, 5] += 0.5  # 修改第六个关节
        # start_state.position[0, 6] += 0.6  # 修改第七个关节
        
        print(f"\n规划避障轨迹...")
        print(f"从偏移位置回到retract位置")
        
        # 规划轨迹
        result = motion_gen.plan_single(
            start_state, 
            goal_pose, 
            MotionGenPlanConfig(
                max_attempts=5,
                enable_graph=True,
                enable_opt=True,
                timeout=10.0
            )
        )
        
        if result.success is not None and (result.success.item() if hasattr(result.success, 'item') else result.success):
            print(f"避障轨迹规划成功！")
            print(f"规划时间: {result.solve_time:.4f}秒")
            print(f"轨迹时间: {result.motion_time:.4f}秒")
            print(f"状态: {result.status}")
            
            # 获取插值轨迹
            interpolated_trajectory = result.get_interpolated_plan()
            
            # 可视化轨迹
            visualizer.visualize_trajectory(
                interpolated_trajectory, 
                start_state, 
                goal_pose,
                interpolation_dt=result.interpolation_dt,
                playback_speed=0.3,  # 更慢速播放以观察避障
                show_trajectory_points=True
            )
            
        else:
            print(f"避障轨迹规划失败！状态: {result.status}")
        
        print("\n演示完成！按回车键退出...")
        input()
        
    finally:
        visualizer.disconnect()


def demo_motion_gen_multiple_goals():
    """多目标运动规划可视化演示"""
    print("=== 多目标运动规划可视化演示 ===")
    
    tensor_args = TensorDeviceType()
    world_file = "collision_table.yml"
    robot_file = "franka.yml"
    
    # 创建运动规划配置
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_file,
        world_file,
        tensor_args,
        interpolation_dt=0.015,
        collision_checker_type=CollisionCheckerType.PRIMITIVE,
        use_cuda_graph=True,
        num_trajopt_seeds=4,
    )
    motion_gen = MotionGen(motion_gen_config)
    motion_gen.warmup()
    
    # 创建可视化器
    visualizer = MotionGenPyBulletVisualizer(gui=True)
    
    try:
        # 加载障碍物
        world_cfg = WorldConfig.from_dict(
            load_yaml(join_path(get_world_configs_path(), world_file))
        )
        obstacle_ids = visualizer.load_obstacles_from_world_config(world_cfg)
        
        # 定义多个目标位置
        goal_positions = [
            [0.4, 0.3, 0.5, 1.0, 0.0, 0.0, 0.0],    # 目标1
            [0.4, -0.3, 0.3, 1.0, 0.0, 0.0, 0.0],   # 目标2
            [0.2, 0.0, 0.6, 1.0, 0.0, 0.0, 0.0],    # 目标3
        ]
        
        # 获取起始状态
        retract_cfg = motion_gen.get_retract_config()
        current_state = JointState.from_position(retract_cfg.view(1, -1))
        
        for i, goal_pos in enumerate(goal_positions):
            print(f"\n=== 规划到目标 {i+1} ===")
            
            # 创建目标姿态
            goal_pose = Pose.from_list(goal_pos)
            
            # 规划轨迹
            result = motion_gen.plan_single(
                current_state, 
                goal_pose, 
                MotionGenPlanConfig(max_attempts=3)
            )
            
            if result.success is not None and (result.success.item() if hasattr(result.success, 'item') else result.success):
                print(f"到目标 {i+1} 的轨迹规划成功！")
                print(f"规划时间: {result.solve_time:.4f}秒")
                
                # 获取插值轨迹
                interpolated_trajectory = result.get_interpolated_plan()
                
                # 可视化轨迹
                visualizer.visualize_trajectory(
                    interpolated_trajectory, 
                    current_state, 
                    goal_pose,
                    interpolation_dt=result.interpolation_dt,
                    playback_speed=0.5,
                    show_trajectory_points=(i == 0)  # 只在第一次显示轨迹点
                )
                
                # 更新当前状态为轨迹的终点
                final_joint_state = interpolated_trajectory.position[-1]
                if hasattr(final_joint_state, 'view'):
                    current_state = JointState.from_position(final_joint_state.view(1, -1))
                else:
                    # 如果final_joint_state不是tensor，需要转换
                    current_state = JointState.from_position(
                        torch.tensor(final_joint_state).view(1, -1)
                    )
                
                if i < len(goal_positions) - 1:
                    print(f"按回车键继续到下一个目标...")
                    input()
                    
            else:
                print(f"到目标 {i+1} 的轨迹规划失败！状态: {result.status}")
                break
        
        print("\n所有目标完成！按回车键退出...")
        input()
        
    finally:
        visualizer.disconnect()


def main():
    """主函数"""
    print("欢迎使用PyBullet运动规划可视化演示！")
    print("\n可用的演示：")
    print("1. 简单运动规划")
    print("2. 避障运动规划")
    print("3. 多目标运动规划")
    
    choice = input("\n请选择演示 (1-3): ").strip()
    
    if choice == "1":
        demo_motion_gen_simple_visualization()
    elif choice == "2":
        demo_motion_gen_collision_avoidance()
    elif choice == "3":
        demo_motion_gen_multiple_goals()
    else:
        print("无效选择，运行简单演示...")
        demo_motion_gen_simple_visualization()


if __name__ == "__main__":
    setup_curobo_logger("error")
    main() 