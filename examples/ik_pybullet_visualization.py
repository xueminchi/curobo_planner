#!/usr/bin/env python3
"""
使用PyBullet可视化逆运动学(IK)求解过程
"""

import time
import numpy as np
import pybullet as p
import pybullet_data

# Third Party
import torch

# CuRobo
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

# Local
from pybullet_kinematics_visualization import PyBulletKinematicsVisualizer

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class IKPyBulletVisualizer(PyBulletKinematicsVisualizer):
    """扩展PyBullet可视化器以支持IK可视化"""
    
    def __init__(self, robot_config_name="franka.yml", gui=True):
        super().__init__(robot_config_name, gui)
        self.target_markers = []
        self.solved_markers = []
        
    def add_target_marker(self, position, orientation=None, size=0.05, color=[1, 0, 0, 0.8]):
        """添加目标位置标记
        
        Args:
            position: 目标位置 [x, y, z]
            orientation: 目标方向（四元数）
            size: 标记大小
            color: 标记颜色 [r, g, b, a]
        """
        if orientation is None:
            orientation = [0, 0, 0, 1]
            
        # 创建一个半透明的球体作为目标标记
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
        
        self.target_markers.append(marker_id)
        return marker_id
    
    def add_solved_marker(self, position, orientation=None, size=0.03, color=[0, 1, 0, 0.8]):
        """添加求解结果标记
        
        Args:
            position: 求解位置 [x, y, z]
            orientation: 求解方向（四元数）
            size: 标记大小
            color: 标记颜色 [r, g, b, a]
        """
        if orientation is None:
            orientation = [0, 0, 0, 1]
            
        # 创建一个半透明的立方体作为求解结果标记
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
        
        self.solved_markers.append(marker_id)
        return marker_id
    
    def clear_markers(self):
        """清除所有标记"""
        for marker_id in self.target_markers + self.solved_markers:
            p.removeBody(marker_id)
        self.target_markers.clear()
        self.solved_markers.clear()
    
    def load_obstacles_from_world_config(self, world_cfg):
        """
        从WorldConfig加载障碍物并在PyBullet中显示
        
        Args:
            world_cfg: WorldConfig对象
            
        Returns:
            list: 创建的障碍物ID列表
        """
        obstacle_ids = []
        
        if hasattr(world_cfg, 'cuboid') and world_cfg.cuboid is not None:
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
    
    def visualize_ik_process(self, target_pose, ik_solutions, success_flags, delay=1.0):
        """可视化IK求解过程
        
        Args:
            target_pose: 目标姿态 (position, quaternion)
            ik_solutions: IK求解结果列表
            success_flags: 成功标记列表
            delay: 每个解的显示延迟
        """
        # 清除之前的标记
        self.clear_markers()
        
        # 添加目标标记
        target_pos, target_quat = target_pose
        self.add_target_marker(target_pos, target_quat)
        
        print(f"\n可视化IK求解过程：")
        print(f"目标位置：{target_pos}")
        print(f"目标姿态：{target_quat}")
        
        # 逐个显示IK解
        for i, (solution, success) in enumerate(zip(ik_solutions, success_flags)):
            if success:
                print(f"\n解 {i+1}: 成功")
                # 设置机器人配置
                extended_solution = self._extend_joint_configuration(solution)
                self.set_joint_angles(extended_solution)
                
                # 获取实际末端执行器位置
                actual_pos, actual_quat = self.get_end_effector_pose()
                if actual_pos is not None:
                    # 添加求解结果标记
                    self.add_solved_marker(actual_pos, actual_quat)
                    
                    # 计算误差
                    pos_error = np.linalg.norm(np.array(actual_pos) - np.array(target_pos))
                    print(f"实际位置：{actual_pos}")
                    print(f"位置误差：{pos_error:.6f}")
                
                p.stepSimulation()
                time.sleep(delay)
            else:
                print(f"\n解 {i+1}: 失败")


def demo_basic_ik_visualization():
    """基础IK可视化演示"""
    print("=== 基础IK可视化演示 ===")
    
    # 设置tensor参数
    tensor_args = TensorDeviceType()
    
    # 加载机器人配置
    config_file = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))
    robot_cfg = RobotConfig.from_dict(config_file["robot_cfg"], tensor_args)
    
    # 创建IK求解器
    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        None,
        rotation_threshold=0.05,
        position_threshold=0.005,
        num_seeds=10,
        self_collision_check=False,
        self_collision_opt=False,
        tensor_args=tensor_args,
        use_cuda_graph=True,
    )
    ik_solver = IKSolver(ik_config)
    
    # 创建可视化器
    visualizer = IKPyBulletVisualizer(gui=True)
    
    try:
        # 生成目标姿态
        print("\n生成随机目标姿态...")
        q_sample = ik_solver.sample_configs(10)
        kin_state = ik_solver.fk(q_sample)
        target_position = kin_state.ee_position[0].cpu().numpy()
        target_quaternion = kin_state.ee_quaternion[0].cpu().numpy()
        
        goal = Pose(kin_state.ee_position, kin_state.ee_quaternion)
        
        print(f"目标位置: {target_position}")
        print(f"目标姿态: {target_quaternion}")
        
        # 求解IK
        print("\n求解IK...")
        st_time = time.time()
        result = ik_solver.solve_batch(goal)
        solve_time = time.time() - st_time
        
        print(f"求解时间: {solve_time:.4f}秒")
        print(f"成功率: {torch.count_nonzero(result.success).item()}/{len(result.success)}")
        
        # 收集成功的解
        successful_solutions = []
        success_flags = []
        
        for i in range(len(result.success)):
            success = result.success[i].item()
            success_flags.append(success)
            
            if success:
                solution = result.js_solution[i].position.cpu().numpy().flatten()
                successful_solutions.append(solution)
                print(f"解 {i+1}: {solution}")
        
        # 可视化IK过程
        print("\n开始可视化IK求解过程...")
        visualizer.visualize_ik_process(
            (target_position, target_quaternion),
            [result.js_solution[i].position.cpu().numpy().flatten() for i in range(len(result.success))],
            success_flags,
            delay=2.0
        )
        
        print("\n演示完成！按回车键退出...")
        input()
        
    finally:
        visualizer.disconnect()


def demo_multiple_targets_ik():
    """多目标IK可视化演示"""
    print("=== 多目标IK可视化演示 ===")
    
    tensor_args = TensorDeviceType()
    
    # 加载机器人配置
    config_file = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))
    robot_cfg = RobotConfig.from_dict(config_file["robot_cfg"], tensor_args)
    
    # 创建IK求解器
    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        None,
        rotation_threshold=0.05,
        position_threshold=0.005,
        num_seeds=20,
        self_collision_check=False,
        self_collision_opt=False,
        tensor_args=tensor_args,
        use_cuda_graph=True,
    )
    ik_solver = IKSolver(ik_config)
    
    # 创建可视化器
    visualizer = IKPyBulletVisualizer(gui=True)
    
    try:
        # 生成多个目标
        num_targets = 5
        print(f"\n生成 {num_targets} 个目标姿态...")
        
        q_samples = ik_solver.sample_configs(num_targets)
        kin_states = ik_solver.fk(q_samples)
        goals = Pose(kin_states.ee_position, kin_states.ee_quaternion)
        
        # 清除标记
        visualizer.clear_markers()
        
        # 添加所有目标标记
        for i in range(num_targets):
            target_pos = kin_states.ee_position[i].cpu().numpy()
            target_quat = kin_states.ee_quaternion[i].cpu().numpy()
            visualizer.add_target_marker(target_pos, target_quat, 
                                       color=[1, 0, 0, 0.6])
            print(f"目标 {i+1}: 位置 {target_pos}")
        
        # 逐个求解并可视化
        for i in range(num_targets):
            print(f"\n求解目标 {i+1}...")
            
            # 单个目标求解
            single_goal = Pose(
                kin_states.ee_position[i:i+1], 
                kin_states.ee_quaternion[i:i+1]
            )
            
            result = ik_solver.solve_batch(single_goal)
            
            if result.success[0]:
                solution = result.js_solution[0].position.cpu().numpy().flatten()
                extended_solution = visualizer._extend_joint_configuration(solution)
                
                print(f"求解成功！关节配置: {solution}")
                
                # 设置机器人配置
                visualizer.set_joint_angles(extended_solution)
                
                # 获取实际位置
                actual_pos, actual_quat = visualizer.get_end_effector_pose()
                if actual_pos is not None:
                    visualizer.add_solved_marker(actual_pos, actual_quat,
                                               color=[0, 1, 0, 0.8])
                    
                    target_pos = kin_states.ee_position[i].cpu().numpy()
                    pos_error = np.linalg.norm(np.array(actual_pos) - target_pos)
                    print(f"位置误差: {pos_error:.6f}")
                
                p.stepSimulation()
                time.sleep(2.0)
            else:
                print("求解失败！")
        
        print("\n所有目标求解完成！按回车键退出...")
        input()
        
    finally:
        visualizer.disconnect()


def demo_collision_aware_ik():
    """避障IK可视化演示"""
    print("=== 避障IK可视化演示 ===")
    
    tensor_args = TensorDeviceType()
    
    # 加载机器人和世界配置
    robot_file = "franka.yml"
    world_file = "collision_cage.yml"
    
    robot_cfg = RobotConfig.from_dict(
        load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
    )
    world_cfg = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), world_file))
    )
    
    # 创建避障IK求解器
    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        rotation_threshold=0.05,
        position_threshold=0.005,
        num_seeds=50,
        self_collision_check=True,
        self_collision_opt=True,
        tensor_args=tensor_args,
        use_cuda_graph=True,
    )
    ik_solver = IKSolver(ik_config)
    
    # 创建可视化器
    visualizer = IKPyBulletVisualizer(gui=True)
    
    try:
        # 加载障碍物可视化
        print("\n加载障碍物...")
        obstacle_ids = visualizer.load_obstacles_from_world_config(world_cfg)
        print(f"成功加载 {len(obstacle_ids)} 个障碍物")
        
        print("\n这个演示展示避障IK，红色半透明物体是障碍物")
        print("IK求解器会避开这些障碍物")
        
        # 生成多个目标
        num_targets = 10
        print(f"\n生成 {num_targets} 个目标姿态...")
        
        q_samples = ik_solver.sample_configs(num_targets)
        kin_states = ik_solver.fk(q_samples)
        
        # 清除标记
        visualizer.clear_markers()
        
        # 添加所有目标标记
        for i in range(num_targets):
            target_pos = kin_states.ee_position[i].cpu().numpy()
            target_quat = kin_states.ee_quaternion[i].cpu().numpy()
            visualizer.add_target_marker(target_pos, target_quat, 
                                       color=[1, 0, 0, 0.6])
            print(f"目标 {i+1}: 位置 {target_pos}")
        
        # 逐个求解并可视化避障IK
        total_success = 0
        for i in range(num_targets):
            print(f"\n求解避障IK目标 {i+1}...")
            
            # 单个目标求解
            single_goal = Pose(
                kin_states.ee_position[i:i+1], 
                kin_states.ee_quaternion[i:i+1]
            )
            
            st_time = time.time()
            result = ik_solver.solve_batch(single_goal)
            solve_time = time.time() - st_time
            
            if result.success[0]:
                total_success += 1
                solution = result.js_solution[0].position.cpu().numpy().flatten()
                extended_solution = visualizer._extend_joint_configuration(solution)
                
                print(f"求解成功！避障解: {solution}")
                print(f"求解时间: {solve_time:.4f}秒")
                
                # 设置机器人配置
                visualizer.set_joint_angles(extended_solution)
                
                # 获取实际位置
                actual_pos, actual_quat = visualizer.get_end_effector_pose()
                if actual_pos is not None:
                    visualizer.add_solved_marker(actual_pos, actual_quat,
                                               color=[0, 1, 0, 0.8])
                    
                    target_pos = kin_states.ee_position[i].cpu().numpy()
                    pos_error = np.linalg.norm(np.array(actual_pos) - target_pos)
                    print(f"位置误差: {pos_error:.6f}")
                
                p.stepSimulation()
                time.sleep(2.0)
            else:
                print("求解失败！")
        
        print(f"\n所有目标求解完成！总成功率: {total_success}/{num_targets}")
        
        p.stepSimulation()
        print("\n演示完成！按回车键退出...")
        input()
        
    finally:
        visualizer.disconnect()


def main():
    """主函数"""
    print("欢迎使用PyBullet IK可视化演示！")
    print("\n可用的演示：")
    print("1. 基础IK可视化")
    print("2. 多目标IK可视化")
    print("3. 避障IK可视化")
    
    choice = input("\n请选择演示 (1-3): ").strip()
    
    if choice == "1":
        demo_basic_ik_visualization()
    elif choice == "2":
        demo_multiple_targets_ik()
    elif choice == "3":
        demo_collision_aware_ik()
    else:
        print("无效选择，运行基础演示...")
        demo_basic_ik_visualization()


if __name__ == "__main__":
    main() 