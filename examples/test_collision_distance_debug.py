#!/usr/bin/env python3
"""
深度调试碰撞距离计算问题
"""

import numpy as np
import torch

# CuRobo
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import Cuboid, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.util.logger import setup_curobo_logger
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig

def debug_collision_distance():
    """深度调试碰撞距离计算"""
    setup_curobo_logger("info")
    
    tensor_args = TensorDeviceType()
    robot_file = "franka.yml"
    
    # 创建立方体障碍物
    cuboid = Cuboid(
        name="debug_cube",
        pose=[0.5, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0],
        dims=[0.1, 0.1, 0.1],
    )
    
    world_config = WorldConfig(cuboid=[cuboid])
    
    print("🔍 调试信息:")
    print(f"   障碍物位置: {cuboid.pose[:3]}")
    print(f"   障碍物尺寸: {cuboid.dims}")
    
    # 尝试不同的collision_activation_distance值
    activation_distances = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]
    
    for activation_dist in activation_distances:
        print(f"\n🧪 测试 collision_activation_distance = {activation_dist}")
        print("="*50)
        
        try:
            # 创建RobotWorld配置
            robot_world_config = RobotWorldConfig.load_from_config(
                robot_file,
                world_config,
                tensor_args,
                collision_activation_distance=activation_dist,
                collision_checker_type=CollisionCheckerType.PRIMITIVE,
            )
            
            robot_world = RobotWorld(robot_world_config)
            
            # 测试关节配置
            joint_config = [0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0]
            joint_positions = torch.tensor(joint_config, dtype=tensor_args.dtype, device=tensor_args.device).unsqueeze(0)
            
            # 获取机械臂状态
            kin_state = robot_world.get_kinematics(joint_positions)
            print(f"   机械臂球体数量: {kin_state.link_spheres_tensor.shape[1] if kin_state.link_spheres_tensor is not None else 'None'}")
            
            if kin_state.link_spheres_tensor is not None:
                # 打印前几个球体的位置
                spheres = kin_state.link_spheres_tensor[0]  # 取第一个batch
                print(f"   前5个球体位置和半径:")
                for i in range(min(5, spheres.shape[0])):
                    sphere = spheres[i]
                    print(f"     球体{i}: 位置({sphere[0]:.3f}, {sphere[1]:.3f}, {sphere[2]:.3f}), 半径{sphere[3]:.3f}")
                
                # 计算到障碍物的距离
                robot_spheres = kin_state.link_spheres_tensor.unsqueeze(1)
                
                # 方法1: get_world_self_collision_distance_from_joints
                d_world, d_self = robot_world.get_world_self_collision_distance_from_joints(joint_positions)
                print(f"   方法1 - 世界距离: {d_world.item():.4f}m, 自碰撞: {d_self.item():.4f}m")
                
                # 方法2: get_collision_vector
                d_world_vec, d_world_gradient = robot_world.get_collision_vector(robot_spheres)
                print(f"   方法2 - 世界距离: {d_world_vec.item():.4f}m")
                
                # 检查collision_cost配置
                if robot_world.collision_cost is not None:
                    print(f"   collision_cost配置:")
                    print(f"     activation_distance: {robot_world.collision_cost.activation_distance}")
                    print(f"     weight: {robot_world.collision_cost.weight}")
                    
                    # 直接调用collision_cost
                    cost = robot_world.collision_cost.forward(robot_spheres)
                    print(f"     直接成本: {cost.item():.4f}")
                    
                    # 检查world_coll_checker
                    if hasattr(robot_world.collision_cost, 'world_coll_checker'):
                        world_checker = robot_world.collision_cost.world_coll_checker
                        print(f"     world_checker类型: {type(world_checker)}")
                        print(f"     world_checker.max_distance: {world_checker.max_distance}")
                        
                        # 计算到障碍物中心的距离
                        obstacle_center = torch.tensor([0.5, 0.0, 0.3], device=tensor_args.device)
                        sphere_center = spheres[0][:3]  # 取第一个球体
                        geometric_distance = torch.norm(sphere_center - obstacle_center).item()
                        print(f"     几何距离到障碍物中心: {geometric_distance:.4f}m")
                        print(f"     几何距离到障碍物表面: {geometric_distance - 0.05:.4f}m")  # 障碍物半径0.05m
                        
        except Exception as e:
            print(f"   配置错误: {e}")

def main():
    """主函数"""
    print("🔍 深度调试碰撞距离计算")
    print("="*60)
    
    debug_collision_distance()
    
    print("\n✅ 调试完成！")

if __name__ == "__main__":
    main() 