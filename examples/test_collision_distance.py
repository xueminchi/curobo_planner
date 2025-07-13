#!/usr/bin/env python3
"""
测试机械臂与障碍物之间的距离计算
专门用于调试碰撞距离计算问题
"""

import numpy as np
import torch

# CuRobo
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import Cuboid, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.util.logger import setup_curobo_logger
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig

def create_simple_world():
    """创建一个简单的世界，只包含一个立方体障碍物"""
    # 创建一个立方体障碍物，放在机械臂前方
    cuboid = Cuboid(
        name="test_cube",
        pose=[0.5, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0],  # 机械臂前方50cm处，无旋转
        dims=[0.1, 0.1, 0.1],  # 10cm x 10cm x 10cm的立方体
    )
    
    world_config = WorldConfig(cuboid=[cuboid])
    print(f"📦 创建了简单世界，包含一个立方体障碍物:")
    if cuboid.pose is not None:
        print(f"   位置: {cuboid.pose[:3]}")
    print(f"   尺寸: {cuboid.dims}")
    
    return world_config

def test_collision_distance():
    """测试碰撞距离计算"""
    setup_curobo_logger("info")
    
    tensor_args = TensorDeviceType()
    robot_file = "franka.yml"
    
    # 创建简单世界
    world_config = create_simple_world()
    
    print("\n🔧 初始化RobotWorld...")
    
    # 创建RobotWorld配置
    robot_world_config = RobotWorldConfig.load_from_config(
        robot_file,
        world_config,
        tensor_args,
        collision_activation_distance=0.1,  # 10cm激活距离
        collision_checker_type=CollisionCheckerType.PRIMITIVE,
    )
    
    robot_world = RobotWorld(robot_world_config)
    
    # 测试不同的关节配置
    test_configs = [
        # 配置1: 机械臂初始位置
        [0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0],
        # 配置2: 机械臂向前伸展（更接近障碍物）
        [0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0],
        # 配置3: 机械臂向上抬起
        [0.0, -0.5, 0.0, -1.5, 0.0, 1.0, 0.0],
        # 配置4: 机械臂向右移动
        [0.5, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0],
    ]
    
    print("\n🧪 开始测试不同的关节配置...")
    print("="*60)
    
    for i, config in enumerate(test_configs):
        print(f"\n📍 测试配置 {i+1}: {config}")
        
        # 转换为tensor
        joint_positions = torch.tensor(config, dtype=tensor_args.dtype, device=tensor_args.device).unsqueeze(0)
        
        # 方法1: 使用get_world_self_collision_distance_from_joints
        print("\n🔍 方法1: get_world_self_collision_distance_from_joints")
        try:
            d_world, d_self = robot_world.get_world_self_collision_distance_from_joints(joint_positions)
            print(f"   世界碰撞距离: {d_world}")
            print(f"   自碰撞距离: {d_self}")
            if hasattr(d_world, 'min'):
                print(f"   世界最小距离: {d_world.min().item():.4f}m")
            if hasattr(d_self, 'min'):
                print(f"   自碰撞最小距离: {d_self.min().item():.4f}m")
        except Exception as e:
            print(f"   错误: {e}")
        
        # 方法2: 使用get_collision_vector
        print("\n🔍 方法2: get_collision_vector")
        try:
            # 获取机械臂球体表示
            kin_state = robot_world.get_kinematics(joint_positions)
            if kin_state.link_spheres_tensor is not None:
                robot_spheres = kin_state.link_spheres_tensor.unsqueeze(1)  # 添加时间维度
                
                # 获取碰撞向量
                d_world_vec, d_world_gradient = robot_world.get_collision_vector(robot_spheres)
                d_self_vec = robot_world.get_self_collision_distance(robot_spheres)
                
                print(f"   世界碰撞向量距离: {d_world_vec}")
                print(f"   自碰撞向量距离: {d_self_vec}")
                if hasattr(d_world_vec, 'min'):
                    print(f"   世界最小距离: {d_world_vec.min().item():.4f}m")
                if hasattr(d_self_vec, 'min'):
                    print(f"   自碰撞最小距离: {d_self_vec.min().item():.4f}m")
                
                # 打印梯度信息
                if d_world_gradient is not None:
                    print(f"   梯度形状: {d_world_gradient.shape}")
                    print(f"   梯度范数: {torch.norm(d_world_gradient).item():.4f}")
            
        except Exception as e:
            print(f"   错误: {e}")
        
        # 方法3: 使用get_collision_distance
        print("\n🔍 方法3: get_collision_distance")
        try:
            # 获取机械臂球体表示
            kin_state = robot_world.get_kinematics(joint_positions)
            if kin_state.link_spheres_tensor is not None:
                robot_spheres = kin_state.link_spheres_tensor.unsqueeze(1)  # 添加时间维度
                
                # 获取碰撞距离
                d_world_dist = robot_world.get_collision_distance(robot_spheres)
                d_self_dist = robot_world.get_self_collision_distance(robot_spheres)
                
                print(f"   世界碰撞距离: {d_world_dist}")
                print(f"   自碰撞距离: {d_self_dist}")
                if hasattr(d_world_dist, 'min'):
                    print(f"   世界最小距离: {d_world_dist.min().item():.4f}m")
                if hasattr(d_self_dist, 'min'):
                    print(f"   自碰撞最小距离: {d_self_dist.min().item():.4f}m")
            
        except Exception as e:
            print(f"   错误: {e}")
        
        # 方法4: 直接使用PrimitiveCollisionCost
        print("\n🔍 方法4: 直接使用PrimitiveCollisionCost")
        try:
            if robot_world.collision_cost is not None:
                kin_state = robot_world.get_kinematics(joint_positions)
                if kin_state.link_spheres_tensor is not None:
                    robot_spheres = kin_state.link_spheres_tensor.unsqueeze(1)  # 添加时间维度
                    
                    # 直接调用collision_cost
                    cost = robot_world.collision_cost.forward(robot_spheres)
                    print(f"   碰撞成本: {cost}")
                    if hasattr(cost, 'min'):
                        print(f"   最小成本: {cost.min().item():.4f}")
                    else:
                        print(f"   成本值: {cost}")
                        
        except Exception as e:
            print(f"   错误: {e}")
        
        print("-" * 60)

def main():
    """主函数"""
    print("🧪 碰撞距离计算测试")
    print("="*60)
    
    test_collision_distance()
    
    print("\n✅ 测试完成！")

if __name__ == "__main__":
    main() 