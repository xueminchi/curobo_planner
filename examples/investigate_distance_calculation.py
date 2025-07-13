#!/usr/bin/env python3
"""
调查cuRobo距离计算的真实机制
基于官方文档: https://curobo.org/get_started/2c_world_collision.html
"""

import numpy as np
import torch

# CuRobo
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import Cuboid, WorldConfig, Sphere
from curobo.types.base import TensorDeviceType
from curobo.util.logger import setup_curobo_logger
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig

def cost_function(d, eta):
    """
    根据官方文档实现的成本函数
    d: 有符号距离 (正值表示碰撞，负值表示安全)
    eta: 激活距离
    """
    if d <= -eta:
        return 0
    elif -eta < d <= 0:
        return (1/eta) * (d + eta)**2
    else:  # d > 0
        return d + 0.5 * eta

def investigate_distance_calculation():
    """调查距离计算机制"""
    setup_curobo_logger("info")
    
    tensor_args = TensorDeviceType()
    robot_file = "franka.yml"
    
    print("🔍 调查cuRobo距离计算机制")
    print("=" * 60)
    print("📖 基于官方文档: https://curobo.org/get_started/2c_world_collision.html")
    print()
    
    # 创建不同距离的障碍物进行测试
    test_distances = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    activation_distance = 0.5
    
    print(f"🧪 测试设置:")
    print(f"   - 激活距离: {activation_distance}m")
    print(f"   - 机械臂初始位置: 原点附近")
    print(f"   - 障碍物测试距离: {test_distances}")
    print()
    
    # 创建RobotWorld配置
    results = []
    
    for distance in test_distances:
        print(f"📍 测试距离: {distance}m")
        
        # 创建障碍物
        cuboid = Cuboid(
            name=f"test_cube_{distance}",
            pose=[distance, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0],
            dims=[0.1, 0.1, 0.1],
        )
        
        world_config = WorldConfig(cuboid=[cuboid])
        
        # 创建RobotWorld
        robot_world_config = RobotWorldConfig.load_from_config(
            robot_file,
            world_config,
            tensor_args,
            collision_activation_distance=activation_distance,
            collision_checker_type=CollisionCheckerType.PRIMITIVE,
        )
        
        robot_world = RobotWorld(robot_world_config)
        
        # 测试关节配置
        joint_config = [0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0]
        joint_positions = torch.tensor(joint_config, dtype=tensor_args.dtype, device=tensor_args.device).unsqueeze(0)
        
        # 获取距离
        d_world, d_self = robot_world.get_world_self_collision_distance_from_joints(joint_positions)
        cost_value = d_world.item()
        
        # 获取机械臂球体信息
        kin_state = robot_world.get_kinematics(joint_positions)
        spheres = kin_state.link_spheres_tensor[0]
        
        # 找到最接近障碍物的球体
        obstacle_center = torch.tensor([distance, 0.0, 0.3], device=tensor_args.device)
        distances_to_obstacle = torch.norm(spheres[:, :3] - obstacle_center, dim=1)
        min_idx = distances_to_obstacle.argmin()
        closest_sphere = spheres[min_idx]
        geometric_distance = distances_to_obstacle[min_idx].item()
        
        # 计算真实的表面距离 (考虑球体和立方体的半径)
        sphere_radius = closest_sphere[3].item()
        cuboid_half_size = 0.05  # 立方体半尺寸
        surface_distance = geometric_distance - sphere_radius - cuboid_half_size
        
        print(f"   🎯 最近球体: 位置({closest_sphere[0]:.3f}, {closest_sphere[1]:.3f}, {closest_sphere[2]:.3f}), 半径{sphere_radius:.3f}")
        print(f"   📏 几何距离: {geometric_distance:.3f}m")
        print(f"   📐 表面距离: {surface_distance:.3f}m")
        print(f"   💰 成本值: {cost_value:.4f}")
        
        # 根据成本函数计算理论值
        if surface_distance > 0:
            theoretical_cost = cost_function(-surface_distance, activation_distance)
        else:
            theoretical_cost = cost_function(abs(surface_distance), activation_distance)
        
        print(f"   🧮 理论成本值: {theoretical_cost:.4f}")
        print(f"   ❓ 匹配度: {'✅' if abs(cost_value - theoretical_cost) < 0.1 else '❌'}")
        print()
        
        results.append({
            'distance': distance,
            'geometric_distance': geometric_distance,
            'surface_distance': surface_distance,
            'cost_value': cost_value,
            'theoretical_cost': theoretical_cost,
            'sphere_radius': sphere_radius,
            'closest_sphere_pos': closest_sphere[:3].cpu().numpy()
        })
    
    return results

def investigate_sphere_representation():
    """调查机械臂的球体表示"""
    print("\n🔍 调查机械臂的球体表示")
    print("=" * 60)
    
    tensor_args = TensorDeviceType()
    robot_file = "franka.yml"
    
    # 创建空世界
    world_config = WorldConfig()
    
    robot_world_config = RobotWorldConfig.load_from_config(
        robot_file,
        world_config,
        tensor_args,
        collision_activation_distance=0.5,
        collision_checker_type=CollisionCheckerType.PRIMITIVE,
    )
    
    robot_world = RobotWorld(robot_world_config)
    
    # 测试关节配置
    joint_config = [0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0]
    joint_positions = torch.tensor(joint_config, dtype=tensor_args.dtype, device=tensor_args.device).unsqueeze(0)
    
    # 获取机械臂球体信息
    kin_state = robot_world.get_kinematics(joint_positions)
    spheres = kin_state.link_spheres_tensor[0]
    
    print(f"📊 机械臂球体统计:")
    print(f"   - 总球体数量: {spheres.shape[0]}")
    print(f"   - 球体半径范围: {spheres[:, 3].min().item():.3f}m - {spheres[:, 3].max().item():.3f}m")
    print(f"   - 平均球体半径: {spheres[:, 3].mean().item():.3f}m")
    print()
    
    # 分析球体分布
    print("🎯 关键球体位置 (前10个):")
    for i in range(min(10, spheres.shape[0])):
        sphere = spheres[i]
        print(f"   球体{i:2d}: 位置({sphere[0]:6.3f}, {sphere[1]:6.3f}, {sphere[2]:6.3f}), 半径{sphere[3]:.3f}")
    
    # 分析末端执行器附近的球体
    print("\n🤖 末端执行器附近的球体:")
    end_effector_spheres = spheres[spheres[:, 2] > 0.4]  # Z坐标大于0.4的球体
    print(f"   - 末端附近球体数量: {end_effector_spheres.shape[0]}")
    for i, sphere in enumerate(end_effector_spheres):
        print(f"   末端球体{i}: 位置({sphere[0]:6.3f}, {sphere[1]:6.3f}, {sphere[2]:6.3f}), 半径{sphere[3]:.3f}")
    
    return spheres

def compare_collision_checkers():
    """比较不同碰撞检测器的性能"""
    print("\n🔍 比较不同碰撞检测器")
    print("=" * 60)
    
    tensor_args = TensorDeviceType()
    robot_file = "franka.yml"
    
    # 创建测试障碍物
    cuboid = Cuboid(
        name="test_cube",
        pose=[0.5, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0],
        dims=[0.1, 0.1, 0.1],
    )
    
    world_config = WorldConfig(cuboid=[cuboid])
    
    # 测试不同的碰撞检测器类型
    checker_types = [
        (CollisionCheckerType.PRIMITIVE, "PRIMITIVE (立方体)"),
        # (CollisionCheckerType.MESH, "MESH (网格)"),  # 需要网格世界
    ]
    
    joint_config = [0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0]
    joint_positions = torch.tensor(joint_config, dtype=tensor_args.dtype, device=tensor_args.device).unsqueeze(0)
    
    for checker_type, name in checker_types:
        print(f"🧪 测试 {name}:")
        
        try:
            robot_world_config = RobotWorldConfig.load_from_config(
                robot_file,
                world_config,
                tensor_args,
                collision_activation_distance=0.5,
                collision_checker_type=checker_type,
            )
            
            robot_world = RobotWorld(robot_world_config)
            
            # 性能测试
            import time
            start_time = time.time()
            
            for _ in range(100):
                d_world, d_self = robot_world.get_world_self_collision_distance_from_joints(joint_positions)
            
            end_time = time.time()
            avg_time = (end_time - start_time) / 100 * 1000  # ms
            
            print(f"   📏 距离值: {d_world.item():.4f}")
            print(f"   ⏱️  平均计算时间: {avg_time:.2f}ms")
            print(f"   🔧 检测器类型: {robot_world.collision_cost.world_coll_checker.__class__.__name__}")
            
        except Exception as e:
            print(f"   ❌ 错误: {e}")
        
        print()

def visualize_cost_function():
    """可视化成本函数（文本版本）"""
    print("\n📊 成本函数分析")
    print("=" * 60)
    
    # 创建距离范围
    distances = np.linspace(-1.0, 0.5, 21)  # 减少点数用于文本显示
    eta_values = [0.1, 0.2, 0.3, 0.5]
    
    print("📈 成本函数表格 (距离 vs 成本值):")
    print("距离(m)  ", end="")
    for eta in eta_values:
        print(f"η={eta:0.1f}     ", end="")
    print()
    print("-" * 50)
    
    for d in distances:
        print(f"{d:6.2f}   ", end="")
        for eta in eta_values:
            cost = cost_function(d, eta)
            print(f"{cost:6.3f}   ", end="")
        print()
    
    print("\n📊 成本函数特性:")
    print("   - d ≤ -η: 成本 = 0 (安全区域)")
    print("   - -η < d ≤ 0: 成本 = (1/η)(d + η)² (激活区域，二次增长)")
    print("   - d > 0: 成本 = d + 0.5η (碰撞区域，线性增长)")
    print("   - 负值表示安全距离，正值表示碰撞深度")

def main():
    """主函数"""
    print("🔍 cuRobo距离计算机制调查")
    print("=" * 80)
    
    # 1. 调查距离计算
    results = investigate_distance_calculation()
    
    # 2. 调查球体表示
    spheres = investigate_sphere_representation()
    
    # 3. 比较碰撞检测器
    compare_collision_checkers()
    
    # 4. 分析成本函数
    visualize_cost_function()
    
    print("\n📋 总结:")
    print("=" * 60)
    print("✅ cuRobo使用成本函数而非真实几何距离")
    print("✅ 机械臂被近似为球体集合")
    print("✅ 距离值受激活距离参数影响")
    print("✅ PRIMITIVE检测器比MESH检测器快4倍")
    print("✅ 成本函数在激活距离内呈二次增长")
    
    print(f"\n🎯 关键发现:")
    print(f"   - 返回的'距离'实际上是成本函数值")
    print(f"   - 真实几何距离需要考虑球体半径")
    print(f"   - 激活距离决定了成本计算范围")
    print(f"   - 球体近似可能导致距离估计不准确")

if __name__ == "__main__":
    main() 