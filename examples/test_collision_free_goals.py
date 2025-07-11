#!/usr/bin/env python3
"""
测试collision-free目标生成功能
"""

import sys
sys.path.append('.')

import numpy as np
from motion_gen_scene_selector import SceneMotionGenVisualizer, get_available_world_configs
from curobo.geom.types import WorldConfig
from curobo.util_file import get_world_configs_path, join_path, load_yaml

def test_collision_free_goal_generation():
    """测试无碰撞目标生成功能"""
    print("🧪 测试collision-free目标生成功能")
    print("=" * 50)
    
    # 创建可视化器
    visualizer = SceneMotionGenVisualizer(gui=False)
    
    # 测试多个世界配置
    test_configs = ["collision_table.yml", "collision_cage.yml", "collision_primitives_3d.yml"]
    
    for config_file in test_configs:
        print(f"\n📁 测试配置: {config_file}")
        
        try:
            # 加载世界配置
            world_cfg_dict = load_yaml(join_path(get_world_configs_path(), config_file))
            world_cfg = WorldConfig.from_dict(world_cfg_dict)
            
            # 统计障碍物数量
            obstacle_count = 0
            if hasattr(world_cfg, 'cuboid') and world_cfg.cuboid is not None:
                obstacle_count += len(world_cfg.cuboid)
                print(f"  - 立方体障碍物: {len(world_cfg.cuboid)} 个")
            
            if hasattr(world_cfg, 'sphere') and world_cfg.sphere is not None:
                obstacle_count += len(world_cfg.sphere)
                print(f"  - 球体障碍物: {len(world_cfg.sphere)} 个")
            
            if hasattr(world_cfg, 'capsule') and world_cfg.capsule is not None:
                obstacle_count += len(world_cfg.capsule)
                print(f"  - 胶囊体障碍物: {len(world_cfg.capsule)} 个")
            
            print(f"  📊 总障碍物数量: {obstacle_count}")
            
            # 测试目标生成
            print(f"\n🎯 生成无碰撞目标...")
            
            successful_goals = 0
            total_attempts = 10
            
            for i in range(total_attempts):
                goal = visualizer.generate_collision_free_goal(world_cfg, max_attempts=20)
                if goal is not None:
                    successful_goals += 1
                    print(f"  ✅ 目标 {i+1}: {goal[:3]}")
                else:
                    print(f"  ❌ 目标 {i+1}: 生成失败")
            
            success_rate = successful_goals / total_attempts * 100
            print(f"\n📈 成功率: {successful_goals}/{total_attempts} ({success_rate:.1f}%)")
            
            if success_rate >= 80:
                print(f"✅ {config_file} 测试通过")
            else:
                print(f"⚠️  {config_file} 成功率较低")
                
        except Exception as e:
            print(f"❌ {config_file} 测试失败: {e}")
    
    visualizer.disconnect()
    print(f"\n🎉 collision-free目标生成测试完成！")


def test_collision_detection_functions():
    """测试碰撞检测函数"""
    print("\n🧪 测试碰撞检测函数")
    print("=" * 50)
    
    visualizer = SceneMotionGenVisualizer(gui=False)
    
    # 测试立方体碰撞检测
    print("\n📦 测试立方体碰撞检测...")
    
    # 创建一个测试立方体 (位置[0,0,0], 尺寸[1,1,1])
    class TestCuboid:
        def __init__(self):
            self.pose = [0, 0, 0, 1, 0, 0, 0]  # x,y,z,qw,qx,qy,qz
            self.dims = [1.0, 1.0, 1.0]  # x,y,z dimensions
    
    test_cuboid = TestCuboid()
    
    # 测试点
    test_points = [
        ([0, 0, 0], True, "中心点"),
        ([0.4, 0.4, 0.4], True, "内部点"),
        ([0.6, 0, 0], False, "边界外点"),
        ([1.0, 0, 0], False, "边界上点"),
        ([0.5, 0.5, 0.5], True, "边界内点"),
    ]
    
    for point, expected, description in test_points:
        result = visualizer._check_point_cuboid_collision(
            np.array(point), test_cuboid, safety_margin=0.1
        )
        status = "✅" if result == expected else "❌"
        print(f"  {status} {description}: {point} -> {result} (期望: {expected})")
    
    # 测试球体碰撞检测
    print("\n🌕 测试球体碰撞检测...")
    
    class TestSphere:
        def __init__(self):
            self.position = [0, 0, 0]
            self.radius = 0.5
    
    test_sphere = TestSphere()
    
    test_points = [
        ([0, 0, 0], True, "中心点"),
        ([0.3, 0, 0], True, "内部点"),
        ([0.7, 0, 0], False, "外部点"),
        ([0.5, 0, 0], True, "边界点(含安全距离)"),
    ]
    
    for point, expected, description in test_points:
        result = visualizer._check_point_sphere_collision(
            np.array(point), test_sphere, safety_margin=0.1
        )
        status = "✅" if result == expected else "❌"
        print(f"  {status} {description}: {point} -> {result} (期望: {expected})")
    
    visualizer.disconnect()
    print(f"\n🎉 碰撞检测函数测试完成！")


def test_workspace_bounds():
    """测试工作空间边界"""
    print("\n🧪 测试工作空间边界")
    print("=" * 50)
    
    visualizer = SceneMotionGenVisualizer(gui=False)
    
    # 创建一个空的世界配置(无障碍物)
    empty_world_dict = {"cuboid": [], "sphere": [], "capsule": []}
    empty_world_cfg = WorldConfig.from_dict(empty_world_dict)
    
    print("📐 测试工作空间边界限制...")
    
    goals_generated = []
    for i in range(20):
        goal = visualizer.generate_collision_free_goal(empty_world_cfg, max_attempts=10)
        if goal is not None:
            goals_generated.append(goal[:3])  # 只要位置信息
    
    if goals_generated:
        goals_array = np.array(goals_generated)
        
        print(f"  生成了 {len(goals_generated)} 个目标")
        print(f"  X 范围: [{goals_array[:, 0].min():.3f}, {goals_array[:, 0].max():.3f}]")
        print(f"  Y 范围: [{goals_array[:, 1].min():.3f}, {goals_array[:, 1].max():.3f}]")
        print(f"  Z 范围: [{goals_array[:, 2].min():.3f}, {goals_array[:, 2].max():.3f}]")
        
        # 检查是否在预期范围内
        x_in_range = (goals_array[:, 0] >= 0.2) & (goals_array[:, 0] <= 0.7)
        y_in_range = (goals_array[:, 1] >= -0.5) & (goals_array[:, 1] <= 0.5)
        z_in_range = (goals_array[:, 2] >= 0.3) & (goals_array[:, 2] <= 0.8)
        
        all_in_range = x_in_range & y_in_range & z_in_range
        
        if np.all(all_in_range):
            print("  ✅ 所有目标都在工作空间范围内")
        else:
            print(f"  ⚠️  {np.sum(~all_in_range)} 个目标超出工作空间范围")
    else:
        print("  ❌ 未能生成任何目标")
    
    visualizer.disconnect()
    print(f"\n🎉 工作空间边界测试完成！")


if __name__ == "__main__":
    test_collision_free_goal_generation()
    test_collision_detection_functions()
    test_workspace_bounds() 