#!/usr/bin/env python3
"""
演示collision-free目标生成功能在多目标规划中的应用
"""

import sys
sys.path.append('.')

from motion_gen_scene_selector import SceneMotionGenVisualizer
from curobo.geom.types import WorldConfig
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.util.logger import setup_curobo_logger
from curobo.util_file import get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig


def demo_collision_free_goal_generation():
    """演示collision-free目标生成功能"""
    print("🎯 Collision-Free目标生成演示")
    print("=" * 60)
    
    setup_curobo_logger("error")
    tensor_args = TensorDeviceType()
    robot_file = "franka.yml"
    world_file = "collision_primitives_3d.yml"  # 包含多种障碍物的复杂场景
    
    # 创建可视化器
    visualizer = SceneMotionGenVisualizer(gui=False)
    
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
    
    try:
        # 加载世界配置
        world_cfg = WorldConfig.from_dict(
            load_yaml(join_path(get_world_configs_path(), world_file))
        )
        
        print(f"\n🌍 加载世界配置: {world_file}")
        
        # 统计障碍物
        total_obstacles = 0
        if hasattr(world_cfg, 'cuboid') and world_cfg.cuboid is not None:
            total_obstacles += len(world_cfg.cuboid)
            print(f"  📦 立方体: {len(world_cfg.cuboid)} 个")
        if hasattr(world_cfg, 'sphere') and world_cfg.sphere is not None:
            total_obstacles += len(world_cfg.sphere)
            print(f"  🌕 球体: {len(world_cfg.sphere)} 个")
        if hasattr(world_cfg, 'capsule') and world_cfg.capsule is not None:
            total_obstacles += len(world_cfg.capsule)
            print(f"  💊 胶囊体: {len(world_cfg.capsule)} 个")
        
        print(f"  📊 总计: {total_obstacles} 个障碍物")
        
        # 获取起始状态
        retract_cfg = motion_gen.get_retract_config()
        current_state = JointState.from_position(retract_cfg.view(1, -1))
        
        # 测试固定目标 vs collision-free目标的差异
        print(f"\n🔍 测试固定目标 vs Collision-Free目标")
        print("-" * 40)
        
        # 1. 测试一些可能导致碰撞的固定目标
        problematic_goals = [
            [0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0],    # 可能与障碍物碰撞
            [0.6, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0],    # 可能与障碍物碰撞
            [0.3, 0.3, 0.3, 1.0, 0.0, 0.0, 0.0],    # 可能与障碍物碰撞
        ]
        
        failed_goals = 0
        successful_goals = 0
        
        print(f"\n📌 测试固定目标 (可能导致碰撞):")
        for i, goal_pos in enumerate(problematic_goals, 1):
            goal_pose = Pose.from_list(goal_pos)
            
            result = motion_gen.plan_single(
                current_state, 
                goal_pose, 
                MotionGenPlanConfig(max_attempts=2)
            )
            
            if result.success is not None and (result.success.item() if hasattr(result.success, 'item') else result.success):
                print(f"  ✅ 固定目标 {i}: {goal_pos[:3]} - 规划成功")
                successful_goals += 1
            else:
                print(f"  ❌ 固定目标 {i}: {goal_pos[:3]} - 规划失败 ({result.status})")
                failed_goals += 1
        
        print(f"\n📊 固定目标结果: {successful_goals} 成功, {failed_goals} 失败")
        
        # 2. 测试collision-free目标生成
        print(f"\n🎯 测试Collision-Free目标生成:")
        
        collision_free_successful = 0
        for i in range(5):
            # 生成collision-free目标
            goal_pos = visualizer.generate_collision_free_goal(world_cfg)
            
            if goal_pos is None:
                print(f"  ❌ Collision-Free目标 {i+1}: 生成失败")
                continue
            
            goal_pose = Pose.from_list(goal_pos)
            
            result = motion_gen.plan_single(
                current_state, 
                goal_pose, 
                MotionGenPlanConfig(max_attempts=2)
            )
            
            if result.success is not None and (result.success.item() if hasattr(result.success, 'item') else result.success):
                print(f"  ✅ Collision-Free目标 {i+1}: {goal_pos[:3]} - 规划成功")
                collision_free_successful += 1
            else:
                print(f"  ⚠️  Collision-Free目标 {i+1}: {goal_pos[:3]} - 规划失败 ({result.status})")
        
        print(f"\n📊 Collision-Free目标结果: {collision_free_successful}/5 成功")
        
        # 3. 性能对比
        print(f"\n📈 性能对比:")
        fixed_success_rate = successful_goals / len(problematic_goals) * 100
        cf_success_rate = collision_free_successful / 5 * 100
        
        print(f"  📌 固定目标成功率: {fixed_success_rate:.1f}% ({successful_goals}/{len(problematic_goals)})")
        print(f"  🎯 Collision-Free目标成功率: {cf_success_rate:.1f}% ({collision_free_successful}/5)")
        
        improvement = cf_success_rate - fixed_success_rate
        if improvement > 0:
            print(f"  📊 性能提升: +{improvement:.1f}%")
            print(f"  🎉 Collision-Free目标生成显著提高了规划成功率！")
        else:
            print(f"  📊 性能差异: {improvement:.1f}%")
        
        # 4. 演示重试机制
        print(f"\n🔄 演示智能重试机制:")
        print("当规划失败时，自动生成新的collision-free目标...")
        
        # 使用一个肯定会失败的目标来触发重试
        impossible_goal = [0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0]  # 非常低的高度，可能碰撞
        goal_pose = Pose.from_list(impossible_goal)
        
        result = motion_gen.plan_single(
            current_state, 
            goal_pose, 
            MotionGenPlanConfig(max_attempts=1)
        )
        
        if not (result.success is not None and (result.success.item() if hasattr(result.success, 'item') else result.success)):
            print(f"  ❌ 原目标失败: {impossible_goal[:3]}")
            print(f"  🔄 启动重试机制...")
            
            for retry in range(3):
                new_goal = visualizer.generate_collision_free_goal(world_cfg)
                if new_goal is None:
                    print(f"    ⚠️  重试 {retry+1}: 无法生成目标")
                    continue
                
                print(f"    🎯 重试 {retry+1}: 新目标 {new_goal[:3]}")
                new_goal_pose = Pose.from_list(new_goal)
                
                retry_result = motion_gen.plan_single(
                    current_state, 
                    new_goal_pose, 
                    MotionGenPlanConfig(max_attempts=1)
                )
                
                if retry_result.success is not None and (retry_result.success.item() if hasattr(retry_result.success, 'item') else retry_result.success):
                    print(f"    ✅ 重试 {retry+1}: 规划成功！")
                    print(f"    📊 规划时间: {retry_result.solve_time:.4f}秒")
                    break
                else:
                    print(f"    ❌ 重试 {retry+1}: 规划失败")
            else:
                print(f"    ❌ 所有重试都失败了")
        
        print(f"\n🎉 Collision-Free目标生成演示完成！")
        print(f"\n💡 优势总结:")
        print(f"  1. 🎯 智能避障: 自动避开已知障碍物")
        print(f"  2. 🔄 自适应重试: 失败时自动生成新目标")
        print(f"  3. 📈 提高成功率: 显著减少规划失败")
        print(f"  4. 🌍 场景适应: 适用于任意复杂场景")
        
    except Exception as e:
        print(f"❌ 演示过程中发生错误: {e}")
    finally:
        visualizer.disconnect()


if __name__ == "__main__":
    demo_collision_free_goal_generation() 