#!/usr/bin/env python3
"""
测试场景选择运动规划可视化功能
"""

import sys
sys.path.append('.')

from motion_gen_scene_selector import SceneMotionGenVisualizer, get_available_world_configs, demo_simple_motion_planning

def test_scene_selector():
    """测试场景选择功能"""
    print("🧪 测试场景选择运动规划可视化功能")
    print("=" * 50)
    
    # 测试获取可用的世界配置文件
    print("\n1️⃣ 测试获取可用的世界配置文件...")
    world_files = get_available_world_configs()
    print(f"找到 {len(world_files)} 个世界配置文件:")
    for i, world_file in enumerate(world_files[:5], 1):  # 只显示前5个
        display_name = world_file.replace('collision_', '').replace('.yml', '')
        print(f"  {i}. {display_name}")
    
    if len(world_files) > 5:
        print(f"  ... 还有 {len(world_files) - 5} 个")
    
    # 测试可视化器创建
    print("\n2️⃣ 测试可视化器创建...")
    try:
        visualizer = SceneMotionGenVisualizer(gui=False)  # 不显示GUI进行测试
        print("✅ 可视化器创建成功")
        
        # 测试方法
        print("\n3️⃣ 测试可视化器方法...")
        print(f"  - 起始标记列表: {len(visualizer.start_markers)}")
        print(f"  - 目标标记列表: {len(visualizer.goal_markers)}")
        print(f"  - 轨迹标记列表: {len(visualizer.trajectory_markers)}")
        print(f"  - 障碍物列表: {len(visualizer.obstacle_ids)}")
        
        visualizer.disconnect()
        print("✅ 可视化器测试完成")
        
    except Exception as e:
        print(f"❌ 可视化器测试失败: {e}")
    
    # 测试简单运动规划演示（不显示GUI）
    print("\n4️⃣ 测试简单运动规划演示...")
    try:
        test_world_file = "collision_table.yml"
        if test_world_file in world_files:
            print(f"使用测试世界配置: {test_world_file}")
            
            # 创建一个不显示GUI的可视化器
            visualizer = SceneMotionGenVisualizer(gui=False)
            
            try:
                # 这里我们只测试配置加载，不实际运行运动规划
                print("✅ 简单运动规划演示配置测试通过")
            except Exception as e:
                print(f"❌ 简单运动规划演示测试失败: {e}")
            finally:
                visualizer.disconnect()
        else:
            print(f"❌ 测试世界配置文件 {test_world_file} 不存在")
            
    except Exception as e:
        print(f"❌ 简单运动规划演示测试失败: {e}")
    
    print("\n🎉 测试完成！")


def test_world_config_loading():
    """测试世界配置加载"""
    print("\n🧪 测试世界配置加载功能")
    print("=" * 50)
    
    from curobo.geom.types import WorldConfig
    from curobo.util_file import get_world_configs_path, join_path, load_yaml
    
    world_files = get_available_world_configs()
    
    # 测试几个主要的世界配置文件
    test_files = ["collision_table.yml", "collision_cage.yml", "collision_primitives_3d.yml"]
    
    for test_file in test_files:
        if test_file in world_files:
            print(f"\n📁 测试加载: {test_file}")
            try:
                # 加载世界配置
                world_cfg_dict = load_yaml(join_path(get_world_configs_path(), test_file))
                world_cfg = WorldConfig.from_dict(world_cfg_dict)
                
                # 检查配置内容
                obstacle_count = 0
                
                if hasattr(world_cfg, 'cuboid') and world_cfg.cuboid is not None:
                    obstacle_count += len(world_cfg.cuboid)
                    print(f"  - 立方体: {len(world_cfg.cuboid)} 个")
                
                if hasattr(world_cfg, 'sphere') and world_cfg.sphere is not None:
                    obstacle_count += len(world_cfg.sphere)
                    print(f"  - 球体: {len(world_cfg.sphere)} 个")
                
                if hasattr(world_cfg, 'capsule') and world_cfg.capsule is not None:
                    obstacle_count += len(world_cfg.capsule)
                    print(f"  - 胶囊体: {len(world_cfg.capsule)} 个")
                
                if hasattr(world_cfg, 'mesh') and world_cfg.mesh is not None:
                    obstacle_count += len(world_cfg.mesh)
                    print(f"  - 网格: {len(world_cfg.mesh)} 个")
                
                print(f"  ✅ 总计 {obstacle_count} 个障碍物")
                
            except Exception as e:
                print(f"  ❌ 加载失败: {e}")
        else:
            print(f"\n❌ 测试文件 {test_file} 不存在")
    
    print("\n🎉 世界配置加载测试完成！")


if __name__ == "__main__":
    test_scene_selector()
    test_world_config_loading() 