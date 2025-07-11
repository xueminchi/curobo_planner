#!/usr/bin/env python3
"""
测试世界配置可视化功能
"""

from curobo.geom.types import WorldConfig
from curobo.util.logger import setup_curobo_logger
from curobo.util_file import get_world_configs_path, join_path, load_yaml

from world_visualization_pybullet import WorldVisualizerPyBullet


def test_primitives_3d():
    """测试包含多种几何体的primitives_3d配置"""
    print("=== 测试 collision_primitives_3d.yml ===")
    
    try:
        # 加载世界配置
        world_file = "collision_primitives_3d.yml"
        world_cfg_dict = load_yaml(join_path(get_world_configs_path(), world_file))
        world_cfg = WorldConfig.from_dict(world_cfg_dict)
        
        print(f"✅ 成功加载配置文件: {world_file}")
        
        # 创建可视化器（无GUI版本用于测试）
        visualizer = WorldVisualizerPyBullet(gui=False)
        
        # 加载世界配置
        visualizer.load_world_config(world_cfg)
        
        print(f"✅ 成功可视化配置，共创建 {len(visualizer.obstacle_ids)} 个障碍物")
        
        visualizer.disconnect()
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


def test_with_gui():
    """使用GUI测试primitives_3d配置"""
    print("=== GUI测试 collision_primitives_3d.yml ===")
    print("这个配置包含立方体、球体和胶囊体")
    
    try:
        # 加载世界配置
        world_file = "collision_primitives_3d.yml"
        world_cfg_dict = load_yaml(join_path(get_world_configs_path(), world_file))
        world_cfg = WorldConfig.from_dict(world_cfg_dict)
        
        # 创建可视化器
        visualizer = WorldVisualizerPyBullet(gui=True)
        
        # 加载世界配置
        visualizer.load_world_config(world_cfg)
        
        print(f"\n🎯 可视化完成！你应该看到：")
        print("  📦 红色立方体 (2个)")
        print("  🌕 绿色球体 (2个)")
        print("  💊 蓝色胶囊体 (2个)")
        print("\n按回车键退出...")
        input()
        
        visualizer.disconnect()
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


if __name__ == "__main__":
    setup_curobo_logger("error")
    
    print("开始测试世界配置可视化功能...")
    
    # 先进行无GUI测试
    success1 = test_primitives_3d()
    
    if success1:
        print("\n✅ 无GUI测试通过！")
        
        # 询问是否进行GUI测试
        choice = input("\n是否进行GUI可视化测试？(y/n): ").strip().lower()
        if choice in ['y', 'yes', '是']:
            success2 = test_with_gui()
            if success2:
                print("✅ GUI测试通过！")
            else:
                print("❌ GUI测试失败")
        else:
            print("跳过GUI测试")
    else:
        print("❌ 测试失败") 