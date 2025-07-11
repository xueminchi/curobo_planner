#!/usr/bin/env python3
"""
简单的PyBullet可视化测试脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pybullet_kinematics_visualization import PyBulletKinematicsVisualizer
import numpy as np


def test_basic_functionality():
    """测试基本功能"""
    print("Testing basic PyBullet visualization functionality...")
    
    try:
        # 创建可视化器 (无GUI模式用于测试)
        visualizer = PyBulletKinematicsVisualizer(gui=False)
        print("✓ Visualizer created successfully")
        
        # 测试重置到收缩配置
        visualizer.reset_to_retract_config()
        print("✓ Reset to retract configuration")
        
        # 测试设置关节角度
        joint_angles = [0.0, -1.3, 0.0, -2.5, 0.0, 1.0, 0.0, 0.04, 0.04]
        visualizer.set_joint_angles(joint_angles)
        print("✓ Set joint angles")
        
        # 测试获取关节角度
        current_angles = visualizer.get_joint_angles()
        print(f"✓ Retrieved joint angles: {len(current_angles)} joints")
        
        # 测试获取末端执行器位姿
        ee_pos, ee_orn = visualizer.get_end_effector_pose()
        if ee_pos is not None:
            print(f"✓ End effector position: {ee_pos}")
        else:
            print("! End effector position not found")
        
        # 测试随机配置
        print("\nTesting random configurations...")
        visualizer.visualize_random_configurations(num_configs=3, delay=0.1)
        print("✓ Random configurations test completed")
        
        # 清理
        visualizer.disconnect()
        print("✓ Disconnected successfully")
        
        print("\n=== All tests passed! ===")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_gui():
    """测试GUI模式"""
    print("\nTesting with GUI mode...")
    print("This will open a PyBullet window for 10 seconds...")
    
    try:
        # 创建带GUI的可视化器
        visualizer = PyBulletKinematicsVisualizer(gui=True)
        print("✓ GUI visualizer created")
        
        # 简单的演示
        visualizer.reset_to_retract_config()
        
        # 创建简单的轨迹
        trajectory = []
        for t in np.linspace(0, np.pi, 20):
            config = visualizer.retract_config.copy()
            config[0] = np.sin(t) * 0.5  # 第一个关节
            trajectory.append(config)
        
        visualizer.visualize_trajectory(trajectory, delay=0.2)
        
        print("✓ GUI test completed")
        visualizer.disconnect()
        return True
        
    except Exception as e:
        print(f"✗ GUI test failed: {e}")
        return False


if __name__ == "__main__":
    print("=== PyBullet Visualization Test ===\n")
    
    # 测试基本功能
    basic_test_passed = test_basic_functionality()
    
    if basic_test_passed:
        # 询问是否运行GUI测试
        response = input("\nWould you like to run the GUI test? (y/n): ").lower().strip()
        if response == 'y' or response == 'yes':
            test_with_gui()
        else:
            print("GUI test skipped.")
    else:
        print("Basic test failed. GUI test will be skipped.")
    
    print("\nTest completed!") 