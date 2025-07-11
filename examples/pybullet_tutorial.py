#!/usr/bin/env python3
"""
PyBullet机器人可视化教程
展示如何使用PyBullet可视化工具的所有主要功能
"""

from pybullet_kinematics_visualization import PyBulletKinematicsVisualizer
import numpy as np
import time

def tutorial_step_1():
    """教程步骤1：基本机器人加载和配置"""
    print("\n" + "="*60)
    print("教程步骤 1: 基本机器人加载和配置")
    print("="*60)
    
    print("正在创建PyBullet可视化器...")
    visualizer = PyBulletKinematicsVisualizer(gui=True)
    
    print(f"机器人关节数量: {len(visualizer.joint_names)}")
    print(f"机器人关节名称: {visualizer.joint_names}")
    print(f"收缩配置: {visualizer.retract_config}")
    
    print("\n正在重置到收缩配置...")
    visualizer.reset_to_retract_config()
    
    # 获取末端执行器位置
    ee_pos, ee_orn = visualizer.get_end_effector_pose()
    if ee_pos is not None:
        print(f"末端执行器位置: {ee_pos}")
    
    input("\n按回车键继续下一步...")
    return visualizer

def tutorial_step_2(visualizer):
    """教程步骤2：手动设置关节角度"""
    print("\n" + "="*60)
    print("教程步骤 2: 手动设置关节角度")
    print("="*60)
    
    # 定义几个预设的关节配置
    configurations = [
        {
            "name": "准备位置",
            "angles": [0.0, -0.5, 0.0, -1.5, 0.0, 1.0, 0.0, 0.04, 0.04]
        },
        {
            "name": "右伸展",
            "angles": [1.0, -0.5, 0.0, -1.5, 0.0, 1.0, 0.0, 0.04, 0.04]
        },
        {
            "name": "左伸展", 
            "angles": [-1.0, -0.5, 0.0, -1.5, 0.0, 1.0, 0.0, 0.04, 0.04]
        },
        {
            "name": "向上",
            "angles": [0.0, -1.0, 0.0, -0.5, 0.0, 1.5, 0.0, 0.04, 0.04]
        }
    ]
    
    for i, config in enumerate(configurations):
        print(f"\n{i+1}. 设置到 '{config['name']}' 位置...")
        print(f"   关节角度: {config['angles']}")
        
        visualizer.set_joint_angles(config['angles'])
        
        # 获取末端执行器位置
        ee_pos, ee_orn = visualizer.get_end_effector_pose()
        if ee_pos is not None:
            print(f"   末端执行器位置: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
        
        time.sleep(1.5)
    
    input("\n按回车键继续下一步...")

def tutorial_step_3(visualizer):
    """教程步骤3：随机配置可视化"""
    print("\n" + "="*60)
    print("教程步骤 3: 随机配置可视化")
    print("="*60)
    
    print("正在生成并可视化随机配置...")
    print("这将展示机器人在不同随机位置的样子")
    
    visualizer.visualize_random_configurations(num_configs=5, delay=2.0)
    
    input("\n按回车键继续下一步...")

def tutorial_step_4(visualizer):
    """教程步骤4：轨迹可视化"""
    print("\n" + "="*60)
    print("教程步骤 4: 轨迹可视化")
    print("="*60)
    
    print("正在创建几种不同的轨迹模式...")
    
    # 轨迹1：正弦波运动
    print("\n4.1 正弦波运动 (第一个关节)")
    trajectory1 = []
    base_config = visualizer.retract_config.copy()
    
    for t in np.linspace(0, 2*np.pi, 40):
        config = base_config.copy()
        config[0] = base_config[0] + np.sin(t) * 0.8  # 第一个关节
        trajectory1.append(config)
    
    visualizer.visualize_trajectory(trajectory1, delay=0.08)
    
    # 轨迹2：复合运动
    print("\n4.2 复合运动 (多个关节协调)")
    trajectory2 = []
    
    for t in np.linspace(0, 2*np.pi, 50):
        config = base_config.copy()
        config[0] = base_config[0] + np.sin(t) * 0.5      # 第一个关节
        config[1] = base_config[1] + np.cos(t) * 0.3      # 第二个关节
        config[2] = base_config[2] + np.sin(t * 2) * 0.2  # 第三个关节
        trajectory2.append(config)
    
    visualizer.visualize_trajectory(trajectory2, delay=0.06)
    
    # 轨迹3：圆形运动（末端执行器）
    print("\n4.3 尝试圆形运动模式")
    trajectory3 = []
    
    for t in np.linspace(0, 2*np.pi, 60):
        config = base_config.copy()
        config[0] = base_config[0] + np.sin(t) * 0.4
        config[1] = base_config[1] + np.sin(t + np.pi/2) * 0.3
        config[3] = base_config[3] + np.cos(t) * 0.2
        trajectory3.append(config)
    
    visualizer.visualize_trajectory(trajectory3, delay=0.05)
    
    input("\n按回车键继续下一步...")

def tutorial_step_5(visualizer):
    """教程步骤5：关节限制测试"""
    print("\n" + "="*60)
    print("教程步骤 5: 关节限制和安全测试")
    print("="*60)
    
    print("正在测试关节限制...")
    print("这将尝试一些接近关节限制的配置")
    
    # 测试关节限制
    test_configs = [
        {
            "name": "关节1最大正值",
            "angles": [2.5, -0.5, 0.0, -1.5, 0.0, 1.0, 0.0, 0.04, 0.04]
        },
        {
            "name": "关节1最大负值",
            "angles": [-2.5, -0.5, 0.0, -1.5, 0.0, 1.0, 0.0, 0.04, 0.04]
        },
        {
            "name": "关节2最大正值",
            "angles": [0.0, 1.5, 0.0, -1.5, 0.0, 1.0, 0.0, 0.04, 0.04]
        },
        {
            "name": "关节4最大负值",
            "angles": [0.0, -0.5, 0.0, -2.8, 0.0, 1.0, 0.0, 0.04, 0.04]
        }
    ]
    
    for config in test_configs:
        print(f"\n测试: {config['name']}")
        try:
            visualizer.set_joint_angles(config['angles'])
            ee_pos, ee_orn = visualizer.get_end_effector_pose()
            if ee_pos is not None:
                print(f"  末端执行器位置: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
            else:
                print("  无法获取末端执行器位置")
        except Exception as e:
            print(f"  错误: {e}")
        
        time.sleep(1.5)
    
    input("\n按回车键继续下一步...")

def tutorial_step_6(visualizer):
    """教程步骤6：高级功能展示"""
    print("\n" + "="*60)
    print("教程步骤 6: 高级功能展示")
    print("="*60)
    
    print("6.1 获取当前关节状态")
    current_angles = visualizer.get_joint_angles()
    print(f"当前关节角度: {current_angles}")
    
    print("\n6.2 计算关节角度变化")
    target_angles = [0.5, -1.0, 0.5, -2.0, 0.5, 1.5, 0.5, 0.04, 0.04]
    
    print(f"目标角度: {target_angles}")
    
    # 平滑插值到目标位置
    print("\n6.3 平滑插值运动")
    n_steps = 30
    interpolated_trajectory = []
    
    for i in range(n_steps + 1):
        alpha = i / n_steps
        interpolated_config = []
        
        for j in range(len(current_angles)):
            interpolated_angle = current_angles[j] * (1 - alpha) + target_angles[j] * alpha
            interpolated_config.append(interpolated_angle)
        
        interpolated_trajectory.append(interpolated_config)
    
    visualizer.visualize_trajectory(interpolated_trajectory, delay=0.1)
    
    input("\n按回车键完成教程...")

def main():
    """主教程函数"""
    print("欢迎使用PyBullet机器人可视化教程!")
    print("这个教程将向您展示如何使用PyBullet可视化工具的所有主要功能。")
    print("\n注意：这将打开一个PyBullet 3D窗口，请确保您的系统支持OpenGL。")
    
    response = input("\n是否开始教程? (y/n): ").lower().strip()
    if response not in ['y', 'yes', '是', '开始']:
        print("教程取消。")
        return
    
    visualizer = None
    
    try:
        # 执行教程步骤
        visualizer = tutorial_step_1()
        tutorial_step_2(visualizer)
        tutorial_step_3(visualizer)
        tutorial_step_4(visualizer)
        tutorial_step_5(visualizer)
        tutorial_step_6(visualizer)
        
        print("\n" + "="*60)
        print("教程完成！")
        print("="*60)
        print("您已经学会了如何使用PyBullet可视化工具的所有主要功能：")
        print("1. 加载和配置机器人")
        print("2. 手动设置关节角度")
        print("3. 生成随机配置")
        print("4. 创建和可视化轨迹")
        print("5. 测试关节限制")
        print("6. 使用高级功能")
        print("\n现在您可以使用这些功能来创建自己的机器人可视化应用！")
        
    except KeyboardInterrupt:
        print("\n\n教程被用户中断。")
    except Exception as e:
        print(f"\n\n教程过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if visualizer:
            try:
                visualizer.disconnect()
            except:
                pass
        print("\n教程结束。感谢您的参与！")

if __name__ == "__main__":
    main() 