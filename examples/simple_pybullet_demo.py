#!/usr/bin/env python3
"""
简单的PyBullet可视化演示脚本
"""

from pybullet_kinematics_visualization import PyBulletKinematicsVisualizer
import numpy as np
import time

def main():
    """主演示函数"""
    print("=== 简单的PyBullet可视化演示 ===")
    print("这将打开一个PyBullet窗口，显示Franka机器人的运动")
    print("请等待几秒钟...")
    
    try:
        # 创建可视化器
        visualizer = PyBulletKinematicsVisualizer(gui=True)
        
        # 重置到收缩配置
        print("\n1. 重置到收缩配置...")
        visualizer.reset_to_retract_config()
        time.sleep(2)
        
        # 显示一些随机配置
        print("\n2. 显示3个随机配置...")
        visualizer.visualize_random_configurations(num_configs=3, delay=1.5)
        
        # 创建简单的摆动动作
        print("\n3. 创建简单的摆动动作...")
        trajectory = []
        base_config = visualizer.retract_config.copy()
        
        # 创建第一个关节的摆动
        for t in np.linspace(0, 2*np.pi, 30):
            config = base_config.copy()
            config[0] = base_config[0] + np.sin(t) * 0.5  # 第一个关节摆动
            config[1] = base_config[1] + np.sin(t * 0.5) * 0.3  # 第二个关节轻微摆动
            trajectory.append(config)
        
        # 可视化轨迹
        visualizer.visualize_trajectory(trajectory, delay=0.1)
        
        # 重置到收缩配置
        print("\n4. 重置到收缩配置...")
        visualizer.reset_to_retract_config()
        time.sleep(1)
        
        print("\n演示完成！按回车键退出...")
        input()
        
    except KeyboardInterrupt:
        print("\n演示被用户中断")
    except Exception as e:
        print(f"\n演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理
        try:
            visualizer.disconnect()
        except:
            pass
        print("演示结束")

if __name__ == "__main__":
    main() 