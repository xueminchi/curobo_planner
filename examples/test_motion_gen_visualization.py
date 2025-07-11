#!/usr/bin/env python3
"""
运动规划可视化功能测试脚本
"""

import torch
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.util.logger import setup_curobo_logger
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

from motion_gen_pybullet_visualization import MotionGenPyBulletVisualizer


def test_simple_motion_gen():
    """测试简单的运动规划可视化"""
    print("=== 测试简单运动规划可视化 ===")
    
    tensor_args = TensorDeviceType()
    
    # 创建简单的世界配置
    world_config = {
        "cuboid": {
            "table": {
                "dims": [1.0, 1.0, 0.1],
                "pose": [0.0, 0.0, -0.05, 1, 0, 0, 0.0],
            },
        },
    }
    
    try:
        # 创建运动规划配置
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            "franka.yml",
            world_config,
            tensor_args,
            interpolation_dt=0.05,
            use_cuda_graph=True,
        )
        motion_gen = MotionGen(motion_gen_config)
        motion_gen.warmup()
        
        # 创建可视化器
        visualizer = MotionGenPyBulletVisualizer(gui=True)
        
        # 获取起始状态
        retract_cfg = motion_gen.get_retract_config()
        start_state = JointState.from_position(retract_cfg.view(1, -1))
        
        # 设置简单的目标姿态
        goal_pose = Pose.from_list([0.3, 0.2, 0.4, 1.0, 0.0, 0.0, 0.0])
        
        print("规划轨迹...")
        
        # 规划轨迹
        result = motion_gen.plan_single(
            start_state, 
            goal_pose, 
            MotionGenPlanConfig(max_attempts=3)
        )
        
        if result.success.item():
            print(f"轨迹规划成功！")
            print(f"规划时间: {result.solve_time:.4f}秒")
            
            # 获取插值轨迹
            interpolated_trajectory = result.get_interpolated_plan()
            
            print(f"轨迹包含 {len(interpolated_trajectory.position)} 个点")
            
            # 简单可视化（显示起始和目标标记）
            visualizer.clear_all_markers()
            
            # 设置起始位置
            start_joints = start_state.position[0].cpu().numpy()
            extended_start = visualizer._extend_joint_configuration(start_joints)
            visualizer.set_joint_angles(extended_start)
            start_ee_pos, start_ee_quat = visualizer.get_end_effector_pose()
            
            if start_ee_pos:
                visualizer.add_start_marker(start_ee_pos, start_ee_quat)
                print(f"起始位置: {start_ee_pos}")
            
            # 添加目标标记
            goal_pos = goal_pose.position[0].cpu().numpy()
            goal_quat = goal_pose.quaternion[0].cpu().numpy()
            visualizer.add_goal_marker(goal_pos, goal_quat)
            print(f"目标位置: {goal_pos}")
            
            # 设置到最终位置
            final_joints = interpolated_trajectory.position[-1].cpu().numpy()
            extended_final = visualizer._extend_joint_configuration(final_joints)
            visualizer.set_joint_angles(extended_final)
            
            final_ee_pos, _ = visualizer.get_end_effector_pose()
            if final_ee_pos:
                error = torch.norm(torch.tensor(final_ee_pos) - torch.tensor(goal_pos)).item()
                print(f"最终位置: {final_ee_pos}")
                print(f"位置误差: {error:.6f}m")
            
            print("\n测试成功！可视化器已显示起始位置(绿色立方体)和目标位置(红色球体)")
            print("机器人已移动到最终位置")
            print("按回车键退出...")
            input()
            
        else:
            print(f"轨迹规划失败！状态: {result.status}")
            return False
            
        visualizer.disconnect()
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        return False


if __name__ == "__main__":
    setup_curobo_logger("error")
    
    print("开始测试运动规划可视化功能...")
    success = test_simple_motion_gen()
    
    if success:
        print("✅ 所有测试通过！")
    else:
        print("❌ 测试失败") 