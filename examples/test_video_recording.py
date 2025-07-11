#!/usr/bin/env python3
"""
测试视频录制功能
"""

import time
import os
from datetime import datetime

# CuRobo
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.util.logger import setup_curobo_logger
from curobo.util_file import get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

# Local
from motion_gen_scene_selector import SceneMotionGenVisualizer


def test_video_recording():
    """测试视频录制功能"""
    print("=== 测试视频录制功能 ===")
    
    # 设置参数
    tensor_args = TensorDeviceType()
    world_file = "collision_table.yml"
    robot_file = "franka.yml"
    
    # 创建运动规划配置
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_file,
        world_file,
        tensor_args,
        interpolation_dt=0.02,
        use_cuda_graph=True,
        num_trajopt_seeds=4,
    )
    motion_gen = MotionGen(motion_gen_config)
    motion_gen.warmup()
    
    # 创建可视化器
    visualizer = SceneMotionGenVisualizer(gui=True)
    
    try:
        print(f"📁 视频保存目录: {visualizer.video_save_path}")
        
        # 加载障碍物
        world_cfg = WorldConfig.from_dict(
            load_yaml(join_path(get_world_configs_path(), world_file))
        )
        obstacle_ids = visualizer.load_obstacles_from_world_config(world_cfg)
        print(f"加载了 {len(obstacle_ids)} 个障碍物")
        
        # 获取起始状态
        retract_cfg = motion_gen.get_retract_config()
        start_state = JointState.from_position(retract_cfg.view(1, -1))
        
        # 设置目标姿态
        goal_pose = Pose.from_list([0.4, 0.2, 0.4, 1.0, 0.0, 0.0, 0.0])
        
        print(f"开始规划轨迹...")
        
        # 规划轨迹
        result = motion_gen.plan_single(
            start_state, 
            goal_pose, 
            MotionGenPlanConfig(max_attempts=3)
        )
        
        if result.success is not None and (result.success.item() if hasattr(result.success, 'item') else result.success):
            print(f"✅ 轨迹规划成功！")
            print(f"规划时间: {result.solve_time:.4f}秒")
            
            # 获取插值轨迹
            interpolated_trajectory = result.get_interpolated_plan()
            
            # 生成视频文件名
            video_name = f"test_recording_{datetime.now().strftime('%H%M%S')}.mp4"
            
            print(f"开始录制视频测试...")
            
            # 测试录制功能
            visualizer.visualize_trajectory(
                interpolated_trajectory, 
                start_state, 
                goal_pose,
                interpolation_dt=result.interpolation_dt,
                playback_speed=0.5,
                show_trajectory_points=True,
                record_video=True,
                video_name=video_name
            )
            
            # 检查视频文件是否存在
            video_path = f"{visualizer.video_save_path}/{video_name}"
            if os.path.exists(video_path):
                file_size = os.path.getsize(video_path)
                print(f"✅ 视频文件已创建: {video_path}")
                print(f"📊 文件大小: {file_size / (1024*1024):.2f} MB")
            else:
                print(f"❌ 视频文件未找到: {video_path}")
                
        else:
            print(f"❌ 轨迹规划失败！状态: {result.status}")
        
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        visualizer.disconnect()
        print("测试完成")


if __name__ == "__main__":
    setup_curobo_logger("error")
    test_video_recording() 