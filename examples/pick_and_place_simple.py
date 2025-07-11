#!/usr/bin/env python3
"""
Pick and Place 演示脚本 (简化版本)
专注于核心功能，减少调试输出，提供更流畅的演示体验
"""

import time
import os
import numpy as np
import pybullet as p
from datetime import datetime
from typing import Optional, List

# Third Party
import torch

# CuRobo
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.util.logger import setup_curobo_logger
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

# Local
from pybullet_kinematics_visualization import PyBulletKinematicsVisualizer

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class PickAndPlaceVisualizer(PyBulletKinematicsVisualizer):
    """简化的Pick and Place可视化器"""
    
    def __init__(self, robot_config_name="franka.yml", gui=True, enable_video=False):
        super().__init__(robot_config_name, gui)
        self.obstacle_ids: List[int] = []
        self.target_object_id: Optional[int] = None
        self.target_markers: List[int] = []
        self.sphere_markers: List[int] = []
        self.sphere_offsets: List[List[float]] = []
        self.motion_gen: Optional[MotionGen] = None
        
        # 视频录制相关
        self.enable_video = enable_video
        self.video_folder: Optional[str] = None
        self.video_log_id: Optional[int] = None
        self.video_counter = 0
        
        if self.enable_video:
            self._setup_video_recording()
    
    def _setup_video_recording(self):
        """设置视频录制文件夹和路径"""
        # 创建基于当前日期时间的文件夹
        current_time = datetime.now()
        folder_name = f"pick_and_place_videos_{current_time.strftime('%Y%m%d_%H%M%S')}"
        
        # 在当前目录创建视频文件夹
        self.video_folder = os.path.join(os.getcwd(), folder_name)
        
        try:
            os.makedirs(self.video_folder, exist_ok=True)
            print(f"📁 视频文件夹已创建: {self.video_folder}")
        except Exception as e:
            print(f"❌ 创建视频文件夹失败: {e}")
            self.enable_video = False
            return
        
        # 设置视频录制参数
        p.setAdditionalSearchPath(self.video_folder)
        print(f"🎥 视频录制已启用，文件将保存到: {folder_name}")
    
    def start_video_recording(self, stage_name="trajectory"):
        """开始录制视频"""
        if not self.enable_video or self.video_folder is None:
            return None
        
        # 生成带时间戳的视频文件名
        timestamp = datetime.now().strftime("%H%M%S")
        video_filename = f"{stage_name}_{timestamp}.mp4"
        video_path = os.path.join(self.video_folder, video_filename)
        
        try:
            self.video_log_id = p.startStateLogging(
                p.STATE_LOGGING_VIDEO_MP4, 
                video_path,
                objectUniqueIds=[]  # 录制整个场景
            )
            print(f"🎬 开始录制视频: {video_filename}")
            self.video_counter += 1
            return video_filename
        except Exception as e:
            print(f"❌ 开始录制视频失败: {e}")
            return None
    
    def stop_video_recording(self):
        """停止录制视频"""
        if not self.enable_video or self.video_log_id is None:
            return
        
        try:
            p.stopStateLogging(self.video_log_id)
            print(f"⏹️  视频录制已停止")
            self.video_log_id = None
        except Exception as e:
            print(f"❌ 停止录制视频失败: {e}")
    
    def create_world_objects(self):
        """创建世界中的物体"""
        self.clear_obstacles()
        
        # 创建目标立方体
        target_dims = [0.05, 0.05, 0.05]
        target_position = [0.45, 0.35, 0.025]
        
        target_collision_shape = p.createCollisionShape(
            p.GEOM_BOX, 
            halfExtents=[target_dims[0]/2, target_dims[1]/2, target_dims[2]/2]
        )
        target_visual_shape = p.createVisualShape(
            p.GEOM_BOX, 
            halfExtents=[target_dims[0]/2, target_dims[1]/2, target_dims[2]/2],
            rgbaColor=[1.0, 0.2, 0.2, 0.8]
        )
        
        self.target_object_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=target_collision_shape,
            baseVisualShapeIndex=target_visual_shape,
            basePosition=target_position
        )
        
        # 创建障碍物
        obstacles = [
            {
                "position": [-0.2, -0.3, 0.6],
                "dims": [0.08, 0.08, 1.2],
                "color": [0.2, 0.2, 0.8, 0.7]
            },
            {
                "position": [0.6, 0.0, 0.55],
                "dims": [0.35, 0.1, 1.1],
                "color": [0.2, 0.8, 0.2, 0.7]
            }
        ]
        
        for obs in obstacles:
            collision_shape = p.createCollisionShape(
                p.GEOM_BOX, 
                halfExtents=[obs["dims"][0]/2, obs["dims"][1]/2, obs["dims"][2]/2]
            )
            visual_shape = p.createVisualShape(
                p.GEOM_BOX, 
                halfExtents=[obs["dims"][0]/2, obs["dims"][1]/2, obs["dims"][2]/2],
                rgbaColor=obs["color"]
            )
            
            obstacle_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=obs["position"]
            )
            
            self.obstacle_ids.append(obstacle_id)
        
        return target_position, target_dims
    
    def add_position_marker(self, position, size=0.02, color=[1, 1, 0, 0.8]):
        """添加位置标记"""
        visual_shape = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=size,
            rgbaColor=color
        )
        
        marker_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_shape,
            basePosition=position
        )
        
        self.target_markers.append(marker_id)
        return marker_id
    
    def create_sphere_markers(self, joint_state):
        """创建球体标记来显示抓取对象的几何表示"""
        if self.motion_gen is None:
            return
        
        try:
            # 获取运动学状态
            kin_state = self.motion_gen.compute_kinematics(joint_state)
            
            # 获取附加对象的球体信息
            attached_spheres = self.motion_gen.kinematics.kinematics_config.get_link_spheres("attached_object")
            
            if attached_spheres is not None and len(attached_spheres) > 0:
                # 计算当前末端执行器位置
                ee_pos, ee_quat = self.get_end_effector_pose()
                
                if ee_pos is not None:
                    # 清除之前的球体标记
                    self.clear_sphere_markers()
                    
                    # 为每个球体创建可视化标记
                    for i, sphere_data in enumerate(attached_spheres.cpu().numpy()):
                        x, y, z, radius = sphere_data
                        
                        if radius > 0:  # 只处理有效球体
                            # 计算球体在世界坐标系中的位置
                            # 这里使用简化的计算，实际应该使用完整的变换矩阵
                            world_pos = [ee_pos[0] + x, ee_pos[1] + y, ee_pos[2] + z]
                            
                            # 创建可视化球体，使用较大的半径确保可见
                            visual_radius = max(radius * 2, 0.02)
                            
                            visual_shape = p.createVisualShape(
                                p.GEOM_SPHERE,
                                radius=visual_radius,
                                rgbaColor=[1.0, 1.0, 0.0, 1.0]  # 亮黄色，不透明
                            )
                            
                            sphere_marker = p.createMultiBody(
                                baseMass=0,
                                baseVisualShapeIndex=visual_shape,
                                basePosition=world_pos
                            )
                            
                            self.sphere_markers.append(sphere_marker)
                            
                            # 保存球体相对于末端执行器的偏移
                            offset = [x, y, z]
                            self.sphere_offsets.append(offset)
                    
                    print(f"✨ 创建了 {len(self.sphere_markers)} 个球体标记")
                    
        except Exception as e:
            print(f"⚠️  创建球体标记时出错: {e}")
    
    def update_sphere_markers(self):
        """更新球体标记位置"""
        if len(self.sphere_markers) == 0 or len(self.sphere_offsets) == 0:
            return
        
        try:
            ee_pos, ee_quat = self.get_end_effector_pose()
            
            if ee_pos is not None:
                for i, (sphere_id, offset) in enumerate(zip(self.sphere_markers, self.sphere_offsets)):
                    new_pos = [
                        ee_pos[0] + offset[0],
                        ee_pos[1] + offset[1],
                        ee_pos[2] + offset[2]
                    ]
                    
                    p.resetBasePositionAndOrientation(
                        sphere_id,
                        new_pos,
                        [0, 0, 0, 1]
                    )
                    
        except Exception as e:
            pass  # 静默处理错误
    
    def clear_sphere_markers(self):
        """清除球体标记"""
        for sphere_id in self.sphere_markers:
            try:
                p.removeBody(sphere_id)
            except:
                pass
        
        self.sphere_markers.clear()
        self.sphere_offsets.clear()
    
    def clear_obstacles(self):
        """清除所有障碍物"""
        for obstacle_id in self.obstacle_ids:
            try:
                p.removeBody(obstacle_id)
            except:
                pass
        self.obstacle_ids.clear()
        
        if self.target_object_id is not None:
            try:
                p.removeBody(self.target_object_id)
            except:
                pass
            self.target_object_id = None
            
        for marker_id in self.target_markers:
            try:
                p.removeBody(marker_id)
            except:
                pass
        self.target_markers.clear()
        
        self.clear_sphere_markers()
    
    def safe_get_joint_state(self, trajectory, index=-1):
        """安全地从轨迹中获取关节状态"""
        try:
            if hasattr(trajectory, 'position') and trajectory.position is not None:
                final_position = trajectory.position[index]
                
                if torch.is_tensor(final_position):
                    if final_position.dim() == 1:
                        return JointState.from_position(final_position.view(1, -1))
                    elif final_position.dim() == 2:
                        return JointState.from_position(final_position)
                    else:
                        return JointState.from_position(final_position[0].view(1, -1))
                else:
                    final_position = torch.tensor(final_position, dtype=torch.float32)
                    return JointState.from_position(final_position.view(1, -1))
            else:
                return None
                
        except Exception as e:
            print(f"❌ 获取关节状态时出错: {e}")
            return None
    
    def play_trajectory(self, trajectory, dt=0.02, speed=2.0, show_attached_object=False, stage_name="trajectory"):
        """播放轨迹动画"""
        if trajectory is None or not hasattr(trajectory, 'position'):
            print("❌ 无效的轨迹数据")
            return
        
        # 开始录制视频（如果启用）
        video_filename = None
        if self.enable_video:
            video_filename = self.start_video_recording(stage_name)
        
        try:
            print(f"🎬 播放轨迹: {len(trajectory.position)} 个关键点")
            
            for i, joint_pos in enumerate(trajectory.position):
                if hasattr(joint_pos, 'cpu'):
                    joint_config = joint_pos.cpu().numpy()
                else:
                    joint_config = joint_pos
                
                extended_config = self._extend_joint_configuration(joint_config)
                self.set_joint_angles(extended_config)
                
                # 如果需要显示附加物体
                if show_attached_object and self.target_object_id is not None:
                    ee_pos, ee_quat = self.get_end_effector_pose()
                    if ee_pos is not None:
                        object_pos = [ee_pos[0], ee_pos[1], ee_pos[2] - 0.05]
                        p.resetBasePositionAndOrientation(
                            self.target_object_id, 
                            object_pos, 
                            ee_quat
                        )
                
                # 更新球体标记
                if show_attached_object:
                    self.update_sphere_markers()
                
                p.stepSimulation()
                time.sleep(dt / speed)
                
                # 显示进度
                if i % 10 == 0:
                    progress = (i + 1) / len(trajectory.position) * 100
                    print(f"\r播放进度: {progress:.1f}%", end='', flush=True)
            
            print(f"\n✅ 轨迹播放完成")
            
        except KeyboardInterrupt:
            print(f"\n⏹️  轨迹播放被中断")
        finally:
            # 停止视频录制
            if self.enable_video:
                self.stop_video_recording()
                if video_filename:
                    print(f"📹 视频已保存: {video_filename}")


def create_world_config():
    """创建世界配置"""
    world_config = {
        "cuboid": {
            "table": {
                "dims": [1.2, 1.2, 0.05],
                "pose": [0.4, 0.0, -0.025, 1, 0, 0, 0.0]
            },
            "target_cube": {
                "dims": [0.05, 0.05, 0.05],
                "pose": [0.45, 0.35, 0.025, 1, 0, 0, 0.0]
            },
            "obstacle1": {
                "dims": [0.08, 0.08, 1.2],
                "pose": [-0.2, -0.3, 0.6, 1, 0, 0, 0.0]
            },
            "obstacle2": {
                "dims": [0.35, 0.1, 1.1],
                "pose": [0.6, 0.0, 0.55, 1, 0, 0, 0.0]
            }
        }
    }
    
    return world_config


def run_pick_and_place_demo(enable_video=False):
    """运行Pick and Place演示"""
    print("🤖 Pick and Place 演示 (简化版本)")
    print("="*50)
    
    tensor_args = TensorDeviceType()
    robot_file = "franka.yml"
    world_config = create_world_config()
    
    # 创建运动规划器
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_file,
        world_config,
        tensor_args,
        interpolation_dt=0.02,
        collision_checker_type=CollisionCheckerType.PRIMITIVE,
        use_cuda_graph=True,
        num_trajopt_seeds=4,
        num_graph_seeds=4,
    )
    motion_gen = MotionGen(motion_gen_config)
    motion_gen.warmup()
    
    # 创建可视化器
    visualizer = PickAndPlaceVisualizer(gui=True, enable_video=enable_video)
    visualizer.motion_gen = motion_gen
    
    try:
        # 创建世界
        target_pos, target_dims = visualizer.create_world_objects()
        
        # 定义关键位置
        approach_height = 0.15
        grasp_height = 0.08
        
        approach_pos = [target_pos[0], target_pos[1], target_pos[2] + target_dims[2]/2 + approach_height]
        grasp_pos = [target_pos[0], target_pos[1], target_pos[2] + target_dims[2]/2 + grasp_height]
        place_pos = [0.35, -0.35, 0.45]
        
        # 添加位置标记
        visualizer.add_position_marker(approach_pos, 0.02, [1, 0.5, 0, 0.8])
        visualizer.add_position_marker(grasp_pos, 0.025, [1, 1, 0, 0.9])
        visualizer.add_position_marker(place_pos, 0.03, [0, 1, 1, 0.8])
        
        print(f"📍 关键位置:")
        print(f"  🟠 接近: {approach_pos}")
        print(f"  🟡 抓取: {grasp_pos}")
        print(f"  🔵 放置: {place_pos}")
        
        # 获取起始状态
        retract_cfg = motion_gen.get_retract_config()
        start_state = JointState.from_position(retract_cfg.view(1, -1))
        
        if enable_video:
            print(f"\n🎥 视频录制已启用")
        
        print(f"\n开始演示...")
        
        # === 阶段1: 移动到接近位置 ===
        print(f"\n🚀 阶段1: 移动到接近位置")
        approach_pose = Pose.from_list([
            approach_pos[0], approach_pos[1], approach_pos[2], 
            0.0, 1.0, 0.0, 0.0
        ])
        
        result1 = motion_gen.plan_single(
            start_state, 
            approach_pose, 
            MotionGenPlanConfig(max_attempts=5, enable_graph=True, enable_opt=True)
        )
        
        if result1.success is not None and result1.success.item():
            print(f"✅ 规划成功 (耗时: {result1.solve_time:.3f}s)")
            trajectory1 = result1.get_interpolated_plan()
            visualizer.play_trajectory(trajectory1, speed=2.0, stage_name="approach")
            current_state = visualizer.safe_get_joint_state(trajectory1)
            if current_state is None:
                print("❌ 无法获取当前状态")
                return
        else:
            print(f"❌ 规划失败")
            return
        
        # === 阶段2: 移动到抓取位置 ===
        print(f"\n🎯 阶段2: 移动到抓取位置")
        grasp_pose = Pose.from_list([
            grasp_pos[0], grasp_pos[1], grasp_pos[2], 
            0.0, 1.0, 0.0, 0.0
        ])
        
        result2 = motion_gen.plan_single(
            current_state, 
            grasp_pose, 
            MotionGenPlanConfig(max_attempts=5, enable_graph=True, enable_opt=True)
        )
        
        if result2.success is not None and result2.success.item():
            print(f"✅ 规划成功 (耗时: {result2.solve_time:.3f}s)")
            trajectory2 = result2.get_interpolated_plan()
            visualizer.play_trajectory(trajectory2, speed=2.0, stage_name="grasp")
            current_state = visualizer.safe_get_joint_state(trajectory2)
            if current_state is None:
                print("❌ 无法获取当前状态")
                return
        else:
            print(f"❌ 规划失败")
            return
        
        # === 阶段3: 抓取物体 ===
        print(f"\n🤏 阶段3: 抓取物体")
        success = motion_gen.attach_objects_to_robot(
            joint_state=current_state,
            object_names=["target_cube"],
            surface_sphere_radius=0.01,
            remove_obstacles_from_world_config=False
        )
        
        if success:
            print(f"✅ 物体抓取成功")
            # 创建球体标记
            visualizer.create_sphere_markers(current_state)
            
            # 更新PyBullet中的物体位置
            if visualizer.target_object_id is not None:
                ee_pos, ee_quat = visualizer.get_end_effector_pose()
                if ee_pos is not None:
                    object_pos = [ee_pos[0], ee_pos[1], ee_pos[2] - 0.05]
                    p.resetBasePositionAndOrientation(
                        visualizer.target_object_id, 
                        object_pos, 
                        ee_quat
                    )
        else:
            print(f"❌ 物体抓取失败")
            return
        
        # === 阶段4: 移动到放置位置 ===
        print(f"\n🚚 阶段4: 移动到放置位置")
        place_pose = Pose.from_list([
            place_pos[0], place_pos[1], place_pos[2], 
            0.0, 1.0, 0.0, 0.0
        ])
        
        result3 = motion_gen.plan_single(
            current_state, 
            place_pose, 
            MotionGenPlanConfig(max_attempts=8, enable_graph=True, enable_opt=True)
        )
        
        if result3.success is not None and result3.success.item():
            print(f"✅ 规划成功 (耗时: {result3.solve_time:.3f}s)")
            trajectory3 = result3.get_interpolated_plan()
            visualizer.play_trajectory(trajectory3, speed=2.0, show_attached_object=True, stage_name="place")
            current_state = visualizer.safe_get_joint_state(trajectory3)
            if current_state is None:
                print("❌ 无法获取当前状态")
                return
        else:
            print(f"❌ 规划失败")
            return
        
        # === 阶段5: 放置物体 ===
        print(f"\n📤 阶段5: 放置物体")
        motion_gen.detach_object_from_robot()
        print(f"✅ 物体放置成功")
        
        # 更新物体位置
        if visualizer.target_object_id is not None:
            final_pos = [place_pos[0], place_pos[1], place_pos[2] - 0.05]
            p.resetBasePositionAndOrientation(
                visualizer.target_object_id, 
                final_pos, 
                [0, 0, 0, 1]
            )
        
        # 清除球体标记
        visualizer.clear_sphere_markers()
        
        # === 阶段6: 返回起始位置 ===
        print(f"\n🏠 阶段6: 返回起始位置")
        
        # 创建返回起始位置的目标状态
        retract_pose = Pose.from_list([
            0.4, 0.0, 0.4,  # 安全的起始位置
            0.0, 1.0, 0.0, 0.0
        ])
        
        result4 = motion_gen.plan_single(
            current_state, 
            retract_pose, 
            MotionGenPlanConfig(max_attempts=5, enable_graph=True, enable_opt=True)
        )
        
        if result4.success is not None and result4.success.item():
            print(f"✅ 规划成功 (耗时: {result4.solve_time:.3f}s)")
            trajectory4 = result4.get_interpolated_plan()
            visualizer.play_trajectory(trajectory4, speed=2.0, stage_name="return")
        else:
            print(f"❌ 返回起始位置失败")
        
        print(f"\n🎉 Pick and Place 演示完成！")
        if enable_video:
            print(f"📹 所有视频已保存到: {visualizer.video_folder}")
        
        input("按回车键退出...")
        
    except Exception as e:
        print(f"❌ 演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # 确保停止任何正在进行的视频录制
        if visualizer.enable_video:
            visualizer.stop_video_recording()
        visualizer.disconnect()


def main():
    """主函数"""
    print("🤖 Pick and Place 演示 (简化版本)")
    print("="*50)
    print("✨ 特点:")
    print("• 🎯 简化的执行流程")
    print("• 🚀 2倍速度播放")
    print("• 💎 球体可视化抓取对象")
    print("• 📹 可选的视频录制功能")
    print("• 🔄 自动避障规划")
    
    # 询问是否启用视频录制
    video_choice = input("\n是否启用视频录制功能？(y/n): ").lower()
    enable_video = video_choice in ['y', 'yes', '是']
    
    if enable_video:
        print("🎥 视频录制已启用")
        print("📁 视频将保存在以当前时间命名的文件夹中")
        print("🏷️  每个阶段的视频都会有相应的时间标签")
    
    response = input("\n开始演示吗？(y/n): ").lower()
    if response in ['y', 'yes', '是']:
        run_pick_and_place_demo(enable_video)
    else:
        print("演示已取消")


if __name__ == "__main__":
    setup_curobo_logger("error")
    main() 