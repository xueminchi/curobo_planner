#!/usr/bin/env python3
"""
可选择场景的运动规划可视化演示
"""

import os
import time
import numpy as np
import pybullet as p
import pybullet_data
from datetime import datetime

# Third Party
import torch

# CuRobo
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import Cuboid, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.util.logger import setup_curobo_logger
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

# Local
from pybullet_kinematics_visualization import PyBulletKinematicsVisualizer

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class SceneMotionGenVisualizer(PyBulletKinematicsVisualizer):
    """场景选择的运动规划可视化器"""
    
    def __init__(self, robot_config_name="franka.yml", gui=True):
        super().__init__(robot_config_name, gui)
        self.start_markers = []
        self.goal_markers = []
        self.trajectory_markers = []
        self.obstacle_ids = []
        self.recording_log_id = None
        self.video_save_path = None
        self._setup_video_directory()
        
    def _setup_video_directory(self):
        """设置视频保存目录"""
        # 创建带日期的文件夹名称
        current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_dir = f"motion_planning_videos_{current_date}"
        
        # 使用相对路径，避免类型问题
        self.video_save_path = video_dir
        
        # 确保目录存在
        os.makedirs(self.video_save_path, exist_ok=True)
        
        # 获取绝对路径用于显示
        abs_path = os.path.abspath(self.video_save_path)
        print(f"📁 视频保存目录: {abs_path}")
        
    def start_recording(self, video_name="trajectory_video.mp4"):
        """开始录制视频
        
        Args:
            video_name: 视频文件名
        """
        if self.recording_log_id is not None:
            print("⚠️  已经在录制中，请先停止当前录制")
            return False
            
        # 构建完整的视频路径
        video_path = f"{self.video_save_path}/{video_name}"
        
        try:
            # 开始录制
            self.recording_log_id = p.startStateLogging(
                p.STATE_LOGGING_VIDEO_MP4, 
                video_path
            )
            print(f"🎬 开始录制视频: {video_name}")
            print(f"📹 录制状态: ID = {self.recording_log_id}")
            return True
            
        except Exception as e:
            print(f"❌ 录制启动失败: {e}")
            return False
            
    def stop_recording(self):
        """停止录制视频"""
        if self.recording_log_id is None:
            print("⚠️  没有正在进行的录制")
            return False
            
        try:
            p.stopStateLogging(self.recording_log_id)
            print(f"🎬 录制完成，视频已保存")
            self.recording_log_id = None
            return True
            
        except Exception as e:
            print(f"❌ 停止录制失败: {e}")
            return False
            
    def is_recording(self):
        """检查是否正在录制"""
        return self.recording_log_id is not None
    
    def cleanup_recording(self):
        """清理录制状态"""
        if self.is_recording():
            self.stop_recording()
            
    def disconnect(self):
        """断开PyBullet连接"""
        # 确保停止录制
        self.cleanup_recording()
        # 调用父类的disconnect方法
        if hasattr(super(), 'disconnect'):
            super().disconnect()
        else:
            # 如果父类没有disconnect方法，手动断开
            if hasattr(p, 'disconnect'):
                try:
                    p.disconnect()
                except:
                    pass
    
    def add_start_marker(self, position, orientation=None, size=0.05, color=[0, 1, 0, 0.8]):
        """添加起始位置标记（绿色立方体）"""
        if orientation is None:
            orientation = [0, 0, 0, 1]
            
        visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[size, size, size],
            rgbaColor=color
        )
        
        marker_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_shape,
            basePosition=position,
            baseOrientation=orientation
        )
        
        self.start_markers.append(marker_id)
        return marker_id
    
    def add_goal_marker(self, position, orientation=None, size=0.05, color=[1, 0, 0, 0.8]):
        """添加目标位置标记（红色球体）"""
        if orientation is None:
            orientation = [0, 0, 0, 1]
            
        visual_shape = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=size,
            rgbaColor=color
        )
        
        marker_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_shape,
            basePosition=position,
            baseOrientation=orientation
        )
        
        self.goal_markers.append(marker_id)
        return marker_id
    
    def add_trajectory_point_marker(self, position, size=0.02, color=[0, 0, 1, 0.4]):
        """添加轨迹点标记（蓝色小球）"""
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
        
        self.trajectory_markers.append(marker_id)
        return marker_id
    
    def clear_all_markers(self):
        """清除所有标记"""
        all_markers = self.start_markers + self.goal_markers + self.trajectory_markers
        for marker_id in all_markers:
            try:
                p.removeBody(marker_id)
            except:
                pass
                
        self.start_markers.clear()
        self.goal_markers.clear()
        self.trajectory_markers.clear()
    
    def clear_obstacles(self):
        """清除所有障碍物"""
        for obstacle_id in self.obstacle_ids:
            try:
                p.removeBody(obstacle_id)
            except:
                pass
        self.obstacle_ids.clear()
    
    def load_obstacles_from_world_config(self, world_cfg):
        """从WorldConfig加载障碍物并在PyBullet中显示"""
        self.clear_obstacles()
        
        # 加载立方体障碍物
        if hasattr(world_cfg, 'cuboid') and world_cfg.cuboid is not None and len(world_cfg.cuboid) > 0:
            print(f"加载 {len(world_cfg.cuboid)} 个立方体障碍物...")
            
            for cuboid_data in world_cfg.cuboid:
                dims = cuboid_data.dims
                pose = cuboid_data.pose
                name = cuboid_data.name
                
                # 获取颜色信息
                if hasattr(cuboid_data, 'color') and cuboid_data.color is not None:
                    color = cuboid_data.color
                    if len(color) == 3:
                        color.append(0.7)  # 添加alpha值
                else:
                    color = [0.8, 0.2, 0.2, 0.7]  # 默认半透明红色
                
                # 创建立方体
                collision_shape = p.createCollisionShape(
                    p.GEOM_BOX, 
                    halfExtents=[dims[0]/2, dims[1]/2, dims[2]/2]
                )
                visual_shape = p.createVisualShape(
                    p.GEOM_BOX, 
                    halfExtents=[dims[0]/2, dims[1]/2, dims[2]/2],
                    rgbaColor=color
                )
                
                position = [pose[0], pose[1], pose[2]]
                orientation = [pose[4], pose[5], pose[6], pose[3]]
                
                obstacle_id = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=collision_shape,
                    baseVisualShapeIndex=visual_shape,
                    basePosition=position,
                    baseOrientation=orientation
                )
                
                self.obstacle_ids.append(obstacle_id)
                print(f"  - {name}: 位置 {position}, 尺寸 {dims}")
        
        # 加载球体障碍物
        if hasattr(world_cfg, 'sphere') and world_cfg.sphere is not None and len(world_cfg.sphere) > 0:
            print(f"加载 {len(world_cfg.sphere)} 个球体障碍物...")
            
            for sphere_data in world_cfg.sphere:
                position = sphere_data.position
                radius = sphere_data.radius
                name = sphere_data.name
                
                color = [0.2, 0.8, 0.2, 0.7]  # 绿色
                
                collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
                visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
                
                obstacle_id = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=collision_shape,
                    baseVisualShapeIndex=visual_shape,
                    basePosition=position
                )
                
                self.obstacle_ids.append(obstacle_id)
                print(f"  - {name}: 位置 {position}, 半径 {radius}")
        
        # 加载胶囊体障碍物
        if hasattr(world_cfg, 'capsule') and world_cfg.capsule is not None and len(world_cfg.capsule) > 0:
            print(f"加载 {len(world_cfg.capsule)} 个胶囊体障碍物...")
            
            for capsule_data in world_cfg.capsule:
                radius = capsule_data.radius
                base = capsule_data.base
                tip = capsule_data.tip
                pose = capsule_data.pose
                name = capsule_data.name
                
                height = np.linalg.norm(np.array(tip) - np.array(base))
                color = [0.2, 0.2, 0.8, 0.7]  # 蓝色
                
                collision_shape = p.createCollisionShape(p.GEOM_CAPSULE, radius=radius, height=height)
                visual_shape = p.createVisualShape(p.GEOM_CAPSULE, radius=radius, length=height, rgbaColor=color)
                
                position = [pose[0], pose[1], pose[2]]
                orientation = [pose[4], pose[5], pose[6], pose[3]]
                
                obstacle_id = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=collision_shape,
                    baseVisualShapeIndex=visual_shape,
                    basePosition=position,
                    baseOrientation=orientation
                )
                
                self.obstacle_ids.append(obstacle_id)
                print(f"  - {name}: 位置 {position}, 半径 {radius}, 高度 {height:.3f}")
        
        return self.obstacle_ids
    
    def generate_collision_free_goal(self, world_cfg, max_attempts=50, safety_margin=0.1):
        """生成无碰撞的目标位置
        
        Args:
            world_cfg: 世界配置对象
            max_attempts: 最大尝试次数
            safety_margin: 安全距离边界
            
        Returns:
            list: 无碰撞的目标姿态 [x, y, z, qw, qx, qy, qz] 或 None
        """
        # 定义机器人工作空间范围
        workspace_bounds = {
            'x': [0.2, 0.7],    # x轴范围
            'y': [-0.5, 0.5],   # y轴范围  
            'z': [0.3, 0.8]     # z轴范围
        }
        
        for attempt in range(max_attempts):
            # 随机生成目标位置
            x = np.random.uniform(workspace_bounds['x'][0], workspace_bounds['x'][1])
            y = np.random.uniform(workspace_bounds['y'][0], workspace_bounds['y'][1])
            z = np.random.uniform(workspace_bounds['z'][0], workspace_bounds['z'][1])
            
            target_pos = np.array([x, y, z])
            
            # 检查是否与障碍物碰撞
            is_collision_free = True
            
            # 检查立方体障碍物
            if hasattr(world_cfg, 'cuboid') and world_cfg.cuboid is not None:
                for cuboid in world_cfg.cuboid:
                    if self._check_point_cuboid_collision(target_pos, cuboid, safety_margin):
                        is_collision_free = False
                        break
            
            if not is_collision_free:
                continue
                
            # 检查球体障碍物
            if hasattr(world_cfg, 'sphere') and world_cfg.sphere is not None:
                for sphere in world_cfg.sphere:
                    if self._check_point_sphere_collision(target_pos, sphere, safety_margin):
                        is_collision_free = False
                        break
            
            if not is_collision_free:
                continue
                
            # 检查胶囊体障碍物 (简化处理)
            if hasattr(world_cfg, 'capsule') and world_cfg.capsule is not None:
                for capsule in world_cfg.capsule:
                    if self._check_point_capsule_collision(target_pos, capsule, safety_margin):
                        is_collision_free = False
                        break
            
            if is_collision_free:
                # 返回无碰撞目标姿态，保持标准方向
                return [x, y, z, 1.0, 0.0, 0.0, 0.0]
        
        print(f"⚠️  经过 {max_attempts} 次尝试，未能找到无碰撞目标位置")
        return None
    
    def _check_point_cuboid_collision(self, point, cuboid, safety_margin):
        """检查点是否与立方体障碍物碰撞"""
        pose = cuboid.pose
        dims = cuboid.dims
        
        # 立方体中心位置
        center = np.array([pose[0], pose[1], pose[2]])
        
        # 简化处理：假设立方体没有旋转，检查点是否在扩展的立方体内
        half_dims = np.array([dims[0]/2 + safety_margin, 
                             dims[1]/2 + safety_margin, 
                             dims[2]/2 + safety_margin])
        
        # 检查点是否在立方体内
        return (abs(point[0] - center[0]) < half_dims[0] and
                abs(point[1] - center[1]) < half_dims[1] and
                abs(point[2] - center[2]) < half_dims[2])
    
    def _check_point_sphere_collision(self, point, sphere, safety_margin):
        """检查点是否与球体障碍物碰撞"""
        sphere_pos = np.array(sphere.position)
        radius = sphere.radius + safety_margin
        
        # 计算点到球心的距离
        distance = np.linalg.norm(point - sphere_pos)
        return distance < radius
    
    def _check_point_capsule_collision(self, point, capsule, safety_margin):
        """检查点是否与胶囊体障碍物碰撞（简化处理）"""
        pose = capsule.pose
        radius = capsule.radius + safety_margin
        
        # 简化处理：将胶囊体当作球体处理，使用pose作为中心
        capsule_center = np.array([pose[0], pose[1], pose[2]])
        distance = np.linalg.norm(point - capsule_center)
        
        # 使用胶囊体半径加上高度的一半作为安全距离
        base = np.array(capsule.base)
        tip = np.array(capsule.tip)
        height = np.linalg.norm(tip - base)
        safe_distance = radius + height/2
        
        return distance < safe_distance
    
    def visualize_trajectory(self, trajectory, start_state, goal_pose, 
                           interpolation_dt=0.02, playback_speed=1.0, 
                           show_trajectory_points=False, record_video=False,
                           video_name="trajectory_video.mp4"):
        """可视化运动轨迹"""
        print(f"\n开始可视化运动轨迹...")
        print(f"轨迹长度: {len(trajectory.position)} 个时间步")
        
        # 开始录制视频（如果需要）
        if record_video:
            if self.start_recording(video_name):
                print(f"🎬 开始录制轨迹视频...")
            else:
                print(f"❌ 录制启动失败，继续播放不录制")
                record_video = False
        
        # 清除之前的标记
        self.clear_all_markers()
        
        # 计算起始末端执行器位置
        if hasattr(start_state, 'position'):
            start_joints = start_state.position[0].cpu().numpy()
        else:
            start_joints = start_state
            
        extended_start = self._extend_joint_configuration(start_joints)
        self.set_joint_angles(extended_start)
        start_ee_pos, start_ee_quat = self.get_end_effector_pose()
        
        # 添加起始位置标记
        if start_ee_pos is not None:
            self.add_start_marker(start_ee_pos, start_ee_quat)
            print(f"起始位置: {start_ee_pos}")
        
        # 添加目标位置标记
        if hasattr(goal_pose, 'position'):
            goal_pos = goal_pose.position[0].cpu().numpy() if hasattr(goal_pose.position[0], 'cpu') else goal_pose.position[0]
            goal_quat = goal_pose.quaternion[0].cpu().numpy() if hasattr(goal_pose.quaternion[0], 'cpu') else goal_pose.quaternion[0]
        else:
            goal_pos, goal_quat = goal_pose[:3], goal_pose[3:7]
            
        self.add_goal_marker(goal_pos, goal_quat)
        print(f"目标位置: {goal_pos}")
        
        # 显示轨迹点
        if show_trajectory_points:
            print("预计算轨迹点...")
            for i in range(0, len(trajectory.position), max(1, len(trajectory.position)//20)):
                joint_config = trajectory.position[i].cpu().numpy()
                extended_config = self._extend_joint_configuration(joint_config)
                self.set_joint_angles(extended_config)
                ee_pos, _ = self.get_end_effector_pose()
                if ee_pos is not None:
                    self.add_trajectory_point_marker(ee_pos)
        
        # 播放轨迹
        print(f"\n开始播放轨迹，播放速度: {playback_speed}x")
        print("按 Ctrl+C 可以停止播放")
        
        try:
            for i, joint_positions in enumerate(trajectory.position):
                if hasattr(joint_positions, 'cpu'):
                    joint_config = joint_positions.cpu().numpy()
                else:
                    joint_config = joint_positions
                
                extended_config = self._extend_joint_configuration(joint_config)
                self.set_joint_angles(extended_config)
                p.stepSimulation()
                time.sleep(interpolation_dt / playback_speed)
                
                if i % 10 == 0:
                    progress = (i + 1) / len(trajectory.position) * 100
                    print(f"\r播放进度: {progress:.1f}%", end='', flush=True)
            
            print(f"\n轨迹播放完成！")
            
        except KeyboardInterrupt:
            print(f"\n轨迹播放被用户中断")
        
        # 获取最终位置
        final_ee_pos, final_ee_quat = self.get_end_effector_pose()
        if final_ee_pos is not None:
            final_error = np.linalg.norm(np.array(final_ee_pos) - np.array(goal_pos))
            print(f"最终位置: {final_ee_pos}")
            print(f"目标误差: {final_error:.6f}m")
        
        # 停止录制视频（如果正在录制）
        if record_video and self.is_recording():
            if self.stop_recording():
                print(f"✅ 视频录制完成: {video_name}")
                abs_video_path = os.path.abspath(f"{self.video_save_path}/{video_name}")
                print(f"📁 视频保存路径: {abs_video_path}")


def get_available_world_configs():
    """获取所有可用的世界配置文件"""
    world_configs_path = get_world_configs_path()
    world_files = []
    
    for file in os.listdir(world_configs_path):
        if file.endswith('.yml') and file.startswith('collision_'):
            world_files.append(file)
            
    return sorted(world_files)


def display_world_menu():
    """显示世界配置文件选择菜单"""
    world_files = get_available_world_configs()
    
    print("\n" + "="*60)
    print("🌍 选择世界配置文件")
    print("="*60)
    print("\n可用的世界配置文件：")
    print("-" * 40)
    
    for i, world_file in enumerate(world_files, 1):
        display_name = world_file.replace('collision_', '').replace('.yml', '')
        print(f"{i:2d}. {display_name}")
        
    print(f"\n{len(world_files)+1:2d}. 返回主菜单")
    print("-" * 40)
    
    return world_files


def display_demo_menu():
    """显示演示类型菜单"""
    print("\n" + "="*60)
    print("🎯 选择运动规划演示类型")
    print("="*60)
    print("\n可用的演示类型：")
    print("-" * 40)
    print(" 1. 简单运动规划")
    print(" 2. 避障运动规划")
    print(" 3. 多目标运动规划")
    print(" 4. 返回场景选择")
    print("-" * 40)


def ask_for_video_recording():
    """询问用户是否要录制视频"""
    while True:
        choice = input("\n🎬 是否要录制轨迹视频？(y/n): ").strip().lower()
        if choice in ['y', 'yes', '是', '要']:
            return True
        elif choice in ['n', 'no', '否', '不要']:
            return False
        else:
            print("请输入 y/n 或 yes/no")


def demo_simple_motion_planning(world_file, visualizer):
    """简单运动规划演示"""
    print(f"\n=== 简单运动规划演示 - {world_file} ===")
    
    tensor_args = TensorDeviceType()
    robot_file = "franka.yml"
    
    # 创建运动规划配置
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_file,
        world_file,
        tensor_args,
        interpolation_dt=0.02,
        collision_checker_type=CollisionCheckerType.PRIMITIVE,
        use_cuda_graph=True,
    )
    motion_gen = MotionGen(motion_gen_config)
    motion_gen.warmup()
    
    try:
        # 加载障碍物
        world_cfg = WorldConfig.from_dict(
            load_yaml(join_path(get_world_configs_path(), world_file))
        )
        obstacle_ids = visualizer.load_obstacles_from_world_config(world_cfg)
        
        # 获取起始状态
        retract_cfg = motion_gen.get_retract_config()
        start_state = JointState.from_position(retract_cfg.view(1, -1))
        
        # 设置目标姿态
        goal_pose = Pose.from_list([0.4, 0.2, 0.4, 1.0, 0.0, 0.0, 0.0])
        
        # 询问是否录制视频
        record_video = ask_for_video_recording()
        
        print(f"规划从起始位置到目标位置的轨迹...")
        
        # 规划轨迹
        result = motion_gen.plan_single(
            start_state, 
            goal_pose, 
            MotionGenPlanConfig(max_attempts=3)
        )
        
        if result.success is not None and (result.success.item() if hasattr(result.success, 'item') else result.success):
            print(f"轨迹规划成功！")
            print(f"规划时间: {result.solve_time:.4f}秒")
            print(f"轨迹时间: {result.motion_time:.4f}秒")
            
            # 获取插值轨迹
            interpolated_trajectory = result.get_interpolated_plan()
            
            # 生成视频文件名
            scene_name = world_file.replace('collision_', '').replace('.yml', '')
            video_name = f"simple_motion_{scene_name}_{datetime.now().strftime('%H%M%S')}.mp4"
            
            # 可视化轨迹
            visualizer.visualize_trajectory(
                interpolated_trajectory, 
                start_state, 
                goal_pose,
                interpolation_dt=result.interpolation_dt,
                playback_speed=0.5,
                show_trajectory_points=True,
                record_video=record_video,
                video_name=video_name
            )
            
        else:
            print(f"轨迹规划失败！状态: {result.status}")
        
    except Exception as e:
        print(f"演示过程中发生错误: {e}")


def demo_collision_avoidance(world_file, visualizer):
    """避障运动规划演示"""
    print(f"\n=== 避障运动规划演示 - {world_file} ===")
    
    tensor_args = TensorDeviceType()
    robot_file = "franka.yml"
    
    # 创建运动规划配置
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_file,
        world_file,
        tensor_args,
        interpolation_dt=0.01,
        collision_checker_type=CollisionCheckerType.PRIMITIVE,
        use_cuda_graph=True,
        num_trajopt_seeds=4,
        trajopt_tsteps=32,
    )
    motion_gen = MotionGen(motion_gen_config)
    motion_gen.warmup()
    
    try:
        # 加载障碍物
        world_cfg = WorldConfig.from_dict(
            load_yaml(join_path(get_world_configs_path(), world_file))
        )
        obstacle_ids = visualizer.load_obstacles_from_world_config(world_cfg)
        
        # 获取起始状态
        retract_cfg = motion_gen.get_retract_config()
        start_state = JointState.from_position(retract_cfg.view(1, -1))
        
        # 设置目标姿态
        goal_pose = Pose.from_list([0.4, 0.2, 0.4, 0.0, 1.0, 0.0, 0.0])
        
        # 询问是否录制视频
        record_video = ask_for_video_recording()
        
        print(f"规划避障轨迹...")
        
        # 规划轨迹
        result = motion_gen.plan_single(
            start_state, 
            goal_pose, 
            MotionGenPlanConfig(
                max_attempts=5,
                enable_graph=True,
                enable_opt=True,
                timeout=10.0
            )
        )
        
        if result.success is not None and (result.success.item() if hasattr(result.success, 'item') else result.success):
            print(f"避障轨迹规划成功！")
            print(f"规划时间: {result.solve_time:.4f}秒")
            print(f"轨迹时间: {result.motion_time:.4f}秒")
            print(f"状态: {result.status}")
            
            # 获取插值轨迹
            interpolated_trajectory = result.get_interpolated_plan()
            
            # 生成视频文件名
            scene_name = world_file.replace('collision_', '').replace('.yml', '')
            video_name = f"collision_avoidance_{scene_name}_{datetime.now().strftime('%H%M%S')}.mp4"
            
            # 可视化轨迹
            visualizer.visualize_trajectory(
                interpolated_trajectory, 
                start_state, 
                goal_pose,
                interpolation_dt=result.interpolation_dt,
                playback_speed=0.3,
                show_trajectory_points=True,
                record_video=record_video,
                video_name=video_name
            )
            
        else:
            print(f"避障轨迹规划失败！状态: {result.status}")
        
    except Exception as e:
        print(f"演示过程中发生错误: {e}")


def demo_multiple_goals(world_file, visualizer):
    """多目标运动规划演示"""
    print(f"\n=== 多目标运动规划演示 - {world_file} ===")
    
    tensor_args = TensorDeviceType()
    robot_file = "franka.yml"
    
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
        # 加载障碍物
        world_cfg = WorldConfig.from_dict(
            load_yaml(join_path(get_world_configs_path(), world_file))
        )
        obstacle_ids = visualizer.load_obstacles_from_world_config(world_cfg)
        
        # 定义初始目标位置
        initial_goal_positions = [
            [0.4, 0.3, 0.5, 1.0, 0.0, 0.0, 0.0],    # 目标1
            [0.4, -0.3, 0.3, 1.0, 0.0, 0.0, 0.0],   # 目标2
            [0.2, 0.0, 0.6, 1.0, 0.0, 0.0, 0.0],    # 目标3
        ]
        
        # 获取起始状态
        retract_cfg = motion_gen.get_retract_config()
        current_state = JointState.from_position(retract_cfg.view(1, -1))
        
        # 询问是否录制视频
        record_video = ask_for_video_recording()
        
        successful_goals = 0
        max_goals = 3  # 最多尝试3个目标
        
        for i in range(max_goals):
            print(f"\n=== 规划到目标 {i+1} ===")
            
            # 选择目标位置
            if i < len(initial_goal_positions):
                goal_pos = initial_goal_positions[i]
                print(f"使用预设目标: {goal_pos[:3]}")
            else:
                # 如果超出预设目标，生成新的随机目标
                goal_pos = visualizer.generate_collision_free_goal(world_cfg)
                if goal_pos is None:
                    print(f"❌ 无法生成无碰撞目标，跳过目标 {i+1}")
                    continue
                print(f"生成随机目标: {goal_pos[:3]}")
            
            # 创建目标姿态
            goal_pose = Pose.from_list(goal_pos)
            
            # 规划轨迹
            result = motion_gen.plan_single(
                current_state, 
                goal_pose, 
                MotionGenPlanConfig(max_attempts=3)
            )
            
            if result.success is not None and (result.success.item() if hasattr(result.success, 'item') else result.success):
                print(f"✅ 到目标 {i+1} 的轨迹规划成功！")
                print(f"规划时间: {result.solve_time:.4f}秒")
                
                # 获取插值轨迹
                interpolated_trajectory = result.get_interpolated_plan()
                
                # 生成视频文件名
                scene_name = world_file.replace('collision_', '').replace('.yml', '')
                video_name = f"multi_goal_{scene_name}_target{i+1}_{datetime.now().strftime('%H%M%S')}.mp4"
                
                # 可视化轨迹
                visualizer.visualize_trajectory(
                    interpolated_trajectory, 
                    current_state, 
                    goal_pose,
                    interpolation_dt=result.interpolation_dt,
                    playback_speed=0.5,
                    show_trajectory_points=(i == 0),
                    record_video=record_video,
                    video_name=video_name
                )
                
                # 更新当前状态为轨迹的终点
                if len(interpolated_trajectory.position) > 0:
                    final_joint_state = interpolated_trajectory.position[-1]
                    if torch.is_tensor(final_joint_state) and hasattr(final_joint_state, 'view'):
                        current_state = JointState.from_position(final_joint_state.view(1, -1))
                    else:
                        # 如果final_joint_state不是tensor，需要转换
                        if isinstance(final_joint_state, (list, np.ndarray)):
                            current_state = JointState.from_position(
                                torch.tensor(final_joint_state, dtype=torch.float32).view(1, -1)
                            )
                        else:
                            current_state = JointState.from_position(
                                torch.tensor([final_joint_state], dtype=torch.float32).view(1, -1)
                            )
                
                successful_goals += 1
                
                if i < max_goals - 1:
                    print(f"按回车键继续到下一个目标...")
                    input()
                    
            else:
                print(f"❌ 到目标 {i+1} 的轨迹规划失败！状态: {result.status}")
                print(f"🔄 尝试生成新的无碰撞目标...")
                
                # 尝试生成无碰撞目标
                max_retries = 3
                for retry in range(max_retries):
                    new_goal_pos = visualizer.generate_collision_free_goal(world_cfg)
                    if new_goal_pos is None:
                        print(f"⚠️  重试 {retry+1}/{max_retries}: 无法生成无碰撞目标")
                        continue
                    
                    print(f"🎯 重试 {retry+1}/{max_retries}: 新目标 {new_goal_pos[:3]}")
                    new_goal_pose = Pose.from_list(new_goal_pos)
                    
                    # 用新目标重新规划
                    retry_result = motion_gen.plan_single(
                        current_state, 
                        new_goal_pose, 
                        MotionGenPlanConfig(max_attempts=3)
                    )
                    
                    if retry_result.success is not None and (retry_result.success.item() if hasattr(retry_result.success, 'item') else retry_result.success):
                        print(f"✅ 使用新目标的轨迹规划成功！")
                        print(f"规划时间: {retry_result.solve_time:.4f}秒")
                        
                        # 获取插值轨迹
                        interpolated_trajectory = retry_result.get_interpolated_plan()
                        
                        # 生成视频文件名
                        scene_name = world_file.replace('collision_', '').replace('.yml', '')
                        video_name = f"multi_goal_{scene_name}_target{i+1}_retry{retry+1}_{datetime.now().strftime('%H%M%S')}.mp4"
                        
                        # 可视化轨迹
                        visualizer.visualize_trajectory(
                            interpolated_trajectory, 
                            current_state, 
                            new_goal_pose,
                            interpolation_dt=retry_result.interpolation_dt,
                            playback_speed=0.5,
                            show_trajectory_points=(i == 0),
                            record_video=record_video,
                            video_name=video_name
                        )
                        
                        # 更新当前状态
                        if len(interpolated_trajectory.position) > 0:
                            final_joint_state = interpolated_trajectory.position[-1]
                            if torch.is_tensor(final_joint_state) and hasattr(final_joint_state, 'view'):
                                current_state = JointState.from_position(final_joint_state.view(1, -1))
                            else:
                                if isinstance(final_joint_state, (list, np.ndarray)):
                                    current_state = JointState.from_position(
                                        torch.tensor(final_joint_state, dtype=torch.float32).view(1, -1)
                                    )
                                else:
                                    current_state = JointState.from_position(
                                        torch.tensor([final_joint_state], dtype=torch.float32).view(1, -1)
                                    )
                        
                        successful_goals += 1
                        
                        if i < max_goals - 1:
                            print(f"按回车键继续到下一个目标...")
                            input()
                        break
                    else:
                        print(f"❌ 重试 {retry+1}/{max_retries}: 新目标规划仍然失败")
                
                else:
                    print(f"❌ 经过 {max_retries} 次重试，仍无法找到可达目标，跳过目标 {i+1}")
        
        print(f"\n🎉 多目标规划完成！成功到达 {successful_goals}/{max_goals} 个目标")
        
    except Exception as e:
        print(f"演示过程中发生错误: {e}")


def main():
    """主函数"""
    setup_curobo_logger("error")
    
    print("🚀 启动场景选择的运动规划可视化演示！")
    
    while True:
        try:
            # 显示世界配置文件选择菜单
            world_files = display_world_menu()
            
            world_choice = input(f"\n请选择世界配置文件 (1-{len(world_files)+1}): ").strip()
            
            if world_choice == str(len(world_files) + 1) or world_choice.lower() in ['q', 'quit', 'exit']:
                print("\n👋 再见！")
                break
                
            try:
                world_choice_idx = int(world_choice) - 1
                if 0 <= world_choice_idx < len(world_files):
                    selected_world_file = world_files[world_choice_idx]
                    print(f"\n✅ 选择了世界配置: {selected_world_file}")
                    
                    # 创建可视化器
                    visualizer = SceneMotionGenVisualizer(gui=True)
                    
                    try:
                        # 演示选择循环
                        while True:
                            display_demo_menu()
                            
                            demo_choice = input("\n请选择演示类型 (1-4): ").strip()
                            
                            if demo_choice == "1":
                                demo_simple_motion_planning(selected_world_file, visualizer)
                            elif demo_choice == "2":
                                demo_collision_avoidance(selected_world_file, visualizer)
                            elif demo_choice == "3":
                                demo_multiple_goals(selected_world_file, visualizer)
                            elif demo_choice == "4":
                                break
                            else:
                                print("❌ 无效的选择，请重新输入")
                                continue
                                
                            print("\n按回车键继续...")
                            input()
                            
                    finally:
                        visualizer.disconnect()
                        
                else:
                    print("❌ 无效的选择，请重新输入")
                    
            except ValueError:
                print("❌ 请输入有效的数字")
                
        except KeyboardInterrupt:
            print("\n\n👋 用户中断，退出程序")
            break
        except Exception as e:
            print(f"❌ 发生错误: {e}")


if __name__ == "__main__":
    main() 