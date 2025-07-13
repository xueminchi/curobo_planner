#!/usr/bin/env python3
"""
Pick and Place 演示脚本 (带碰撞距离监控)
基于pick_and_place_fixed.py，添加了机械臂和障碍物之间的距离打印功能
"""

import time
import numpy as np
import pybullet as p
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
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig

# Local
from pybullet_kinematics_visualization import PyBulletKinematicsVisualizer

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class CollisionDistanceMonitor:
    """碰撞距离监控器"""
    
    def __init__(self, motion_gen: MotionGen):
        self.motion_gen = motion_gen
        self.tensor_args = motion_gen.tensor_args
        
        # 创建RobotWorld实例用于碰撞距离计算
        robot_config = motion_gen.robot_cfg
        world_config = motion_gen.world_model
        
        # 创建RobotWorld配置
        robot_world_config = RobotWorldConfig.load_from_config(
            robot_config, 
            world_config, 
            collision_activation_distance=0.0,
            tensor_args=self.tensor_args
        )
        
        # 创建RobotWorld实例
        self.robot_world = RobotWorld(robot_world_config)
        
        print("✅ 碰撞距离监控器初始化完成")
    
    def compute_collision_distance(self, joint_position: torch.Tensor) -> tuple:
        """
        计算当前关节位置的碰撞距离
        
        Args:
            joint_position: 关节位置 tensor [batch, dof]
            
        Returns:
            tuple: (world_distance, self_collision_distance)
        """
        try:
            # 确保输入是正确的形状
            if len(joint_position.shape) == 1:
                joint_position = joint_position.unsqueeze(0)
            
            # 使用RobotWorld计算碰撞距离
            d_world, d_self = self.robot_world.get_world_self_collision_distance_from_joints(
                joint_position
            )
            
            return d_world, d_self
            
        except Exception as e:
            print(f"⚠️ 碰撞距离计算失败: {e}")
            return None, None
    
    def print_collision_distance(self, joint_position: torch.Tensor, stage: str = ""):
        """
        打印当前关节位置的碰撞距离信息
        
        Args:
            joint_position: 关节位置
            stage: 当前阶段名称
        """
        d_world, d_self = self.compute_collision_distance(joint_position)
        
        if d_world is not None and d_self is not None:
            # 转换为numpy并获取标量值
            world_dist = d_world.detach().cpu().numpy()
            self_dist = d_self.detach().cpu().numpy()
            
            # 如果是批次，取第一个
            if len(world_dist.shape) > 0:
                world_dist = world_dist[0] if world_dist.shape[0] > 0 else world_dist
            if len(self_dist.shape) > 0:
                self_dist = self_dist[0] if self_dist.shape[0] > 0 else self_dist
            
            print(f"📏 [{stage}] 碰撞距离 - 世界: {world_dist:.4f}m, 自碰撞: {self_dist:.4f}m")
            
            # 如果距离太小，给出警告
            if world_dist < 0.01:
                print(f"⚠️  [{stage}] 警告: 与世界障碍物距离过近! ({world_dist:.4f}m)")
            if self_dist < 0.01:
                print(f"⚠️  [{stage}] 警告: 自碰撞距离过近! ({self_dist:.4f}m)")


class PickAndPlaceVisualizerWithCollisionDistance(PyBulletKinematicsVisualizer):
    """扩展的可视化器，专门用于Pick and Place演示，带碰撞距离监控"""
    
    def __init__(self, robot_config_name="franka.yml", gui=True):
        super().__init__(robot_config_name, gui)
        self.obstacle_ids = []
        self.target_object_id = None
        self.target_markers = []
        self.sphere_marker_ids = []
        self.sphere_relative_positions = []
        self.motion_gen = None
        self.attached_sphere_positions = []
        self.ee_to_sphere_transforms = []
        
        # 碰撞距离监控器
        self.collision_monitor = None
        
    def set_motion_gen(self, motion_gen: MotionGen):
        """设置MotionGen实例并初始化碰撞监控器"""
        self.motion_gen = motion_gen
        self.collision_monitor = CollisionDistanceMonitor(motion_gen)
        
    def create_world_with_target_object(self):
        """创建包含目标物体和障碍物的世界"""
        self.clear_obstacles()
        
        # 创建目标立方体 - 位置调整到更合适的地方
        target_dims = [0.05, 0.05, 0.05]
        target_position = [0.45, 0.35, 0.025]
        
        target_collision_shape = p.createCollisionShape(
            p.GEOM_BOX, 
            halfExtents=[target_dims[0]/2, target_dims[1]/2, target_dims[2]/2]
        )
        target_visual_shape = p.createVisualShape(
            p.GEOM_BOX, 
            halfExtents=[target_dims[0]/2, target_dims[1]/2, target_dims[2]/2],
            rgbaColor=[1.0, 0.2, 0.2, .3]
        )
        
        self.target_object_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=target_collision_shape,
            baseVisualShapeIndex=target_visual_shape,
            basePosition=target_position
        )
        
        print(f"📦 创建目标立方体: 位置 {target_position}, 尺寸 {target_dims}")
        
        # 创建障碍物
        obstacles = [
            {
                "position": [-0.2, -0.3, 0.6],
                "dims": [0.08, 0.08, 1.2],
                "color": [0.2, 0.2, 0.8, 0.7]
            },
            {
                "position": [0.6, 0.0, 0.55],
                "dims": [0.45, 0.1, 1.1],
                "color": [0.2, 0.8, 0.2, 0.7]
            }
        ]
        
        for i, obs in enumerate(obstacles):
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
            print(f"  🚧 障碍物 {i+1}: 位置 {obs['position']}, 尺寸 {obs['dims']}")
        
        return target_position, target_dims
    
    def add_marker(self, position, size=0.03, color=[0, 1, 1, 0.8], marker_type="sphere"):
        """通用标记添加方法"""
        if marker_type == "sphere":
            visual_shape = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=size,
                rgbaColor=color
            )
        else:
            visual_shape = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[size, size, size],
                rgbaColor=color
            )
        
        marker_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_shape,
            basePosition=position
        )
        
        self.target_markers.append(marker_id)
        return marker_id
    
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
        
        for sphere_id in self.sphere_marker_ids:
            try:
                p.removeBody(sphere_id)
            except:
                pass
        self.sphere_marker_ids.clear()
    
    def safe_get_joint_state_from_trajectory(self, trajectory, index=-1):
        """安全地从轨迹中获取关节状态"""
        try:
            if hasattr(trajectory, 'position'):
                final_position = trajectory.position[index]
                
                if torch.is_tensor(final_position):
                    if final_position.dim() == 1:
                        return JointState.from_position(final_position.view(1, -1))
                    elif final_position.dim() == 2:
                        return JointState.from_position(final_position)
                    else:
                        return JointState.from_position(final_position[0].view(1, -1))
                else:
                    if isinstance(final_position, (list, np.ndarray)):
                        final_position = torch.tensor(final_position, dtype=torch.float32)
                    else:
                        final_position = torch.tensor([final_position], dtype=torch.float32)
                    
                    return JointState.from_position(final_position.view(1, -1))
            else:
                print("❌ 轨迹对象没有position属性")
                return None
                
        except Exception as e:
            print(f"❌ 处理轨迹时出错: {e}")
            return None
    
    def visualize_trajectory_with_collision_monitoring(self, trajectory, stage_name="轨迹", 
                                                    interpolation_dt=0.02, playback_speed=1.0, 
                                                    show_object_attached=False,
                                                    monitor_frequency=5):
        """可视化轨迹并监控碰撞距离"""
        print(f"\n🎬 开始播放轨迹: {stage_name}")
        print(f"轨迹长度: {len(trajectory.position)} 个时间步")
        
        try:
            for i, joint_positions in enumerate(trajectory.position):
                if hasattr(joint_positions, 'cpu'):
                    joint_config = joint_positions.cpu().numpy()
                else:
                    joint_config = joint_positions
                
                # 转换为torch tensor用于碰撞距离计算
                joint_tensor = torch.tensor(joint_config, dtype=torch.float32)
                if joint_tensor.device != self.collision_monitor.tensor_args.device:
                    joint_tensor = joint_tensor.to(self.collision_monitor.tensor_args.device)
                
                # 定期打印碰撞距离
                if self.collision_monitor is not None and i % monitor_frequency == 0:
                    progress = (i + 1) / len(trajectory.position) * 100
                    stage_info = f"{stage_name} - 进度{progress:.1f}%"
                    self.collision_monitor.print_collision_distance(joint_tensor, stage_info)
                
                extended_config = self._extend_joint_configuration(joint_config)
                self.set_joint_angles(extended_config)
                
                # 如果物体已附加，更新物体位置跟随末端执行器
                if show_object_attached and self.target_object_id is not None:
                    ee_pos, ee_quat = self.get_end_effector_pose()
                    if ee_pos is not None:
                        object_pos = [ee_pos[0], ee_pos[1], ee_pos[2] - 0.05]
                        p.resetBasePositionAndOrientation(
                            self.target_object_id, 
                            object_pos, 
                            ee_quat
                        )
                
                # 更新球体标记位置
                if show_object_attached and len(self.sphere_marker_ids) > 0 and self.motion_gen is not None:
                    self._update_sphere_markers(joint_config)
                
                p.stepSimulation()
                time.sleep(interpolation_dt / playback_speed)
                
                if i % 10 == 0:
                    progress = (i + 1) / len(trajectory.position) * 100
                    print(f"\r播放进度: {progress:.1f}%", end='', flush=True)
            
            print(f"\n✅ 轨迹播放完成：{stage_name}")
            
            # 轨迹结束时打印最终碰撞距离
            if self.collision_monitor is not None:
                final_joint_positions = trajectory.position[-1]
                if hasattr(final_joint_positions, 'cpu'):
                    final_joint_config = final_joint_positions.cpu().numpy()
                else:
                    final_joint_config = final_joint_positions
                
                final_joint_tensor = torch.tensor(final_joint_config, dtype=torch.float32)
                if final_joint_tensor.device != self.collision_monitor.tensor_args.device:
                    final_joint_tensor = final_joint_tensor.to(self.collision_monitor.tensor_args.device)
                
                self.collision_monitor.print_collision_distance(final_joint_tensor, f"{stage_name} - 最终位置")
            
        except KeyboardInterrupt:
            print(f"\n⏹️  轨迹播放被中断")
    
    def _update_sphere_markers(self, joint_config):
        """更新球体标记位置"""
        if len(self.sphere_marker_ids) == 0:
            return
            
        try:
            extended_config = self._extend_joint_configuration(joint_config)
            self.set_joint_angles(extended_config)
            ee_pos, ee_quat = self.get_end_effector_pose()
            
            if ee_pos is None:
                return
                
            if len(self.sphere_relative_positions) == 0 and len(self.attached_sphere_positions) > 0:
                initial_ee_pos = ee_pos
                self.sphere_relative_positions = []
                for abs_pos in self.attached_sphere_positions:
                    relative_pos = [
                        abs_pos[0] - initial_ee_pos[0],
                        abs_pos[1] - initial_ee_pos[1], 
                        abs_pos[2] - initial_ee_pos[2]
                    ]
                    self.sphere_relative_positions.append(relative_pos)
            
            for i, sphere_id in enumerate(self.sphere_marker_ids):
                if i < len(self.sphere_relative_positions):
                    relative_pos = self.sphere_relative_positions[i]
                    new_pos = [
                        ee_pos[0] + relative_pos[0],
                        ee_pos[1] + relative_pos[1],
                        ee_pos[2] + relative_pos[2]
                    ]
                    p.resetBasePositionAndOrientation(
                        sphere_id,
                        new_pos,
                        [0, 0, 0, 1]
                    )
                    
        except Exception as e:
            pass


def create_optimized_world():
    """创建优化的Pick and Place世界配置"""
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
                "dims": [0.45, 0.1, 1.1],
                "pose": [0.6, 0.0, 0.55, 1, 0, 0, 0.0]
            }
        }
    }
    
    return world_config


def demo_pick_and_place_with_collision_distance():
    """带碰撞距离监控的Pick and Place演示"""
    print("🤖 Pick and Place 演示 (带碰撞距离监控)")
    print("="*60)
    
    tensor_args = TensorDeviceType()
    robot_file = "franka.yml"
    
    # 创建优化的世界配置
    world_config = create_optimized_world()
    
    # 创建运动规划配置
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_file,
        world_config,
        tensor_args,
        interpolation_dt=0.02,
        collision_checker_type=CollisionCheckerType.PRIMITIVE,
        use_cuda_graph=True,
        num_trajopt_seeds=6,
        num_graph_seeds=4,
    )
    motion_gen = MotionGen(motion_gen_config)
    motion_gen.warmup()
    
    # 创建带碰撞距离监控的可视化器
    visualizer = PickAndPlaceVisualizerWithCollisionDistance(gui=True)
    
    # 设置motion_gen引用
    visualizer.set_motion_gen(motion_gen)
    
    try:
        # 更新世界配置
        world_cfg = WorldConfig.from_dict(world_config)
        motion_gen.update_world(world_cfg)
        print(f"🌍 已将障碍物配置加载到CuRobo运动规划器中")

        # 创建可视化世界
        target_pos, target_dims = visualizer.create_world_with_target_object()
        
        # 定义关键位置
        approach_height = 0.20
        grasp_height = 0.10
        
        approach_position = [target_pos[0], target_pos[1], target_pos[2] + target_dims[2]/2 + approach_height]
        grasp_position = [target_pos[0], target_pos[1], target_pos[2] + target_dims[2]/2 + grasp_height]
        place_position = [0.45, -0.45, 0.55]
        
        # 添加可视化标记
        visualizer.add_marker(approach_position, 0.02, [1, 0.5, 0, 0.8])
        visualizer.add_marker(grasp_position, 0.025, [1, 1, 0, 0.9])
        visualizer.add_marker(place_position, 0.03, [0, 1, 1, 0.8])
        
        print(f"🔶 接近位置: {approach_position}")
        print(f"🟡 抓取位置: {grasp_position}")
        print(f"📍 放置位置: {place_position}")
        
        # 获取起始状态
        retract_cfg = motion_gen.get_retract_config()
        start_state = JointState.from_position(retract_cfg.view(1, -1))
        
        # 打印起始状态的碰撞距离
        print(f"\n📊 起始状态碰撞距离检查:")
        visualizer.collision_monitor.print_collision_distance(retract_cfg, "起始状态")
        
        input("\n按回车键开始演示...")
        
        # === 阶段1: 移动到接近位置 ===
        print(f"\n🚀 阶段1: 规划到接近位置...")
        approach_pose = Pose.from_list([
            approach_position[0], approach_position[1], approach_position[2], 
            0.0, 1.0, 0.0, 0.0
        ])
        
        result1 = motion_gen.plan_single(
            start_state, 
            approach_pose, 
            MotionGenPlanConfig(
                max_attempts=8, 
                enable_graph=True,
                enable_opt=True,
                timeout=15.0
            )
        )
        
        if result1.success is not None and (result1.success.item() if hasattr(result1.success, 'item') else result1.success):
            print(f"✅ 到接近位置的规划成功！")
            
            # 播放轨迹并监控碰撞距离
            trajectory1 = result1.get_interpolated_plan()
            visualizer.visualize_trajectory_with_collision_monitoring(
                trajectory1, 
                stage_name="移动到接近位置",
                interpolation_dt=result1.interpolation_dt,
                playback_speed=0.5,
                monitor_frequency=10
            )
            
            current_state = visualizer.safe_get_joint_state_from_trajectory(trajectory1)
            if current_state is None:
                print("❌ 无法获取轨迹终点状态")
                return
            
        else:
            print(f"❌ 到接近位置的规划失败！状态: {result1.status}")
            return
        
        input("\n按回车键继续到抓取位置...")
        
        # === 阶段2: 移动到抓取位置 ===
        print(f"\n🎯 阶段2: 规划到抓取位置...")
        grasp_pose = Pose.from_list([
            grasp_position[0], grasp_position[1], grasp_position[2], 
            0.0, 1.0, 0.0, 0.0
        ])
        
        result2 = motion_gen.plan_single(
            current_state, 
            grasp_pose, 
            MotionGenPlanConfig(
                max_attempts=5,
                enable_graph=True,
                enable_opt=True
            )
        )
        
        if result2.success is not None and (result2.success.item() if hasattr(result2.success, 'item') else result2.success):
            print(f"✅ 到抓取位置的规划成功！")
            
            # 播放轨迹并监控碰撞距离
            trajectory2 = result2.get_interpolated_plan()
            visualizer.visualize_trajectory_with_collision_monitoring(
                trajectory2, 
                stage_name="移动到抓取位置",
                interpolation_dt=result2.interpolation_dt,
                playback_speed=0.5,
                monitor_frequency=8
            )
            
            current_state = visualizer.safe_get_joint_state_from_trajectory(trajectory2)
            if current_state is None:
                print("❌ 无法获取轨迹终点状态")
                return
            
        else:
            print(f"❌ 到抓取位置的规划失败！状态: {result2.status}")
            return
        
        input("\n按回车键继续到抓取阶段...")
        
        # === 阶段3: 抓取物体 ===
        print(f"\n🤏 阶段3: 抓取物体...")
        
        # 抓取前的碰撞距离检查
        print(f"📊 抓取前碰撞距离检查:")
        visualizer.collision_monitor.print_collision_distance(current_state.position, "抓取前")
        
        success = motion_gen.attach_objects_to_robot(
            joint_state=current_state,
            object_names=["target_cube"],
            surface_sphere_radius=0.01,
            remove_obstacles_from_world_config=False
        )
        
        if success:
            print("✅ 成功将立方体附加到机器人！")
            
            # 抓取后的碰撞距离检查
            print(f"📊 抓取后碰撞距离检查:")
            visualizer.collision_monitor.print_collision_distance(current_state.position, "抓取后")
            
            # 重新初始化碰撞监控器以反映附加物体
            visualizer.collision_monitor = CollisionDistanceMonitor(motion_gen)
            
        else:
            print("❌ 物体附加失败！")
            return
        
        input("\n按回车键继续到移动阶段...")
        
        # === 阶段4: 移动到放置位置 ===
        print(f"\n🚚 阶段4: 规划到放置位置（携带物体）...")
        
        place_pose = Pose.from_list([
            place_position[0], place_position[1], place_position[2], 
            0, 1, 0, 0
        ])
        
        plan_config_4 = MotionGenPlanConfig(
            enable_graph=True,
            enable_opt=True,
            max_attempts=3,
            timeout=10.0,
            check_start_validity=False
        )
        
        result4 = motion_gen.plan_single(
            start_state=current_state,
            goal_pose=place_pose,
            plan_config=plan_config_4
        )
        
        if result4.success is not None and result4.success.item():
            print("✅ 到放置位置的规划成功！")
            
            # 播放轨迹并监控碰撞距离
            trajectory3 = result4.get_interpolated_plan()
            visualizer.visualize_trajectory_with_collision_monitoring(
                trajectory3, 
                stage_name="移动到放置位置（携带物体）",
                interpolation_dt=result4.interpolation_dt,
                playback_speed=0.5,
                show_object_attached=True,
                monitor_frequency=10
            )
            
            current_state = visualizer.safe_get_joint_state_from_trajectory(trajectory3)
            if current_state is None:
                print("❌ 无法获取最终轨迹终点状态")
                return
        else:
            print(f"❌ 到放置位置的规划失败: {result4.status}")
            return
        
        input("\n按回车键继续到放置阶段...")
        
        # === 阶段5: 放置物体 ===
        print(f"\n📤 阶段5: 放置物体...")
        
        # 放置前的碰撞距离检查
        print(f"📊 放置前碰撞距离检查:")
        visualizer.collision_monitor.print_collision_distance(current_state.position, "放置前")
        
        motion_gen.detach_object_from_robot()
        print(f"✅ 成功将立方体从机器人分离！")
        
        # 放置后的碰撞距离检查
        print(f"📊 放置后碰撞距离检查:")
        visualizer.collision_monitor.print_collision_distance(current_state.position, "放置后")
        
        # 在PyBullet中更新物体位置
        if visualizer.target_object_id is not None:
            final_object_position = [place_position[0], place_position[1], place_position[2] - 0.05]
            p.resetBasePositionAndOrientation(
                visualizer.target_object_id, 
                final_object_position, 
                [0, 1, 0, 0]
            )
        
        print(f"\n🎉 Pick and Place 演示完成！")
        print(f"📊 演示统计:")
        print(f"  ✅ 所有阶段成功完成")
        print(f"  📏 全程监控了机械臂与障碍物的碰撞距离")
        print(f"  🎯 物体成功从 {target_pos} 移动到 {place_position}")
        
        input("\n按回车键退出演示...")
        
    except Exception as e:
        print(f"❌ 演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        visualizer.disconnect()


def main():
    """主函数"""
    print("🤖 Pick and Place 演示 (带碰撞距离监控)")
    print("这个版本在运动过程中持续监控机械臂与障碍物的距离")
    print("\n✨ 功能:")
    print("• 📏 实时监控机械臂与世界障碍物的距离")
    print("• 🔍 监控机械臂的自碰撞距离")
    print("• ⚠️  距离过近时自动警告")
    print("• 🎯 在各个阶段详细显示碰撞距离信息")
    print("• 📊 轨迹播放过程中定期打印距离数据")
    
    response = input("\n开始带碰撞距离监控的Pick and Place演示吗？(y/n): ")
    if response.lower() in ['y', 'yes', '是']:
        demo_pick_and_place_with_collision_distance()
    else:
        print("演示已取消")


if __name__ == "__main__":
    setup_curobo_logger("error")
    main() 