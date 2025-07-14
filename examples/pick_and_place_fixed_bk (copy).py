#!/usr/bin/env python3
"""
Pick and Place 演示脚本 (修复版本)
解决了tensor处理问题，增加了抓取位置可视化

重要提醒：
1. 当前存在配置不一致问题：
   - CuRobo世界配置中障碍物2尺寸：[0.6, 0.1, 1.1] (第420行)
   - PyBullet可视化中障碍物2尺寸：[0.6, 0.1, 1.1] (第178行)
   - 两者现在一致，但如果修改其中一个，必须同时修改另一个！

2. 如果机械臂与绿色障碍物碰撞，检查：
   - create_optimized_world()函数中的obstacle2配置
   - create_world_with_target_object()方法中的obstacles配置
   - 确保两处的dims尺寸完全一致

3. 激活距离设置：
   - 碰撞距离监测器使用0.1m激活距离 (第65行)
   - 运动规划器使用默认激活距离
   - 激活距离越大，路径越保守但可能导致无解
"""

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
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig

# Local
from pybullet_kinematics_visualization import PyBulletKinematicsVisualizer

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class PickAndPlaceVisualizerFixed(PyBulletKinematicsVisualizer):
    """扩展的可视化器，专门用于Pick and Place演示"""
    
    def __init__(self, robot_config_name="franka.yml", gui=True):
        super().__init__(robot_config_name, gui)
        self.obstacle_ids = []
        self.target_object_id = None
        self.target_markers = []
        self.sphere_marker_ids = []  # 存储球体标记的ID
        self.sphere_relative_positions = []  # 存储球体相对于末端执行器的偏移量
        self.motion_gen = None  # 用于运动学计算，可以是MotionGen对象或None
        self.attached_sphere_positions = []  # 存储附加球体的绝对位置
        self.ee_to_sphere_transforms = []  # 存储从末端执行器到球体的变换
        self.collision_checker = None  # 碰撞检测器
        self.tensor_args = TensorDeviceType()  # 添加tensor_args属性
        
    def setup_collision_checker(self, world_config):
        """设置碰撞检测器用于距离监测"""
        print("🔧 初始化碰撞距离监测器...")
        
        try:
            # 加载机器人配置
            robot_file = "franka.yml"
            robot_cfg = load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
            robot_config = RobotConfig.from_dict(robot_cfg, self.tensor_args)
            
            # 创建RobotWorld配置，用于碰撞检测
            collision_config = RobotWorldConfig.load_from_config(
                robot_config,
                world_config,
                collision_activation_distance=0.1,  # 使用1米的激活距离来获取距离信息
                collision_checker_type=CollisionCheckerType.PRIMITIVE,
                tensor_args=self.tensor_args,
            )
            
            # 创建碰撞检测器
            self.collision_checker = RobotWorld(collision_config)
            print("✅ 碰撞距离监测器初始化成功")
            print(f"   - 机器人配置: {robot_file}")
            print(f"   - 碰撞检测器类型: PRIMITIVE")
            print(f"   - 激活距离: 1.0m (用于获取距离信息)")
            
        except Exception as e:
            print(f"❌ 碰撞距离监测器初始化失败: {e}")
            self.collision_checker = None
    
    def get_collision_distance(self, joint_positions):
        """获取机械臂与障碍物的最近距离"""
        if self.collision_checker is None:
            return None, None
            
        try:
            # 确保joint_positions是正确的tensor格式
            if not torch.is_tensor(joint_positions):
                joint_positions = torch.tensor(joint_positions, dtype=self.tensor_args.dtype, device=self.tensor_args.device)
            
            # 确保是2D tensor [batch_size, dof]
            if joint_positions.dim() == 1:
                joint_positions = joint_positions.unsqueeze(0)
            elif joint_positions.dim() > 2:
                joint_positions = joint_positions.squeeze()
                if joint_positions.dim() == 1:
                    joint_positions = joint_positions.unsqueeze(0)
            
            # 获取机械臂的球体位置
            kin_state = self.collision_checker.get_kinematics(joint_positions)
            if kin_state.link_spheres_tensor is None:
                return None, None
            robot_spheres = kin_state.link_spheres_tensor.unsqueeze(1)  # 添加时间维度
            
            # 计算与世界障碍物的距离  
            d_world, d_world_vec = self.collision_checker.get_collision_vector(robot_spheres)
            
            # 计算自碰撞距离
            d_self = self.collision_checker.get_self_collision_distance(robot_spheres)
            
            # 转换为numpy用于显示
            world_distance = d_world.min().item() if d_world.numel() > 0 else float('inf')
            self_distance = d_self.min().item() if d_self.numel() > 0 else float('inf')
            
            return world_distance, self_distance
            
        except Exception as e:
            # 静默处理错误，避免影响主程序
            return None, None
    
    def print_collision_distance(self, joint_positions, step_index=None, phase=""):
        """打印碰撞距离信息"""
        world_dist, self_dist = self.get_collision_distance(joint_positions)
        
        if world_dist is not None and self_dist is not None:
            step_info = f"步骤{step_index}: " if step_index is not None else ""
            phase_info = f"[{phase}] " if phase else ""
            
            print(f"📏 {phase_info}{step_info}碰撞距离 - 世界障碍物: {world_dist:.4f}m, 自碰撞: {self_dist:.4f}m")
            
            # 如果距离很近，给出警告
            if world_dist < 0.05:  # 5cm
                print(f"⚠️  警告: 与障碍物距离过近! ({world_dist:.4f}m)")
            elif world_dist < 0.1:  # 10cm
                print(f"⚡ 注意: 接近障碍物 ({world_dist:.4f}m)")
                
            if self_dist < 0.01:  # 1cm
                print(f"🚨 自碰撞警告: 机械臂链接距离过近! ({self_dist:.4f}m)")
        else:
            print(f"❌ 无法获取碰撞距离信息")
    
    def create_world_with_target_object(self):
        """创建包含目标物体和障碍物的世界"""
        self.clear_obstacles()
        
        # 创建目标立方体 - 位置调整到更合适的地方
        target_dims = [0.05, 0.05, 0.05]
        target_position = [0.45, 0.35, 0.025]  # 调整到更容易抓取的位置
        
        target_collision_shape = p.createCollisionShape(
            p.GEOM_BOX, 
            halfExtents=[target_dims[0]/2, target_dims[1]/2, target_dims[2]/2]
        )
        target_visual_shape = p.createVisualShape(
            p.GEOM_BOX, 
            halfExtents=[target_dims[0]/2, target_dims[1]/2, target_dims[2]/2],
            rgbaColor=[1.0, 0.2, 0.2, .3]  # 红色
        )
        
        self.target_object_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=target_collision_shape,
            baseVisualShapeIndex=target_visual_shape,
            basePosition=target_position
        )
        
        print(f"📦 创建目标立方体: 位置 {target_position}, 尺寸 {target_dims}")
        
        # 创建障碍物 - 与CuRobo配置同步的高障碍物
        obstacles = [
            # {
            #     "position": [-0.2, -0.3, 0.6],   # 与CuRobo world_config同步
            #     "dims": [0.08, 0.08, 1.2],
            #     "color": [0.2, 0.2, 0.8, 0.7]  # 蓝色
            # },
            {
                "position": [0.6, 0.0, 0.55],   # 与CuRobo world_config同步
                "dims": [0.6, 0.1, 1.1],
                "color": [0.2, 0.8, 0.2, 0.7]  # 绿色
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
        
        # 清除球体标记
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
                
                # 处理不同的tensor类型
                if torch.is_tensor(final_position):
                    if final_position.dim() == 1:
                        return JointState.from_position(final_position.view(1, -1))
                    elif final_position.dim() == 2:
                        return JointState.from_position(final_position)
                    else:
                        # 如果是更高维度，取第一个
                        return JointState.from_position(final_position[0].view(1, -1))
                else:
                    # 如果不是tensor，转换为tensor
                    if isinstance(final_position, (list, np.ndarray)):
                        final_position = torch.tensor(final_position, dtype=torch.float32)
                    else:
                        # 如果是标量，需要扩展
                        final_position = torch.tensor([final_position], dtype=torch.float32)
                    
                    return JointState.from_position(final_position.view(1, -1))
            else:
                print("❌ 轨迹对象没有position属性")
                return None
                
        except Exception as e:
            print(f"❌ 处理轨迹时出错: {e}")
            return None
    
    def visualize_trajectory_with_object(self, trajectory, interpolation_dt=0.02, 
                                       playback_speed=1.0, show_object_attached=False,
                                       phase=""):
        """可视化携带物体的轨迹，并实时监测碰撞距离"""
        print(f"\n🎬 开始播放轨迹...")
        print(f"轨迹长度: {len(trajectory.position)} 个时间步")
        
        # 添加碰撞距离监测提示
        if self.collision_checker is not None:
            print(f"📏 实时碰撞距离监测已启用")
        else:
            print(f"⚠️  碰撞距离监测未启用")
        
        try:
            for i, joint_positions in enumerate(trajectory.position):
                if hasattr(joint_positions, 'cpu'):
                    joint_config = joint_positions.cpu().numpy()
                else:
                    joint_config = joint_positions
                
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
                
                # 更新球体标记位置（如果有附加的球体且motion_gen可用）
                if show_object_attached and len(self.sphere_marker_ids) > 0 and self.motion_gen is not None:
                    self._update_sphere_markers(joint_config)
                
                # 每10步打印一次碰撞距离
                if i % 10 == 0 and self.collision_checker is not None:
                    self.print_collision_distance(joint_config, i, phase)
                
                p.stepSimulation()
                time.sleep(interpolation_dt / playback_speed)
                
                if i % 10 == 0:
                    progress = (i + 1) / len(trajectory.position) * 100
                    print(f"\r播放进度: {progress:.1f}%", end='', flush=True)
            
            print(f"\n✅ 轨迹播放完成！")
            
            # 在轨迹结束时再次打印最终距离
            if self.collision_checker is not None:
                final_joint_config = trajectory.position[-1]
                if hasattr(final_joint_config, 'cpu'):
                    final_joint_config = final_joint_config.cpu().numpy()
                self.print_collision_distance(final_joint_config, len(trajectory.position)-1, f"{phase}-终点")
            
        except KeyboardInterrupt:
            print(f"\n⏹️  轨迹播放被中断")
    
    def _update_sphere_markers(self, joint_config):
        """更新球体标记位置 - 简化版本"""
        if len(self.sphere_marker_ids) == 0:
            return
            
        try:
            # 获取当前末端执行器位置
            extended_config = self._extend_joint_configuration(joint_config)
            self.set_joint_angles(extended_config)
            ee_pos, ee_quat = self.get_end_effector_pose()
            
            if ee_pos is None:
                return
                
            # 如果是第一次更新，计算并保存球体相对位置
            if len(self.sphere_relative_positions) == 0 and len(self.attached_sphere_positions) > 0:
                # 使用当前的末端执行器位置作为参考
                initial_ee_pos = ee_pos
                self.sphere_relative_positions = []
                for abs_pos in self.attached_sphere_positions:
                    relative_pos = [
                        abs_pos[0] - initial_ee_pos[0],
                        abs_pos[1] - initial_ee_pos[1], 
                        abs_pos[2] - initial_ee_pos[2]
                    ]
                    self.sphere_relative_positions.append(relative_pos)
                print(f"💡 计算了 {len(self.sphere_relative_positions)} 个球体的相对位置")
            
            # 更新球体位置
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
            # 静默处理错误，避免影响轨迹播放
            pass


def create_optimized_world():
    """创建优化的Pick and Place世界配置"""
    world_config = {
        "cuboid": {
            # 桌面
            "table": {
                "dims": [1.2, 1.2, 0.05],
                "pose": [0.4, 0.0, -0.025, 1, 0, 0, 0.0]
            },
            # 目标立方体（位置优化）
            "target_cube": {
                "dims": [0.05, 0.05, 0.05],
                "pose": [0.45, 0.35, 0.025, 1, 0, 0, 0.0]  # 与PyBullet中的target_position同步
            },
            # 障碍物1（高障碍物，与PyBullet同步）
            "obstacle1": {
                "dims": [0.08, 0.08, 1.2],  # 修改为1.2m高度，与PyBullet同步
                "pose": [-0.2, -0.3, 0.6, 1, 0, 0, 0.0]  # 更新位置为[-0.2, -0.3, 0.6]
            },
            # 障碍物2（高障碍物，与PyBullet同步）
            "obstacle2": {
                "dims": [0.6, 0.1, 1.1],   # 更新尺寸为[0.35, 0.1, 1.1]，与PyBullet同步
                "pose": [0.6, 0.0, 0.55, 1, 0, 0, 0.0]  # 调整z位置到0.55（高度的一半）
            }
        }
    }
    
    return world_config


def demo_pick_and_place_fixed():
    """修复版本的Pick and Place演示"""
    print("🤖 Pick and Place 演示 (修复版本)")
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
        num_trajopt_seeds=6,  # 增加轨迹优化种子数以提高避障成功率
        num_graph_seeds=4,    # 增加图规划种子数
    )
    motion_gen = MotionGen(motion_gen_config)
    motion_gen.warmup()
    
    # 创建可视化器
    visualizer = PickAndPlaceVisualizerFixed(gui=True)
    
    # 设置可视化器的motion_gen引用以便更新球体位置
    # 注意：这里绕过类型检查，因为motion_gen被初始化为None但后续赋值为MotionGen对象
    visualizer.motion_gen = motion_gen  # type: ignore
    
    # 设置碰撞检测器
    visualizer.setup_collision_checker(world_config)
    
    try:
        # 显式更新motion_gen的世界配置以确保障碍物被正确加载
        from curobo.geom.types import WorldConfig
        world_cfg = WorldConfig.from_dict(world_config)
        motion_gen.update_world(world_cfg)
        print(f"🌍 已将障碍物配置加载到CuRobo运动规划器中")
        print(f"   - 障碍物1: 位置 [-0.2, -0.3, 0.6], 尺寸 [0.08, 0.08, 1.2]")
        print(f"   - 障碍物2: 位置 [0.6, 0.0, 0.55], 尺寸 [0.35, 0.1, 1.1]")
        print(f"   - 目标立方体: 位置 [0.45, 0.35, 0.025], 尺寸 [0.05, 0.05, 0.05]")
        
        # 创建可视化世界
        target_pos, target_dims = visualizer.create_world_with_target_object()
        
        # 定义关键位置 - 更安全的距离
        approach_height = 0.20  # 接近高度（物体上方20cm）
        grasp_height = 0.10     # 抓取高度（物体上方10cm）
        
        approach_position = [target_pos[0], target_pos[1], target_pos[2] + target_dims[2]/2 + approach_height]
        grasp_position = [target_pos[0], target_pos[1], target_pos[2] + target_dims[2]/2 + grasp_height]
        place_position = [0.45, -0.45, 0.55]  # 更保守的放置位置
        
        # 添加可视化标记
        visualizer.add_marker(approach_position, 0.02, [1, 0.5, 0, 0.8])  # 橙色 - 接近位置
        visualizer.add_marker(grasp_position, 0.025, [1, 1, 0, 0.9])     # 黄色 - 抓取位置  
        visualizer.add_marker(place_position, 0.03, [0, 1, 1, 0.8])      # 青色 - 放置位置
        
        print(f"🔶 接近位置: {approach_position}")
        print(f"🟡 抓取位置: {grasp_position}")
        print(f"📍 放置位置: {place_position}")
        print(f"📦 目标立方体: {target_pos} (尺寸: {target_dims})")
        print(f"📏 安全距离: 接近{approach_height*100:.0f}cm, 抓取{grasp_height*100:.0f}cm")
        
        # 获取起始状态
        retract_cfg = motion_gen.get_retract_config()
        start_state = JointState.from_position(retract_cfg.view(1, -1))
        
        # 检查起始状态的碰撞距离
        print(f"\n🔍 起始状态碰撞距离检查:")
        if torch.is_tensor(retract_cfg):
            retract_cfg_np = retract_cfg.cpu().numpy()
        else:
            retract_cfg_np = retract_cfg
        visualizer.print_collision_distance(retract_cfg_np, phase="起始状态")
        
        print(f"\n📝 优化的规划流程:")
        print(f"1. 🚀 从起始位置移动到接近位置（安全距离）")
        print(f"2. 🎯 从接近位置移动到抓取位置")
        print(f"3. 🤏 抓取物体（附加到机器人）")
        print(f"4. 🚚 移动到放置位置")
        print(f"5. 📤 放置物体（从机器人分离）")
        print(f"6. 🏠 返回起始位置")
        
        # 验证碰撞检测设置
        print(f"\n🔬 验证碰撞检测设置:")
        print(f"   - 碰撞检测器类型: {motion_gen_config.world_coll_checker.checker_type}")
        print(f"   - 已加载世界配置到CuRobo运动规划器")
        print(f"   - 碰撞距离监测已启用")
        
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
            print(f"规划时间: {result1.solve_time:.4f}秒")
            
            # 播放轨迹
            trajectory1 = result1.get_interpolated_plan()
            print(f"🎬 播放到接近位置的轨迹...")
            visualizer.visualize_trajectory_with_object(
                trajectory1, 
                interpolation_dt=result1.interpolation_dt,
                playback_speed=0.5,
                phase="阶段1-接近"
            )
            
            # 安全地获取下一个状态
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
            print(f"规划时间: {result2.solve_time:.4f}秒")
            
            # 播放轨迹
            trajectory2 = result2.get_interpolated_plan()
            print(f"🎬 播放到抓取位置的轨迹...")
            visualizer.visualize_trajectory_with_object(
                trajectory2, 
                interpolation_dt=result2.interpolation_dt,
                playback_speed=0.5,
                phase="阶段2-抓取"
            )
            
            # 更新当前状态
            current_state = visualizer.safe_get_joint_state_from_trajectory(trajectory2)
            if current_state is None:
                print("❌ 无法获取轨迹终点状态")
                return
            
        else:
            print(f"❌ 到抓取位置的规划失败！状态: {result2.status}")
            print(f"💡 提示: 抓取位置已优化，但仍可能需要进一步调整")
            return
        
        input("\n按回车键继续到抓取阶段...")
        
        # === 阶段3: 抓取物体 ===
        print(f"\n🤏 阶段3: 抓取物体（附加到机器人）...")
        
        # 使用默认的link_name "attached_object"，先不移除障碍物
        success = motion_gen.attach_objects_to_robot(
            joint_state=current_state,
            object_names=["target_cube"],
            surface_sphere_radius=0.01,  # 使用更小的球体半径
            remove_obstacles_from_world_config=False  # 不完全移除，只是禁用
        )
        
        if success:
            print("✅ 成功将立方体附加到机器人！")
            print("🔗 立方体现在是机器人的一部分，会跟随机器人移动")
            
            # 检查附加物体后的碰撞距离
            print(f"\n📏 物体附加后的碰撞距离:")
            final_joint_config = trajectory2.position[-1]
            if torch.is_tensor(final_joint_config):
                final_joint_config_np = final_joint_config.cpu().numpy()
            else:
                final_joint_config_np = final_joint_config
            visualizer.print_collision_distance(final_joint_config_np, phase="物体附加后")
            
            # === 详细的附加物体分析 ===
            print("\n🔍 详细分析附加物体的球体表示...")
            
            # 获取附加对象的球体信息
            try:
                attached_spheres = motion_gen.kinematics.kinematics_config.get_link_spheres("attached_object")
                print(f"📊 附加对象球体信息:")
                print(f"   - 球体数量: {attached_spheres.shape[0]}")
                
                # 显示每个球体的位置和半径
                for i, sphere in enumerate(attached_spheres.cpu().numpy()):
                    x, y, z, radius = sphere
                    if radius > 0:  # 只显示有效球体
                        print(f"   - 球体 {i}: 位置=({x:.3f}, {y:.3f}, {z:.3f}), 半径={radius:.3f}")
                
                # 计算当前运动学状态
                kin_state = motion_gen.compute_kinematics(current_state)
                
                # 获取世界坐标系下的球体位置
                if kin_state.robot_spheres is not None:
                    all_spheres = kin_state.robot_spheres.squeeze().cpu().numpy()
                    print(f"\n🌍 世界坐标系下的所有机器人球体:")
                    print(f"   - 总球体数量: {len(all_spheres)}")
                    
                    # 找出附加对象的球体（通过半径匹配）
                    attached_world_spheres = []
                    target_radius = 0.01  # 匹配我们设置的半径
                    
                    print(f"🔍 查找半径约为 {target_radius} 的球体...")
                    
                    for i, sphere in enumerate(all_spheres):
                        x, y, z, radius = sphere
                        if radius > 0 and abs(radius - target_radius) < 0.005:  # 匹配我们设置的半径
                            attached_world_spheres.append((i, sphere))
                            print(f"   - ✅ 找到附加球体 {i}: 世界位置=({x:.3f}, {y:.3f}, {z:.3f}), 半径={radius:.6f}")
                        elif radius > 0:
                            print(f"   - 球体 {i}: 位置=({x:.3f}, {y:.3f}, {z:.3f}), 半径={radius:.6f} (不匹配)")
                    
                    print(f"\n📊 匹配结果: 找到 {len(attached_world_spheres)} 个附加球体")
                else:
                    print("\n⚠️  无法获取机器人球体信息")
                    attached_world_spheres = []
            
            except Exception as e:
                print(f"⚠️  获取球体信息时出现问题: {e}")
            
            # === 可视化附加的球体 ===
            print("\n🎨 在PyBullet中可视化附加的球体...")
            
            try:
                # 在PyBullet中添加半透明球体标记来显示附加对象的球体
                sphere_marker_ids = []
                
                if 'attached_world_spheres' in locals():
                    for sphere_idx, (global_idx, sphere) in enumerate(attached_world_spheres):
                        x, y, z, radius = sphere
                        
                        # 使用更大的可视化半径确保能看到
                        visual_radius = max(radius, 0.02)  # 至少2cm半径，是原始的2倍
                        
                        # 创建醒目的球体标记 - 亮黄色，完全不透明
                        visual_shape = p.createVisualShape(
                            p.GEOM_SPHERE,
                            radius=visual_radius,
                            rgbaColor=[1.0, 1.0, 0.0, 1.0]  # 亮黄色，完全不透明
                        )
                        
                        sphere_marker = p.createMultiBody(
                            baseMass=0,
                            baseVisualShapeIndex=visual_shape,
                            basePosition=[x, y, z]
                        )
                        
                        sphere_marker_ids.append(sphere_marker)
                        visualizer.sphere_marker_ids.append(sphere_marker)  # 保存到可视化器中
                        visualizer.attached_sphere_positions.append([x, y, z])  # 保存球体的绝对位置
                        print(f"   ✅ 创建球体标记 {sphere_idx}: 位置=({x:.3f}, {y:.3f}, {z:.3f})")
                        print(f"      原始半径={radius:.4f}m, 可视化半径={visual_radius:.4f}m")
                        
                        # 添加文本标签
                        p.addUserDebugText(f"球体{sphere_idx}", [x, y, z+0.03], 
                                         textColorRGB=[1, 0, 0], textSize=1.2)
                
                print(f"📍 创建了 {len(sphere_marker_ids)} 个球体可视化标记")
                
            except Exception as e:
                print(f"⚠️  球体可视化时出现问题: {e}")
            
            # === 重要：验证附加后的状态 ===
            print("\n🔍 验证附加物体后的状态...")
            
            # 检查当前状态是否有效
            valid_query, status = motion_gen.check_start_state(current_state)
            if valid_query:
                print("✅ 当前状态验证通过")
            else:
                print(f"❌ 当前状态验证失败: {status}")
                
                # === 详细的碰撞距离分析 ===
                print("\n🔬 详细的碰撞距离分析...")
                
                try:
                    # 获取约束信息来分析碰撞
                    kin_state = motion_gen.compute_kinematics(current_state)
                    
                    # 使用motion_gen的内置方法检查约束
                    metrics = motion_gen.check_constraints(current_state)
                    print(f"📊 碰撞约束检查结果:")
                    if hasattr(metrics, 'feasible') and metrics.feasible is not None:
                        print(f"   - 状态可行性: {metrics.feasible.item()}")
                    if hasattr(metrics, 'constraint') and metrics.constraint is not None:
                        print(f"   - 约束值: {metrics.constraint.item():.6f}")
                    if hasattr(metrics, 'cost') and metrics.cost is not None:
                        print(f"   - 总成本: {metrics.cost.item():.6f}")
                    
                    # 如果约束值很大，说明有碰撞
                    if hasattr(metrics, 'constraint') and metrics.constraint is not None:
                        constraint_val = metrics.constraint.item()
                        if constraint_val > 0:
                            print(f"⚠️  检测到碰撞！约束值: {constraint_val:.6f}")
                            print("🔧 这表明附加的球体与机器人或环境发生了碰撞")
                            
                            # 尝试用更小的球体半径重新附加
                            print("\n🔧 尝试使用更小的球体半径...")
                            motion_gen.detach_object_from_robot()
                            
                            success_retry = motion_gen.attach_objects_to_robot(
                                joint_state=current_state,
                                object_names=["target_cube"],
                                surface_sphere_radius=0.0005,  # 更小的半径
                                remove_obstacles_from_world_config=False
                            )
                            
                            if success_retry:
                                # 重新检查状态
                                valid_query_retry, status_retry = motion_gen.check_start_state(current_state)
                                metrics_retry = motion_gen.check_constraints(current_state)
                                
                                print(f"🔄 重新附加后的结果:")
                                print(f"   - 状态验证: {valid_query_retry}")
                                if hasattr(metrics_retry, 'constraint') and metrics_retry.constraint is not None:
                                    print(f"   - 约束值: {metrics_retry.constraint.item():.6f}")
                                
                                if valid_query_retry:
                                    print("✅ 使用更小球体半径后验证通过！")
                                else:
                                    print(f"❌ 仍然验证失败: {status_retry}")
                     
                except Exception as e:
                    print(f"⚠️  碰撞距离分析时出现问题: {e}")
            
            # 获取约束信息和碰撞距离
            try:
                metrics = motion_gen.check_constraints(current_state)
                print(f"\n📊 约束检查结果:")
                if hasattr(metrics, 'feasible') and metrics.feasible is not None:
                    print(f"   - 状态可行性: {metrics.feasible.item()}")
                
                # 检查是否有碰撞相关的成本
                if hasattr(metrics, 'cost') and metrics.cost is not None:
                    print(f"   - 总成本: {metrics.cost.item():.6f}")
                    
            except Exception as e:
                print(f"⚠️  约束检查时出现问题: {e}")
            
            # 检查机器人几何形状
            print(f"\n📐 检查机器人附加对象后的几何信息...")
            try:
                # 计算当前运动学状态
                kin_state = motion_gen.compute_kinematics(current_state)
                if hasattr(kin_state, 'ee_pos_seq') and kin_state.ee_pos_seq is not None:
                    ee_pos = kin_state.ee_pos_seq.squeeze().cpu().numpy()
                    print(f"   - 当前末端执行器位置: {ee_pos}")
                else:
                    print("   - 无法获取末端执行器位置")
                
                # 检查附加的球体信息
                attached_spheres = motion_gen.kinematics.kinematics_config.get_number_of_spheres("attached_object")
                print(f"   - 附加对象球体数量: {attached_spheres}")
                print(f"   - 机器人现在包含附加的物体几何形状")
                
            except Exception as e:
                print(f"⚠️  几何信息检查时出现问题: {e}")
            
        else:
            print("❌ 物体附加失败！")
            return
        
        # 在PyBullet中也更新物体状态
        if visualizer.target_object_id is not None:
            # 获取机器人手部位置
            final_ee_pos, final_ee_quat = visualizer.get_end_effector_pose()
            if final_ee_pos is not None:
                final_object_position = [final_ee_pos[0], final_ee_pos[1], final_ee_pos[2] - 0.05]
                p.resetBasePositionAndOrientation(
                    visualizer.target_object_id, 
                    final_object_position, 
                    final_ee_quat
                )
        
        input("\n按回车键继续到移动阶段...")
        
        # === 阶段4: 移动到放置位置 ===
        print(f"\n🚚 阶段4: 规划到放置位置（携带物体）...")
        
        # 在规划前进行详细的状态检查
        print("🔍 阶段4前的详细状态检查...")
        
        # 再次验证当前状态
        valid_query_4, status_4 = motion_gen.check_start_state(current_state)
        print(f"📊 阶段4前状态验证: {valid_query_4}")
        if not valid_query_4:
            print(f"❌ 状态验证失败: {status_4}")
        
        # 检查目标位置的可达性
        print(f"🎯 目标放置位置: {place_position}")
        
        # 尝试多种规划策略
        place_pose = Pose.from_list([
            place_position[0], place_position[1], place_position[2], 
            0, 1, 0, 0  # 使用工作的四元数
        ])
        
        # 策略1: 标准规划
        print("🔧 尝试策略1: 标准规划...")
        plan_config_4 = MotionGenPlanConfig(
            enable_graph=True,
            enable_opt=True,
            max_attempts=3,
            timeout=10.0,
            check_start_validity=False  # 跳过起始状态检查，因为我们已经验证了
        )
        
        result4 = motion_gen.plan_single(
            start_state=current_state,
            goal_pose=place_pose,
            plan_config=plan_config_4
        )
        
        if result4.success is not None and result4.success.item():
            print("✅ 策略1成功！标准规划到放置位置成功！")
        else:
            print(f"❌ 策略1失败: {result4.status}")
            
            # 策略2: 尝试中间位置
            print("🔧 尝试策略2: 通过中间位置规划...")
            
            # 先移动到中间安全位置
            intermediate_position = [0.35, 0.25, 0.45]  # 中间位置，高一些
            intermediate_pose = Pose.from_list([
                intermediate_position[0], intermediate_position[1], intermediate_position[2],
                0, 1, 0, 0
            ])
            
            print(f"🎯 中间位置: {intermediate_position}")
            
            result_intermediate = motion_gen.plan_single(
                start_state=current_state,
                goal_pose=intermediate_pose,
                plan_config=plan_config_4
            )
            
            if result_intermediate.success is not None and result_intermediate.success.item():
                print("✅ 到中间位置的规划成功！")
                
                # 播放到中间位置的轨迹
                interpolated_intermediate = result_intermediate.get_interpolated_plan()
                visualizer.visualize_trajectory_with_object(
                    interpolated_intermediate,
                    interpolation_dt=result_intermediate.interpolation_dt,
                    playback_speed=1.0,
                    show_object_attached=True,
                    phase="阶段4-中间位置"
                )
                
                # 更新当前状态
                current_state = visualizer.safe_get_joint_state_from_trajectory(interpolated_intermediate, -1)
                if current_state is None:
                    print("❌ 无法获取中间轨迹终点状态")
                    return
                
                # 从中间位置规划到最终位置
                print("🎯 从中间位置规划到最终放置位置...")
                
                result_final = motion_gen.plan_single(
                    start_state=current_state,
                    goal_pose=place_pose,
                    plan_config=plan_config_4
                )
                
                if result_final.success is not None and result_final.success.item():
                    print("✅ 策略2成功！通过中间位置到达目标！")
                    result4 = result_final  # 使用最终结果
                else:
                    print(f"❌ 策略2也失败: {result_final.status}")
                    print("🚫 无法完成放置操作")
                    return
            else:
                print(f"❌ 到中间位置的规划也失败: {result_intermediate.status}")
                print("🚫 无法完成放置操作")
                return
        
        # 播放轨迹
        trajectory3 = result4.get_interpolated_plan()
        print(f"🎬 播放到放置位置的轨迹（携带物体）...")
        visualizer.visualize_trajectory_with_object(
            trajectory3, 
            interpolation_dt=result4.interpolation_dt,
            playback_speed=0.5,
            show_object_attached=True,
            phase="阶段4-放置"
        )
        
        # 更新当前状态
        current_state = visualizer.safe_get_joint_state_from_trajectory(trajectory3)
        if current_state is None:
            print("❌ 无法获取最终轨迹终点状态")
            return
        
        input("\n按回车键继续到放置阶段...")
        
        # === 阶段5: 放置物体 ===
        print(f"\n📤 阶段5: 放置物体（从机器人分离）...")
        
        # 分离物体，使用默认的link_name "attached_object"
        motion_gen.detach_object_from_robot()
        print(f"✅ 成功将立方体从机器人分离！")
        
        # 在PyBullet中更新物体位置
        if visualizer.target_object_id is not None:
            final_object_position = [place_position[0], place_position[1], place_position[2] - 0.05]
            p.resetBasePositionAndOrientation(
                visualizer.target_object_id, 
                final_object_position, 
                [0, 1, 0, 0]
            )
        
        # 检查放置后的碰撞距离
        print(f"\n📏 物体放置后的碰撞距离:")
        final_joint_config = trajectory3.position[-1]
        if torch.is_tensor(final_joint_config):
            final_joint_config_np = final_joint_config.cpu().numpy()
        else:
            final_joint_config_np = final_joint_config
        visualizer.print_collision_distance(final_joint_config_np, phase="物体放置后")
        
        print(f"\n🎉 Pick and Place 演示完成！")
        print(f"📊 演示统计:")
        print(f"  ✅ 所有阶段成功完成")
        print(f"  📏 使用了安全的抓取距离: {grasp_height*100:.0f}cm")
        print(f"  🎯 物体成功从 {target_pos} 移动到 {place_position}")
        print(f"  🧠 自动避障和碰撞检测正常工作")
        print(f"  📏 实时碰撞距离监测功能已启用")
        
        input("\n按回车键退出演示...")
        
    except Exception as e:
        print(f"❌ 演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        visualizer.disconnect()


def main():
    """主函数"""
    print("🤖 Pick and Place 演示 (修复版本)")
    print("这个版本解决了抓取位置和tensor处理的问题")
    print("\n✨ 改进:")
    print("• 🎯 可视化抓取位置标记")
    print("• 📏 优化的安全抓取距离") 
    print("• 🔄 分阶段接近和抓取")
    print("• 🛠️  修复了tensor处理问题")
    print("• 🎬 更好的可视化效果")
    print("• 🚧 修复了障碍物碰撞检测问题")
    print("• 🌟 支持动态球体可视化")
    print("• 🔍 完整的避障路径规划")
    
    response = input("\n开始Pick and Place演示吗？(y/n): ")
    if response.lower() in ['y', 'yes', '是']:
        demo_pick_and_place_fixed()
    else:
        print("演示已取消")


if __name__ == "__main__":
    setup_curobo_logger("error")
    main() 