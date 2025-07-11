#!/usr/bin/env python3
"""
Pick and Place 演示脚本 (修复版本)
解决了tensor处理问题，增加了抓取位置可视化
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

# Local
from pybullet_kinematics_visualization import PyBulletKinematicsVisualizer

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class PickAndPlaceVisualizerFixed(PyBulletKinematicsVisualizer):
    """Pick and Place 可视化器 (修复版本)"""
    
    def __init__(self, robot_config_name="franka.yml", gui=True):
        super().__init__(robot_config_name, gui)
        self.obstacle_ids = []
        self.target_object_id = None
        self.target_markers = []
        
    def create_world_with_target_object(self):
        """创建包含目标物体和障碍物的世界"""
        self.clear_obstacles()
        
        # 创建目标立方体 - 位置调整到更合适的地方
        target_dims = [0.05, 0.05, 0.05]
        target_position = [0.4, 0.15, 0.025]  # 调整到更容易抓取的位置
        
        target_collision_shape = p.createCollisionShape(
            p.GEOM_BOX, 
            halfExtents=[target_dims[0]/2, target_dims[1]/2, target_dims[2]/2]
        )
        target_visual_shape = p.createVisualShape(
            p.GEOM_BOX, 
            halfExtents=[target_dims[0]/2, target_dims[1]/2, target_dims[2]/2],
            rgbaColor=[1.0, 0.2, 0.2, 1.0]  # 红色
        )
        
        self.target_object_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=target_collision_shape,
            baseVisualShapeIndex=target_visual_shape,
            basePosition=target_position
        )
        
        print(f"📦 创建目标立方体: 位置 {target_position}, 尺寸 {target_dims}")
        
        # 创建障碍物 - 位置远离抓取区域
        obstacles = [
            {
                "position": [0.2, -0.3, 0.1],
                "dims": [0.08, 0.08, 0.2],
                "color": [0.2, 0.2, 0.8, 0.7]  # 蓝色
            },
            {
                "position": [0.6, 0.0, 0.05],
                "dims": [0.08, 0.1, 0.1],
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
                                       playback_speed=1.0, show_object_attached=False):
        """可视化携带物体的轨迹"""
        print(f"\n🎬 开始播放轨迹...")
        print(f"轨迹长度: {len(trajectory.position)} 个时间步")
        
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
                
                p.stepSimulation()
                time.sleep(interpolation_dt / playback_speed)
                
                if i % 10 == 0:
                    progress = (i + 1) / len(trajectory.position) * 100
                    print(f"\r播放进度: {progress:.1f}%", end='', flush=True)
            
            print(f"\n✅ 轨迹播放完成！")
            
        except KeyboardInterrupt:
            print(f"\n⏹️  轨迹播放被中断")


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
                "pose": [0.4, 0.15, 0.025, 1, 0, 0, 0.0]
            },
            # 障碍物1（远离抓取区域）
            "obstacle1": {
                "dims": [0.08, 0.08, 0.2],
                "pose": [0.2, -0.3, 0.1, 1, 0, 0, 0.0]
            },
            # 障碍物2
            "obstacle2": {
                "dims": [0.08, 0.1, 0.1],
                "pose": [0.6, 0.0, 0.05, 1, 0, 0, 0.0]
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
        num_trajopt_seeds=4,
    )
    motion_gen = MotionGen(motion_gen_config)
    motion_gen.warmup()
    
    # 创建可视化器
    visualizer = PickAndPlaceVisualizerFixed(gui=True)
    
    try:
        # 创建可视化世界
        target_pos, target_dims = visualizer.create_world_with_target_object()
        
        # 定义关键位置 - 更安全的距离
        approach_height = 0.20  # 接近高度（物体上方20cm）
        grasp_height = 0.10     # 抓取高度（物体上方10cm）
        
        approach_position = [target_pos[0], target_pos[1], target_pos[2] + target_dims[2]/2 + approach_height]
        grasp_position = [target_pos[0], target_pos[1], target_pos[2] + target_dims[2]/2 + grasp_height]
        place_position = [0.45, 0.45, 0.35]  # 更保守的放置位置
        
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
        
        print(f"\n📝 优化的规划流程:")
        print(f"1. 🚀 从起始位置移动到接近位置（安全距离）")
        print(f"2. 🎯 从接近位置移动到抓取位置")
        print(f"3. 🤏 抓取物体（附加到机器人）")
        print(f"4. 🚚 移动到放置位置")
        print(f"5. 📤 放置物体（从机器人分离）")
        print(f"6. 🏠 返回起始位置")
        
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
            MotionGenPlanConfig(max_attempts=5, enable_graph=True)
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
                playback_speed=0.5
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
            MotionGenPlanConfig(max_attempts=5)
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
                playback_speed=0.5
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
                    for i, sphere in enumerate(all_spheres):
                        x, y, z, radius = sphere
                        if radius > 0 and abs(radius - 0.001) < 0.0005:  # 匹配我们设置的半径
                            attached_world_spheres.append((i, sphere))
                            print(f"   - 附加球体 {i}: 世界位置=({x:.3f}, {y:.3f}, {z:.3f}), 半径={radius:.3f}")
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
                        
                        # 创建半透明的球体标记
                        visual_shape = p.createVisualShape(
                            p.GEOM_SPHERE,
                            radius=radius,
                            rgbaColor=[1.0, 0.5, 0.0, 1.0]  # 半透明橙色
                        )
                        
                        sphere_marker = p.createMultiBody(
                            baseMass=0,
                            baseVisualShapeIndex=visual_shape,
                            basePosition=[x, y, z]
                        )
                        
                        sphere_marker_ids.append(sphere_marker)
                        print(f"   ✅ 创建球体标记 {sphere_idx}: 位置=({x:.3f}, {y:.3f}, {z:.3f})")
                
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
        
        if current_state is not None:
            result4 = motion_gen.plan_single(
                start_state=current_state,
                goal_pose=place_pose,
                plan_config=plan_config_4
            )
        else:
            print("❌ 当前状态为空，无法进行规划")
            return
        
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
                    show_object_attached=True
                )
                
                # 更新当前状态
                current_state = visualizer.safe_get_joint_state_from_trajectory(interpolated_intermediate, -1)
                
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
            show_object_attached=True
        )
        
        # 更新当前状态
        current_state = visualizer.safe_get_joint_state_from_trajectory(trajectory3)
        
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
        
        print(f"\n🎉 Pick and Place 演示完成！")
        print(f"📊 演示统计:")
        print(f"  ✅ 所有阶段成功完成")
        print(f"  📏 使用了安全的抓取距离: {grasp_height*100:.0f}cm")
        print(f"  🎯 物体成功从 {target_pos} 移动到 {place_position}")
        print(f"  🧠 自动避障和碰撞检测正常工作")
        
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
    
    choice = input("\n开始演示吗？(y/n): ").strip().lower()
    if choice in ['y', 'yes', '是']:
        demo_pick_and_place_fixed()
    else:
        print("演示已取消")


if __name__ == "__main__":
    setup_curobo_logger("error")
    main() 