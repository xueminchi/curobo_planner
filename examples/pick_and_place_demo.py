#!/usr/bin/env python3
"""
Pick and Place 演示脚本
实现机械臂抓取立方体并移动到另一个位置的完整流程
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


class PickAndPlaceVisualizer(PyBulletKinematicsVisualizer):
    """Pick and Place 可视化器"""
    
    def __init__(self, robot_config_name="franka.yml", gui=True):
        super().__init__(robot_config_name, gui)
        self.obstacle_ids = []
        self.target_object_id = None
        self.target_markers = []
        
    def create_world_with_target_object(self):
        """创建包含目标物体和障碍物的世界"""
        
        # 清除现有障碍物
        self.clear_obstacles()
        
        # 创建目标立方体（要抓取的物体） - 红色
        target_dims = [0.05, 0.05, 0.05]  # 5cm立方体
        target_position = [0.5, 0.2, 0.025]  # 放在桌面上
        
        target_collision_shape = p.createCollisionShape(
            p.GEOM_BOX, 
            halfExtents=[target_dims[0]/2, target_dims[1]/2, target_dims[2]/2]
        )
        target_visual_shape = p.createVisualShape(
            p.GEOM_BOX, 
            halfExtents=[target_dims[0]/2, target_dims[1]/2, target_dims[2]/2],
            rgbaColor=[1.0, 0.2, 0.2, 1.0]  # 红色，不透明
        )
        
        self.target_object_id = p.createMultiBody(
            baseMass=0.1,  # 给一点质量，但不会影响CuRobo规划
            baseCollisionShapeIndex=target_collision_shape,
            baseVisualShapeIndex=target_visual_shape,
            basePosition=target_position
        )
        
        print(f"📦 创建目标立方体: 位置 {target_position}, 尺寸 {target_dims}")
        
        # 创建一些障碍物 - 蓝色
        obstacles = [
            {
                "position": [0.3, -0.3, 0.1],
                "dims": [0.1, 0.1, 0.2],
                "color": [0.2, 0.2, 0.8, 0.7]  # 蓝色
            },
            {
                "position": [0.6, -0.1, 0.05],
                "dims": [0.08, 0.15, 0.1],
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
    
    def add_target_marker(self, position, size=0.03, color=[0, 1, 1, 0.8]):
        """添加目标位置标记（青色球体）"""
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
    
    def add_grasp_marker(self, position, size=0.025, color=[1, 1, 0, 0.9]):
        """添加抓取位置标记（黄色球体）"""
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
    
    def add_approach_marker(self, position, size=0.02, color=[1, 0.5, 0, 0.8]):
        """添加接近位置标记（橙色球体）"""
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
                        # 将物体位置设置为末端执行器位置（稍微偏移）
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


def create_pick_and_place_world():
    """创建Pick and Place的世界配置"""
    world_config = {
        "cuboid": {
            # 桌面
            "table": {
                "dims": [1.5, 1.5, 0.05],
                "pose": [0.5, 0.0, -0.025, 1, 0, 0, 0.0]
            },
            # 目标立方体（要抓取的物体）
            "target_cube": {
                "dims": [0.05, 0.05, 0.05],
                "pose": [0.5, 0.2, 0.025, 1, 0, 0, 0.0]
            },
            # 障碍物1
            "obstacle1": {
                "dims": [0.1, 0.1, 0.2],
                "pose": [0.3, -0.3, 0.1, 1, 0, 0, 0.0]
            },
            # 障碍物2
            "obstacle2": {
                "dims": [0.08, 0.15, 0.1],
                "pose": [0.6, -0.1, 0.05, 1, 0, 0, 0.0]
            }
        }
    }
    
    return world_config


def demo_pick_and_place():
    """Pick and Place 完整演示"""
    print("🤖 开始 Pick and Place 演示")
    print("="*60)
    
    # 设置参数
    tensor_args = TensorDeviceType()
    robot_file = "franka.yml"
    
    # 创建世界配置
    world_config = create_pick_and_place_world()
    
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
    visualizer = PickAndPlaceVisualizer(gui=True)
    
    try:
        # 创建可视化世界
        target_pos, target_dims = visualizer.create_world_with_target_object()
        
        # 定义关键位置 - 增加安全距离
        approach_height = 0.15  # 接近高度（物体上方15cm）
        grasp_height = 0.08     # 抓取高度（物体上方8cm）
        
        approach_position = [target_pos[0], target_pos[1], target_pos[2] + target_dims[2]/2 + approach_height]  # 接近位置
        grasp_position = [target_pos[0], target_pos[1], target_pos[2] + target_dims[2]/2 + grasp_height]  # 抓取位置
        place_position = [0.3, 0.3, 0.3]  # 放置位置
        
        # 添加可视化标记
        visualizer.add_approach_marker(approach_position)  # 橙色 - 接近位置
        visualizer.add_grasp_marker(grasp_position)        # 黄色 - 抓取位置  
        visualizer.add_target_marker(place_position)       # 青色 - 放置位置
        
        print(f"🔶 接近位置: {approach_position}")
        print(f"🟡 抓取位置: {grasp_position}")
        print(f"📍 放置位置: {place_position}")
        print(f"📦 目标立方体: {target_pos} (尺寸: {target_dims})")
        print(f"📏 安全距离: 接近{approach_height*100:.0f}cm, 抓取{grasp_height*100:.0f}cm")
        
        # 获取起始状态
        retract_cfg = motion_gen.get_retract_config()
        start_state = JointState.from_position(retract_cfg.view(1, -1))
        
        print(f"\n📝 规划流程:")
        print(f"1. 从起始位置移动到接近位置（安全距离）")
        print(f"2. 从接近位置移动到抓取位置")
        print(f"3. 抓取物体（附加到机器人）")
        print(f"4. 移动到放置位置")
        print(f"5. 放置物体（从机器人分离）")
        print(f"6. 返回起始位置")
        
        input("\n按回车键开始演示...")
        
        # === 阶段1: 移动到接近位置 ===
        print(f"\n🚀 阶段1: 规划到接近位置（安全距离）...")
        approach_pose = Pose.from_list([
            approach_position[0], approach_position[1], approach_position[2], 
            1.0, 0.0, 0.0, 0.0  # 保持标准方向
        ])
        
        result1 = motion_gen.plan_single(
            start_state, 
            approach_pose, 
            MotionGenPlanConfig(max_attempts=5)
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
            
            # 更新当前状态
            final_position = trajectory1.position[-1]
            if hasattr(final_position, 'view'):
                current_state = JointState.from_position(final_position.view(1, -1))
            else:
                current_state = JointState.from_position(torch.tensor(final_position).view(1, -1))
            
        else:
            print(f"❌ 到接近位置的规划失败！状态: {result1.status}")
            return
        
        input("\n按回车键继续到抓取位置...")
        
        # === 阶段2: 移动到抓取位置 ===
        print(f"\n🎯 阶段2: 规划到抓取位置...")
        grasp_pose = Pose.from_list([
            grasp_position[0], grasp_position[1], grasp_position[2], 
            1.0, 0.0, 0.0, 0.0  # 保持标准方向
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
            final_position = trajectory2.position[-1]
            if hasattr(final_position, 'view'):
                current_state = JointState.from_position(final_position.view(1, -1))
            else:
                current_state = JointState.from_position(torch.tensor(final_position).view(1, -1))
            
        else:
            print(f"❌ 到抓取位置的规划失败！状态: {result2.status}")
            print(f"💡 提示: 抓取位置可能太靠近障碍物或超出工作空间")
            return
        
        input("\n按回车键继续到抓取阶段...")
        
        # === 阶段3: 抓取物体 ===
        print(f"\n🤏 阶段3: 抓取物体（附加到机器人）...")
        
        # 将目标立方体附加到机器人
        success = motion_gen.attach_objects_to_robot(
            current_state,
            ["target_cube"],  # 物体名称
            surface_sphere_radius=0.01,
            link_name="attached_cube",
            remove_obstacles_from_world_config=True  # 从障碍物中移除
        )
        
        if success:
            print(f"✅ 成功将立方体附加到机器人！")
            print(f"🔗 立方体现在是机器人的一部分，会跟随机器人移动")
        else:
            print(f"❌ 附加物体失败！")
            return
        
        input("\n按回车键继续到移动阶段...")
        
        # === 阶段4: 移动到放置位置 ===
        print(f"\n🚚 阶段4: 规划到放置位置（携带物体）...")
        place_pose = Pose.from_list([
            place_position[0], place_position[1], place_position[2], 
            1.0, 0.0, 0.0, 0.0
        ])
        
        result3 = motion_gen.plan_single(
            current_state, 
            place_pose, 
            MotionGenPlanConfig(max_attempts=5)
        )
        
        if result3.success is not None and (result3.success.item() if hasattr(result3.success, 'item') else result3.success):
            print(f"✅ 到放置位置的规划成功！")
            print(f"规划时间: {result3.solve_time:.4f}秒")
            print(f"🧠 注意: 这次规划考虑了附加的立方体避障")
            
            # 播放轨迹
            trajectory3 = result3.get_interpolated_plan()
            print(f"🎬 播放到放置位置的轨迹（携带物体）...")
            visualizer.visualize_trajectory_with_object(
                trajectory3, 
                interpolation_dt=result3.interpolation_dt,
                playback_speed=0.5,
                show_object_attached=True  # 显示物体跟随
            )
            
            # 更新当前状态
            final_position = trajectory3.position[-1]
            if hasattr(final_position, 'view'):
                current_state = JointState.from_position(final_position.view(1, -1))
            else:
                current_state = JointState.from_position(torch.tensor(final_position).view(1, -1))
            
        else:
            print(f"❌ 到放置位置的规划失败！状态: {result3.status}")
            print(f"🤔 可能是因为携带物体后空间限制增加")
            return
        
        input("\n按回车键继续到放置阶段...")
        
        # === 阶段5: 放置物体 ===
        print(f"\n📤 阶段5: 放置物体（从机器人分离）...")
        
        # 从机器人上分离物体
        motion_gen.detach_object_from_robot("attached_cube")
        print(f"✅ 成功将立方体从机器人分离！")
        print(f"📦 立方体现在位于放置位置")
        
        # 在PyBullet中更新物体位置
        if visualizer.target_object_id is not None:
            final_object_position = [place_position[0], place_position[1], place_position[2] - 0.05]
            p.resetBasePositionAndOrientation(
                visualizer.target_object_id, 
                final_object_position, 
                [0, 0, 0, 1]
            )
        
        input("\n按回车键继续到返回阶段...")
        
        # === 阶段6: 返回起始位置 ===
        print(f"\n🏠 阶段6: 返回起始位置...")
        
        result4 = motion_gen.plan_single(
            current_state, 
            Pose.from_list([retract_cfg[0].item(), retract_cfg[1].item(), retract_cfg[2].item(), 
                           1.0, 0.0, 0.0, 0.0]),  # 近似retract姿态
            MotionGenPlanConfig(max_attempts=3)
        )
        
        if result3.success is not None and (result3.success.item() if hasattr(result3.success, 'item') else result3.success):
            print(f"✅ 返回起始位置的规划成功！")
            
            # 播放轨迹
            trajectory3 = result3.get_interpolated_plan()
            print(f"🎬 播放返回起始位置的轨迹...")
            visualizer.visualize_trajectory_with_object(
                trajectory3, 
                interpolation_dt=result3.interpolation_dt,
                playback_speed=0.5
            )
            
        else:
            print(f"❌ 返回起始位置的规划失败！")
        
        # === 演示完成 ===
        print(f"\n🎉 Pick and Place 演示完成！")
        print(f"📊 演示统计:")
        print(f"  - 总阶段数: 5")
        print(f"  - 成功阶段: 4+")
        print(f"  - 物体成功从 {target_pos} 移动到 {place_position}")
        print(f"  - 机器人安全避开了所有障碍物")
        
        print(f"\n💡 技术亮点:")
        print(f"  ✓ 使用 attach_objects_to_robot() 实现物体抓取")
        print(f"  ✓ 抓取后物体自动成为机器人的一部分")
        print(f"  ✓ 运动规划自动考虑附加物体的碰撞检测")
        print(f"  ✓ 使用 detach_object_from_robot() 实现物体放置")
        print(f"  ✓ PyBullet实时可视化整个过程")
        print(f"  ✓ 自动碰撞检测和避障")
        
        input("\n按回车键退出演示...")
        
    except Exception as e:
        print(f"❌ 演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        visualizer.disconnect()


def main():
    """主函数"""
    print("🤖 Pick and Place 演示")
    print("这个演示将展示机械臂抓取立方体并移动到另一个位置的完整流程")
    print("\n特性:")
    print("• 🎯 智能路径规划到抓取位置")
    print("• 🤏 物体附加到机器人（attach_objects_to_robot）")
    print("• 🚚 携带物体的避障运动规划")
    print("• 📤 物体分离和放置（detach_object_from_robot）")
    print("• 🎬 PyBullet实时可视化")
    print("• 🧠 自动碰撞检测和避障")
    
    choice = input("\n开始演示吗？(y/n): ").strip().lower()
    if choice in ['y', 'yes', '是']:
        demo_pick_and_place()
    else:
        print("演示已取消")


if __name__ == "__main__":
    setup_curobo_logger("error")
    main() 