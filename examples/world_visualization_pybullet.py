#!/usr/bin/env python3
"""
使用PyBullet可视化CuRobo中的所有世界配置文件
"""

import os
import time
import numpy as np
import pybullet as p
import pybullet_data

# Third Party  
import torch

# CuRobo
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.util.logger import setup_curobo_logger
from curobo.util_file import get_world_configs_path, join_path, load_yaml


class WorldVisualizerPyBullet:
    """CuRobo世界配置的PyBullet可视化器"""
    
    def __init__(self, gui=True):
        """初始化PyBullet环境
        
        Args:
            gui: 是否显示GUI界面
        """
        self.gui = gui
        self.physics_client = None
        self.obstacle_ids = []
        self.ground_plane_id = None
        
        self._setup_pybullet()
        
    def _setup_pybullet(self):
        """设置PyBullet环境"""
        if self.gui:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # 设置相机参数
        p.resetDebugVisualizerCamera(
            cameraDistance=3.0,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0.5]
        )
        
        # 加载地面
        self.ground_plane_id = p.loadURDF("plane.urdf")
        
        print("PyBullet环境初始化完成")
        
    def clear_world(self):
        """清除当前世界中的所有障碍物"""
        for obstacle_id in self.obstacle_ids:
            try:
                p.removeBody(obstacle_id)
            except:
                pass  # 忽略已删除的物体
        self.obstacle_ids.clear()
        
    def _load_cuboid(self, name, cuboid_data):
        """加载立方体障碍物
        
        Args:
            name: 障碍物名称
            cuboid_data: 立方体配置数据
        """
        dims = cuboid_data.dims
        pose = cuboid_data.pose
        
        # 获取颜色信息（如果有的话）
        if hasattr(cuboid_data, 'color') and cuboid_data.color is not None:
            color = cuboid_data.color
        else:
            # 默认颜色：半透明红色
            color = [0.8, 0.2, 0.2, 0.7]
            
        # 确保颜色是4个分量
        if len(color) == 3:
            color.append(0.7)  # 默认alpha值
            
        # 创建立方体几何体
        collision_shape = p.createCollisionShape(
            p.GEOM_BOX, 
            halfExtents=[dims[0]/2, dims[1]/2, dims[2]/2]
        )
        visual_shape = p.createVisualShape(
            p.GEOM_BOX, 
            halfExtents=[dims[0]/2, dims[1]/2, dims[2]/2],
            rgbaColor=color
        )
        
        # 位置和姿态
        position = [pose[0], pose[1], pose[2]]
        orientation = [pose[4], pose[5], pose[6], pose[3]]  # [x,y,z,w] -> [x,y,z,w]
        
        # 创建障碍物
        obstacle_id = p.createMultiBody(
            baseMass=0,  # 静态障碍物
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position,
            baseOrientation=orientation
        )
        
        self.obstacle_ids.append(obstacle_id)
        print(f"  📦 立方体 {name}: 位置 {position}, 尺寸 {dims}")
        
    def _load_sphere(self, name, sphere_data):
        """加载球体障碍物
        
        Args:
            name: 障碍物名称
            sphere_data: 球体配置数据
        """
        position = sphere_data.position
        radius = sphere_data.radius
        
        # 获取颜色信息
        if hasattr(sphere_data, 'color') and sphere_data.color is not None:
            color = sphere_data.color
        else:
            # 默认颜色：半透明绿色
            color = [0.2, 0.8, 0.2, 0.7]
            
        # 确保颜色是4个分量
        if len(color) == 3:
            color.append(0.7)
            
        # 创建球体几何体
        collision_shape = p.createCollisionShape(
            p.GEOM_SPHERE,
            radius=radius
        )
        visual_shape = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=radius,
            rgbaColor=color
        )
        
        # 创建障碍物
        obstacle_id = p.createMultiBody(
            baseMass=0,  # 静态障碍物
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position
        )
        
        self.obstacle_ids.append(obstacle_id)
        print(f"  🌕 球体 {name}: 位置 {position}, 半径 {radius}")
        
    def _load_capsule(self, name, capsule_data):
        """加载胶囊体障碍物
        
        Args:
            name: 障碍物名称  
            capsule_data: 胶囊体配置数据
        """
        radius = capsule_data.radius
        base = capsule_data.base
        tip = capsule_data.tip
        pose = capsule_data.pose
        
        # 计算胶囊体高度
        height = np.linalg.norm(np.array(tip) - np.array(base))
        
        # 获取颜色信息
        if hasattr(capsule_data, 'color') and capsule_data.color is not None:
            color = capsule_data.color
        else:
            # 默认颜色：半透明蓝色
            color = [0.2, 0.2, 0.8, 0.7]
            
        # 确保颜色是4个分量
        if len(color) == 3:
            color.append(0.7)
            
        # 创建胶囊体几何体
        collision_shape = p.createCollisionShape(
            p.GEOM_CAPSULE,
            radius=radius,
            height=height
        )
        visual_shape = p.createVisualShape(
            p.GEOM_CAPSULE,
            radius=radius,
            length=height,
            rgbaColor=color
        )
        
        # 位置和姿态
        position = [pose[0], pose[1], pose[2]]
        orientation = [pose[4], pose[5], pose[6], pose[3]]  # [x,y,z,w] -> [x,y,z,w]
        
        # 创建障碍物
        obstacle_id = p.createMultiBody(
            baseMass=0,  # 静态障碍物
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position,
            baseOrientation=orientation
        )
        
        self.obstacle_ids.append(obstacle_id)
        print(f"  💊 胶囊体 {name}: 位置 {position}, 半径 {radius}, 高度 {height:.3f}")
        
    def _load_mesh(self, name, mesh_data):
        """加载网格障碍物
        
        Args:
            name: 障碍物名称
            mesh_data: 网格配置数据
        """
        file_path = mesh_data.file_path
        pose = mesh_data.pose
        
        # 获取颜色信息
        if hasattr(mesh_data, 'color') and mesh_data.color is not None:
            color = mesh_data.color
        else:
            # 默认颜色：半透明黄色
            color = [0.8, 0.8, 0.2, 0.7]
            
        # 确保颜色是4个分量
        if len(color) == 3:
            color.append(0.7)
        
        # 构建完整的文件路径
        # 这里我们尝试从content/assets目录加载
        from curobo.util_file import get_assets_path
        full_path = join_path(get_assets_path(), file_path)
        
        print(f"  🗂️  网格 {name}: 尝试加载 {file_path}")
        
        try:
            # 尝试加载网格文件
            if os.path.exists(full_path):
                # 创建网格的视觉形状
                visual_shape = p.createVisualShape(
                    p.GEOM_MESH,
                    fileName=full_path,
                    rgbaColor=color
                )
                
                # 位置和姿态
                position = [pose[0], pose[1], pose[2]]
                orientation = [pose[4], pose[5], pose[6], pose[3]]  # [x,y,z,w] -> [x,y,z,w]
                
                # 创建障碍物（只有视觉形状，没有碰撞形状以避免复杂性）
                obstacle_id = p.createMultiBody(
                    baseMass=0,
                    baseVisualShapeIndex=visual_shape,
                    basePosition=position,
                    baseOrientation=orientation
                )
                
                self.obstacle_ids.append(obstacle_id)
                print(f"    ✅ 成功加载网格: {file_path}")
            else:
                print(f"    ❌ 网格文件未找到: {full_path}")
                # 创建一个替代的立方体
                self._create_placeholder_mesh(name, pose, color)
                
        except Exception as e:
            print(f"    ❌ 加载网格失败: {e}")
            # 创建一个替代的立方体
            self._create_placeholder_mesh(name, pose, color)
            
    def _create_placeholder_mesh(self, name, pose, color):
        """为无法加载的网格创建占位符立方体"""
        # 创建一个默认大小的立方体作为占位符
        dims = [0.2, 0.2, 0.2]  # 默认尺寸
        
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
        print(f"    🔶 使用占位符立方体替代网格 {name}")
        
    def load_world_config(self, world_cfg):
        """加载世界配置
        
        Args:
            world_cfg: WorldConfig对象
        """
        self.clear_world()
        
        total_obstacles = 0
        
        # 加载立方体
        if hasattr(world_cfg, 'cuboid') and world_cfg.cuboid is not None:
            print(f"\n📦 加载 {len(world_cfg.cuboid)} 个立方体...")
            for cuboid_data in world_cfg.cuboid:
                self._load_cuboid(cuboid_data.name, cuboid_data)
                total_obstacles += 1
                
        # 加载球体
        if hasattr(world_cfg, 'sphere') and world_cfg.sphere is not None:
            print(f"\n🌕 加载 {len(world_cfg.sphere)} 个球体...")
            for sphere_data in world_cfg.sphere:
                self._load_sphere(sphere_data.name, sphere_data)
                total_obstacles += 1
                
        # 加载胶囊体
        if hasattr(world_cfg, 'capsule') and world_cfg.capsule is not None:
            print(f"\n💊 加载 {len(world_cfg.capsule)} 个胶囊体...")
            for capsule_data in world_cfg.capsule:
                self._load_capsule(capsule_data.name, capsule_data)
                total_obstacles += 1
                
        # 加载网格
        if hasattr(world_cfg, 'mesh') and world_cfg.mesh is not None:
            print(f"\n🗂️  加载 {len(world_cfg.mesh)} 个网格...")
            for mesh_data in world_cfg.mesh:
                self._load_mesh(mesh_data.name, mesh_data)
                total_obstacles += 1
                
        # 加载圆柱体
        if hasattr(world_cfg, 'cylinder') and world_cfg.cylinder is not None:
            print(f"\n🛢️  注意: 圆柱体暂不支持可视化 ({len(world_cfg.cylinder)} 个)")
            
        # 加载体素网格
        if hasattr(world_cfg, 'voxel') and world_cfg.voxel is not None:
            print(f"\n🧊 注意: 体素网格暂不支持可视化 ({len(world_cfg.voxel)} 个)")
            
        print(f"\n✅ 成功加载 {total_obstacles} 个障碍物")
        
    def disconnect(self):
        """断开PyBullet连接"""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None


def get_available_world_configs():
    """获取所有可用的世界配置文件"""
    world_configs_path = get_world_configs_path()
    world_files = []
    
    for file in os.listdir(world_configs_path):
        if file.endswith('.yml') and file.startswith('collision_'):
            world_files.append(file)
            
    return sorted(world_files)


def display_world_menu():
    """显示世界配置文件菜单"""
    world_files = get_available_world_configs()
    
    print("\n" + "="*60)
    print("🌍 CuRobo 世界配置可视化器")
    print("="*60)
    print("\n可用的世界配置文件：")
    print("-" * 40)
    
    for i, world_file in enumerate(world_files, 1):
        # 去掉前缀和后缀，美化显示
        display_name = world_file.replace('collision_', '').replace('.yml', '')
        print(f"{i:2d}. {display_name}")
        
    print(f"\n{len(world_files)+1:2d}. 退出")
    print("-" * 40)
    
    return world_files


def visualize_world_config(world_file):
    """可视化指定的世界配置文件
    
    Args:
        world_file: 世界配置文件名
    """
    print(f"\n🔄 加载世界配置: {world_file}")
    print("="*50)
    
    try:
        # 加载世界配置
        world_cfg_dict = load_yaml(join_path(get_world_configs_path(), world_file))
        world_cfg = WorldConfig.from_dict(world_cfg_dict)
        
        # 创建可视化器
        visualizer = WorldVisualizerPyBullet(gui=True)
        
        # 加载世界配置
        visualizer.load_world_config(world_cfg)
        
        print(f"\n🎯 世界配置 '{world_file}' 可视化完成！")
        print("💡 提示:")
        print("   - 使用鼠标旋转和缩放视图")
        print("   - 不同颜色代表不同类型的几何体")
        print("   - 📦 立方体: 红色")
        print("   - 🌕 球体: 绿色") 
        print("   - 💊 胶囊体: 蓝色")
        print("   - 🗂️  网格: 黄色")
        print("\n按回车键返回菜单...")
        input()
        
        visualizer.disconnect()
        return True
        
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return False


def main():
    """主函数"""
    setup_curobo_logger("error")
    
    print("🚀 启动CuRobo世界配置可视化器...")
    
    while True:
        try:
            world_files = display_world_menu()
            
            choice = input(f"\n请选择要可视化的世界配置 (1-{len(world_files)+1}): ").strip()
            
            if choice == str(len(world_files) + 1) or choice.lower() in ['q', 'quit', 'exit']:
                print("\n👋 再见！")
                break
                
            try:
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(world_files):
                    world_file = world_files[choice_idx]
                    visualize_world_config(world_file)
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