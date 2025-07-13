#!/usr/bin/env python3
"""
使用PyBullet可视化world_representation_example.py中的几何体数据
"""

import numpy as np
import pybullet as p
import pybullet_data
import time

# 导入原脚本的数据
from world_representation_example import approximate_geometry, doc_example

# CuRobo
from curobo.geom.types import Capsule, Cuboid, Cylinder, Mesh, Sphere, WorldConfig
from curobo.util_file import get_assets_path, join_path


class WorldRepresentationVisualizer:
    """world_representation_example.py的PyBullet可视化器"""
    
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
            cameraDistance=8.0,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0, 2, 0.5]
        )
        
        # 加载地面
        self.ground_plane_id = p.loadURDF("plane.urdf")
        
        print("PyBullet environment initialized")
        
    def clear_world(self):
        """清除当前世界中的所有障碍物"""
        for obstacle_id in self.obstacle_ids:
            try:
                p.removeBody(obstacle_id)
            except:
                pass
        self.obstacle_ids.clear()
        
    def _load_cuboid(self, name, cuboid_data):
        """加载立方体障碍物"""
        dims = cuboid_data.dims
        pose = cuboid_data.pose
        
        # 获取颜色信息
        if hasattr(cuboid_data, 'color') and cuboid_data.color is not None:
            color = cuboid_data.color
        else:
            color = [0.8, 0.2, 0.2, 0.7]  # 默认红色
            
        # 确保颜色是4个分量
        if len(color) == 3:
            color.append(0.7)
            
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
            baseMass=0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position,
            baseOrientation=orientation
        )
        
        self.obstacle_ids.append(obstacle_id)
        print(f"  📦 Cuboid {name}: position {position}, dimensions {dims}")
        
    def _load_sphere(self, name, sphere_data):
        """加载球体障碍物"""
        position = [sphere_data.pose[0], sphere_data.pose[1], sphere_data.pose[2]]
        radius = sphere_data.radius
        
        # 获取颜色信息
        if hasattr(sphere_data, 'color') and sphere_data.color is not None:
            color = sphere_data.color
        else:
            color = [0.2, 0.8, 0.2, 0.7]  # 默认绿色
            
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
        
        # 姿态
        orientation = [sphere_data.pose[4], sphere_data.pose[5], sphere_data.pose[6], sphere_data.pose[3]]
        
        # 创建障碍物
        obstacle_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position,
            baseOrientation=orientation
        )
        
        self.obstacle_ids.append(obstacle_id)
        print(f"  🌕 Sphere {name}: position {position}, radius {radius}")
        
    def _load_capsule(self, name, capsule_data):
        """加载胶囊体障碍物"""
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
            color = [0.2, 0.2, 0.8, 0.7]  # 默认蓝色
            
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
        orientation = [pose[4], pose[5], pose[6], pose[3]]
        
        # 创建障碍物
        obstacle_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position,
            baseOrientation=orientation
        )
        
        self.obstacle_ids.append(obstacle_id)
        print(f"  💊 Capsule {name}: position {position}, radius {radius}, height {height:.3f}")
        
    def _load_cylinder(self, name, cylinder_data):
        """加载圆柱体障碍物"""
        radius = cylinder_data.radius
        height = cylinder_data.height
        pose = cylinder_data.pose
        
        # 获取颜色信息
        if hasattr(cylinder_data, 'color') and cylinder_data.color is not None:
            color = cylinder_data.color
        else:
            color = [0.8, 0.2, 0.8, 0.7]  # 默认紫色
            
        # 确保颜色是4个分量
        if len(color) == 3:
            color.append(0.7)
            
        # 创建圆柱体几何体
        collision_shape = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=radius,
            height=height
        )
        visual_shape = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=radius,
            length=height,
            rgbaColor=color
        )
        
        # 位置和姿态
        position = [pose[0], pose[1], pose[2]]
        orientation = [pose[4], pose[5], pose[6], pose[3]]
        
        # 创建障碍物
        obstacle_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position,
            baseOrientation=orientation
        )
        
        self.obstacle_ids.append(obstacle_id)
        print(f"  🛢️ Cylinder {name}: position {position}, radius {radius}, height {height}")
        
    def _load_mesh_placeholder(self, name, mesh_data):
        """为网格创建占位符立方体（因为不加载assets）"""
        pose = mesh_data.pose
        
        # 获取颜色信息
        if hasattr(mesh_data, 'color') and mesh_data.color is not None:
            color = mesh_data.color
        else:
            color = [0.8, 0.8, 0.2, 0.7]  # 默认黄色
            
        # 确保颜色是4个分量
        if len(color) == 3:
            color.append(0.7)
            
        # 创建一个默认大小的立方体作为占位符
        dims = [0.3, 0.3, 0.3]  # 默认尺寸
        
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
        print(f"  🔶 Mesh {name}: placeholder box at position {position}")
        
    def visualize_doc_example(self):
        """可视化doc_example中的几何体"""
        print("\n🔄 Visualizing doc_example geometries...")
        print("="*50)
        
        self.clear_world()
        
        # 创建doc_example中的几何体
        obstacle_1 = Cuboid(
            name="cube_1",
            pose=[0.0, 0.0, 0.0, 0.043, -0.471, 0.284, 0.834],
            dims=[0.2, 1.0, 0.2],
            color=[0.8, 0.0, 0.0, 1.0],
        )
        
        # 跳过网格加载（因为用户要求不加载assets）
        mesh_file = join_path(get_assets_path(), "scene/nvblox/srl_ur10_bins.obj")
        obstacle_2 = Mesh(
            name="mesh_1",
            pose=[0.0, 2, 0.5, 0.043, -0.471, 0.284, 0.834],
            file_path=mesh_file,
            scale=[0.5, 0.5, 0.5],
        )
        
        obstacle_3 = Capsule(
            name="capsule",
            radius=0.2,
            base=[0, 0, 0],
            tip=[0, 0, 0.5],
            pose=[0.0, 5, 0.0, 0.043, -0.471, 0.284, 0.834],
            color=[0, 1.0, 0, 1.0],
        )
        
        obstacle_4 = Cylinder(
            name="cylinder_1",
            radius=0.2,
            height=0.5,
            pose=[0.0, 6, 0.0, 0.043, -0.471, 0.284, 0.834],
            color=[0, 1.0, 0, 1.0],
        )
        
        obstacle_5 = Sphere(
            name="sphere_1",
            radius=0.2,
            pose=[0.0, 7, 0.0, 0.043, -0.471, 0.284, 0.834],
            color=[0, 1.0, 0, 1.0],
        )
        
        # 加载几何体
        self._load_cuboid("cube_1", obstacle_1)
        self._load_mesh_placeholder("mesh_1", obstacle_2)  # 占位符
        self._load_capsule("capsule", obstacle_3)
        self._load_cylinder("cylinder_1", obstacle_4)
        self._load_sphere("sphere_1", obstacle_5)
        
        print(f"\n✅ Successfully loaded {len(self.obstacle_ids)} geometries")
        
    def visualize_approximate_geometry(self):
        """可视化approximate_geometry中的几何体"""
        print("\n🔄 Visualizing approximate_geometry...")
        print("="*50)
        
        self.clear_world()
        
        # 创建approximate_geometry中的胶囊体
        obstacle_capsule = Capsule(
            name="capsule",
            radius=0.2,
            base=[0, 0, 0],
            tip=[0, 0, 0.5],
            pose=[0.0, 5, 0.0, 0.043, -0.471, 0.284, 0.834],
            color=[0, 1.0, 0, 1.0],
        )
        
        # 加载胶囊体
        self._load_capsule("capsule", obstacle_capsule)
        
        print(f"\n✅ Successfully loaded {len(self.obstacle_ids)} geometries")
        
    def disconnect(self):
        """断开PyBullet连接"""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None


def main():
    """主函数"""
    print("🚀 Starting world_representation_example.py PyBullet visualizer...")
    
    # 创建可视化器
    visualizer = WorldRepresentationVisualizer(gui=True)
    
    while True:
        print("\n" + "="*60)
        print("🌍 World Representation Example Visualizer")
        print("="*60)
        print("\nChoose visualization:")
        print("1. doc_example() - Multiple geometry types")
        print("2. approximate_geometry() - Capsule only")
        print("3. Exit")
        print("-" * 40)
        
        try:
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == '1':
                visualizer.visualize_doc_example()
                print("\n💡 Tips:")
                print("   - Use mouse to rotate and zoom the view")
                print("   - Different colors represent different geometry types")
                print("   - 📦 Cuboid: Red")
                print("   - 🔶 Mesh (placeholder): Yellow")
                print("   - 💊 Capsule: Green")
                print("   - 🛢️ Cylinder: Green")
                print("   - 🌕 Sphere: Green")
                print("\nPress Enter to continue...")
                input()
                
            elif choice == '2':
                visualizer.visualize_approximate_geometry()
                print("\n💡 Tips:")
                print("   - This shows only the capsule from approximate_geometry()")
                print("   - 💊 Capsule: Green")
                print("\nPress Enter to continue...")
                input()
                
            elif choice == '3':
                print("\n👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice, please try again")
                
        except KeyboardInterrupt:
            print("\n\n👋 User interrupted, exiting...")
            break
        except Exception as e:
            print(f"❌ Error occurred: {e}")
            
    visualizer.disconnect()


if __name__ == "__main__":
    main() 