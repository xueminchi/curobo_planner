#!/usr/bin/env python3
"""
修正版本：将cuRobo的成本函数值转换为更直观的距离表示
"""

import numpy as np
import torch

# CuRobo
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import Cuboid, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.util.logger import setup_curobo_logger
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig

class CorrectedDistanceCalculator:
    """修正的距离计算器，提供更直观的距离表示"""
    
    def __init__(self, robot_file="franka.yml", activation_distance=1.0):
        self.tensor_args = TensorDeviceType()
        self.robot_file = robot_file
        self.activation_distance = activation_distance
        self.robot_world = None
        
    def setup(self, world_config):
        """设置碰撞检测器"""
        robot_world_config = RobotWorldConfig.load_from_config(
            self.robot_file,
            world_config,
            self.tensor_args,
            collision_activation_distance=self.activation_distance,
            collision_checker_type=CollisionCheckerType.PRIMITIVE,
        )
        
        self.robot_world = RobotWorld(robot_world_config)
        
    def get_corrected_distance(self, joint_positions):
        """获取修正后的距离信息"""
        if self.robot_world is None:
            return None
            
        # 确保joint_positions是正确的tensor格式
        if not torch.is_tensor(joint_positions):
            joint_positions = torch.tensor(joint_positions, dtype=self.tensor_args.dtype, device=self.tensor_args.device)
        
        if joint_positions.dim() == 1:
            joint_positions = joint_positions.unsqueeze(0)
            
        # 获取原始成本值
        d_world_cost, d_self_cost = self.robot_world.get_world_self_collision_distance_from_joints(joint_positions)
        
        # 获取机械臂球体信息
        kin_state = self.robot_world.get_kinematics(joint_positions)
        spheres = kin_state.link_spheres_tensor[0]  # 取第一个batch
        
        # 计算几何距离（基于球体近似）
        geometric_info = self._calculate_geometric_distance(spheres)
        
        # 转换成本值为估计距离
        estimated_distance = self._cost_to_distance(d_world_cost.item(), self.activation_distance)
        
        return {
            'cost_value': d_world_cost.item(),
            'self_collision_cost': d_self_cost.item(),
            'estimated_distance': estimated_distance,
            'geometric_info': geometric_info,
            'activation_distance': self.activation_distance
        }
    
    def _calculate_geometric_distance(self, spheres):
        """基于球体近似计算几何距离信息"""
        # 过滤掉异常的球体（半径为负或过大）
        valid_spheres = spheres[spheres[:, 3] > 0]
        valid_spheres = valid_spheres[valid_spheres[:, 3] < 1.0]  # 过滤掉半径>1m的异常球体
        
        if len(valid_spheres) == 0:
            return None
            
        return {
            'num_spheres': len(valid_spheres),
            'avg_radius': valid_spheres[:, 3].mean().item(),
            'min_radius': valid_spheres[:, 3].min().item(),
            'max_radius': valid_spheres[:, 3].max().item(),
            'end_effector_pos': valid_spheres[-5:, :3].mean(dim=0).cpu().numpy()  # 末端附近球体的平均位置
        }
    
    def _cost_to_distance(self, cost_value, eta):
        """将成本值转换为估计距离"""
        if cost_value == 0:
            return f"> {eta}m (安全)"
        elif cost_value < 0.5 * eta:
            # 在激活区域内，尝试反推距离
            # 这是一个近似，因为实际的成本函数更复杂
            estimated_dist = eta - np.sqrt(cost_value * eta)
            return f"~{estimated_dist:.3f}m (近似)"
        else:
            # 在碰撞区域，使用线性关系
            estimated_dist = cost_value - 0.5 * eta
            return f"~{estimated_dist:.3f}m (碰撞)"
    
    def print_corrected_distance(self, joint_positions, phase=""):
        """打印修正后的距离信息"""
        result = self.get_corrected_distance(joint_positions)
        
        if result is None:
            print(f"❌ 距离计算失败")
            return
            
        phase_info = f"[{phase}] " if phase else ""
        
        print(f"📏 {phase_info}距离分析:")
        print(f"   🔢 原始成本值: {result['cost_value']:.4f}")
        print(f"   📐 估计距离: {result['estimated_distance']}")
        print(f"   🤖 自碰撞成本: {result['self_collision_cost']:.4f}")
        
        if result['geometric_info']:
            geo = result['geometric_info']
            print(f"   🎯 几何信息:")
            print(f"     - 有效球体数: {geo['num_spheres']}")
            print(f"     - 平均半径: {geo['avg_radius']:.3f}m")
            print(f"     - 半径范围: {geo['min_radius']:.3f}m - {geo['max_radius']:.3f}m")
            print(f"     - 末端位置: [{geo['end_effector_pos'][0]:.3f}, {geo['end_effector_pos'][1]:.3f}, {geo['end_effector_pos'][2]:.3f}]")
        
        # 根据成本值给出警告
        if result['cost_value'] > 2.0:
            print(f"   ⚠️  警告: 成本值较高，可能接近障碍物")
        elif result['cost_value'] > 1.0:
            print(f"   ⚡ 注意: 成本值中等，需要关注")
        else:
            print(f"   ✅ 成本值较低，相对安全")

def test_corrected_distance():
    """测试修正后的距离计算"""
    setup_curobo_logger("info")
    
    print("🔧 修正版距离计算测试")
    print("=" * 60)
    
    # 创建测试场景
    cuboid = Cuboid(
        name="test_cube",
        pose=[0.5, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0],
        dims=[0.1, 0.1, 0.1],
    )
    
    world_config = WorldConfig(cuboid=[cuboid])
    
    # 创建修正的距离计算器
    calculator = CorrectedDistanceCalculator(activation_distance=1.0)
    calculator.setup(world_config)
    
    # 测试不同的关节配置
    test_configs = [
        ([0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0], "初始位置"),
        ([0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0], "向前伸展"),
        ([0.0, -0.5, 0.0, -1.5, 0.0, 1.0, 0.0], "向上抬起"),
        ([0.5, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0], "向右移动"),
    ]
    
    for config, description in test_configs:
        print(f"\n📍 测试配置: {description}")
        print(f"   关节角度: {config}")
        calculator.print_corrected_distance(config, description)
        print("-" * 40)

def main():
    """主函数"""
    print("🎯 cuRobo距离计算修正版本")
    print("=" * 80)
    print("📖 目标：将成本函数值转换为更直观的距离表示")
    print()
    
    test_corrected_distance()
    
    print("\n📋 修正说明:")
    print("=" * 60)
    print("✅ 原始成本值：cuRobo内部的成本函数输出")
    print("✅ 估计距离：基于成本值的距离估计")
    print("✅ 几何信息：基于球体近似的几何分析")
    print("✅ 过滤异常：去除半径异常的球体")
    print("✅ 直观警告：基于成本值的安全级别提示")
    
    print(f"\n💡 使用建议:")
    print(f"   - 使用估计距离进行直观判断")
    print(f"   - 关注成本值的变化趋势")
    print(f"   - 结合几何信息理解机械臂状态")
    print(f"   - 成本值>2.0时需要特别注意")

if __name__ == "__main__":
    main() 