# PyBullet 机器人可视化工具套件

这个项目提供了完整的PyBullet机器人可视化解决方案，包括运动学、逆运动学和运动规划的可视化，作为CuRobo的直观可视化方案。

## 安装依赖

```bash
pip install -r requirements_pybullet.txt
```

## 功能特性

### 核心可视化器类

#### PyBulletKinematicsVisualizer 类 (基础)

基础可视化器类，提供以下功能：

- **机器人模型加载**: 自动从CuRobo配置文件加载URDF模型
- **关节角度设置**: 设置和获取机器人关节角度
- **末端执行器位姿**: 获取末端执行器的位置和姿态
- **随机配置可视化**: 生成并可视化随机的机器人配置
- **轨迹可视化**: 可视化关节角度轨迹
- **与CuRobo比较**: 比较PyBullet和CuRobo的运动学计算结果

#### IKPyBulletVisualizer 类 (逆运动学)

扩展的IK可视化器，提供：

- **目标位置标记**: 红色球体显示IK目标位置
- **求解结果标记**: 绿色立方体显示IK求解结果
- **IK求解过程可视化**: 逐步显示多个IK解
- **避障IK可视化**: 显示障碍物并可视化避障IK求解
- **多目标IK**: 批量处理和可视化多个目标

#### MotionGenPyBulletVisualizer 类 (运动规划)

最新的运动规划可视化器，提供：

- **完整轨迹可视化**: 播放完整的运动规划轨迹
- **起始/目标标记**: 绿色立方体(起始)和红色球体(目标)
- **轨迹点显示**: 蓝色轨迹点显示运动路径
- **避障轨迹**: 可视化复杂环境中的避障运动
- **多目标规划**: 连续多个目标的运动规划
- **实时播放控制**: 可调节的播放速度和步进控制

### 使用方法

#### 基本可视化

```python
from pybullet_kinematics_visualization import PyBulletKinematicsVisualizer

# 创建可视化器
visualizer = PyBulletKinematicsVisualizer(gui=True)

# 设置关节角度
joint_angles = [0.0, -1.3, 0.0, -2.5, 0.0, 1.0, 0.0, 0.04, 0.04]
visualizer.set_joint_angles(joint_angles)

# 获取末端执行器位姿
ee_pos, ee_orn = visualizer.get_end_effector_pose()
print(f"End effector position: {ee_pos}")
```

#### 随机配置可视化

```python
# 可视化5个随机配置，每个配置显示2秒
visualizer.visualize_random_configurations(num_configs=5, delay=2.0)
```

#### 轨迹可视化

```python
# 创建轨迹（例如：正弦波运动）
trajectory = []
for t in np.linspace(0, 2*np.pi, 50):
    config = [np.sin(t) * 0.5, 0, 0, 0, 0, 0, 0, 0.04, 0.04]
    trajectory.append(config)

# 可视化轨迹
visualizer.visualize_trajectory(trajectory, delay=0.1)
```

#### 逆运动学可视化

```python
from ik_pybullet_visualization import IKPyBulletVisualizer

# 创建IK可视化器
ik_visualizer = IKPyBulletVisualizer(gui=True)

# 运行IK演示
# 1. 基础IK可视化
# 2. 多目标IK可视化  
# 3. 避障IK可视化（包含障碍物显示）
```

#### 运动规划可视化

```python
from motion_gen_pybullet_visualization import MotionGenPyBulletVisualizer

# 创建运动规划可视化器
motion_visualizer = MotionGenPyBulletVisualizer(gui=True)

# 可视化运动规划轨迹
motion_visualizer.visualize_trajectory(
    trajectory=interpolated_trajectory,
    start_state=start_joint_state,
    goal_pose=target_pose,
    interpolation_dt=0.02,
    playback_speed=0.5,  # 半速播放
    show_trajectory_points=True  # 显示轨迹点
)
```

## 运行示例

### 1. 基本运动学可视化演示

```bash
python pybullet_kinematics_visualization.py
```

这将运行基本的可视化演示，包括：
1. 重置到收缩配置
2. 可视化5个随机配置
3. 可视化正弦波轨迹
4. 重置到收缩配置

### 2. 逆运动学可视化演示

```bash
python ik_pybullet_visualization.py
```

提供三种IK演示模式：
1. **基础IK可视化**: 生成随机目标，显示IK求解过程
2. **多目标IK可视化**: 连续求解多个目标位置
3. **避障IK可视化**: 在有障碍物环境中的IK求解

### 3. 运动规划可视化演示

```bash
python motion_gen_pybullet_visualization.py
```

提供三种运动规划演示：
1. **简单运动规划**: 基础的点到点运动规划
2. **避障运动规划**: 复杂环境中的避障轨迹规划
3. **多目标运动规划**: 连续多个目标的运动规划序列

### 与CuRobo比较

要运行与CuRobo的比较演示，请编辑文件最后的部分：

```python
if __name__ == "__main__":
    # 运行基本可视化演示
    # demo_pybullet_visualization()
    
    # 运行与CuRobo的比较
    demo_compare_with_curobo()
```

这将：
1. 使用CuRobo生成随机关节配置
2. 在PyBullet中设置相同的配置
3. 比较两者计算的末端执行器位置
4. 显示位置差异

## 支持的机器人

目前支持所有CuRobo配置文件中定义的机器人，包括：
- Franka Panda (默认)
- UR10e
- Kinova Gen3
- 其他CuRobo支持的机器人

要使用不同的机器人，只需更改配置文件名：

```python
# 使用UR10e机器人
visualizer = PyBulletKinematicsVisualizer(robot_config_name="ur10e.yml")
```

## 注意事项

1. **网格文件路径**: 确保URDF文件中的网格文件路径是正确的
2. **关节限制**: 可视化器会自动使用URDF中定义的关节限制
3. **末端执行器**: 自动从CuRobo配置中获取末端执行器链接名称
4. **性能**: PyBullet可视化比CuRobo慢，但提供了更直观的3D可视化

## 故障排除

### 常见问题

1. **URDF加载失败**
   - 检查网格文件是否存在
   - 确认URDF文件路径正确

2. **关节映射错误**
   - 检查CuRobo配置文件中的关节名称
   - 确认URDF中的关节名称匹配

3. **末端执行器未找到**
   - 检查CuRobo配置中的`ee_link`设置
   - 确认URDF中存在该链接

### 调试模式

可以通过设置`gui=False`来使用无头模式进行调试：

```python
visualizer = PyBulletKinematicsVisualizer(gui=False)
```

这在服务器环境或批处理脚本中很有用。 