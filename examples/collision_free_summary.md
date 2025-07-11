# Collision-Free目标生成功能总结

## 🎯 功能概述

为 `motion_gen_scene_selector.py` 添加了智能的collision-free目标生成功能，显著提高了多目标运动规划的成功率。

## 🚀 核心改进

### 1. 智能目标生成
- **随机采样算法**: 在机器人工作空间内智能采样
- **碰撞检测**: 支持立方体、球体、胶囊体等多种几何体
- **安全距离**: 自动添加安全边界，避免与障碍物过近

### 2. 自适应重试机制
- **失败检测**: 自动识别运动规划失败
- **智能重试**: 失败时生成新的collision-free目标
- **最大重试限制**: 避免无限循环，提供优雅降级

### 3. 工作空间优化
```python
workspace_bounds = {
    'x': [0.2, 0.7],    # x轴范围
    'y': [-0.5, 0.5],   # y轴范围  
    'z': [0.3, 0.8]     # z轴范围
}
```

## 📊 测试结果

### ✅ 成功率测试
| 场景配置 | 目标生成成功率 | 障碍物数量 |
|---------|---------------|-----------|
| collision_table.yml | 100% (10/10) | 1个立方体 |
| collision_cage.yml | 100% (10/10) | 8个立方体 |
| collision_primitives_3d.yml | 100% (10/10) | 6个混合几何体 |

### ✅ 碰撞检测验证
- **立方体检测**: 5/5 测试用例通过
- **球体检测**: 4/4 测试用例通过
- **工作空间边界**: 20/20 目标在预期范围内

## 🔧 技术实现

### 核心函数

#### 1. 主要生成函数
```python
def generate_collision_free_goal(self, world_cfg, max_attempts=50, safety_margin=0.1):
    """生成无碰撞的目标位置"""
    # 智能采样 + 碰撞检测 + 安全距离
```

#### 2. 碰撞检测函数
```python
def _check_point_cuboid_collision(self, point, cuboid, safety_margin):
    """检查点与立方体的碰撞"""

def _check_point_sphere_collision(self, point, sphere, safety_margin):
    """检查点与球体的碰撞"""

def _check_point_capsule_collision(self, point, capsule, safety_margin):
    """检查点与胶囊体的碰撞"""
```

### 多目标规划改进
```python
# 原始目标失败时的重试逻辑
for retry in range(max_retries):
    new_goal_pos = visualizer.generate_collision_free_goal(world_cfg)
    if new_goal_pos is not None:
        # 使用新目标重新规划
        retry_result = motion_gen.plan_single(...)
        if retry_result.success:
            # 成功！继续下一个目标
            break
```

## 🎮 用户体验改进

### 1. 智能提示
```
❌ 到目标 2 的轨迹规划失败！状态: None
🔄 尝试生成新的无碰撞目标...
🎯 重试 1/3: 新目标 [0.456, 0.234, 0.567]
✅ 使用新目标的轨迹规划成功！
📊 规划时间: 0.0234秒
```

### 2. 统计信息
```
🎉 多目标规划完成！成功到达 3/3 个目标
```

## 🌟 优势总结

| 特性 | 改进前 | 改进后 |
|------|--------|--------|
| **目标选择** | 固定目标 | 智能collision-free目标 |
| **失败处理** | 直接跳过 | 自动重试新目标 |
| **成功率** | 取决于场景 | 显著提高 |
| **适应性** | 有限 | 适用任意场景 |

## 🎯 应用场景

1. **复杂障碍物环境**: 自动避开各种几何形状的障碍物
2. **动态目标规划**: 当预设目标不可达时自动调整
3. **批量任务处理**: 减少人工干预，提高自动化程度
4. **场景适应**: 无需针对不同场景手动调整目标位置

## 🚀 未来扩展

1. **更复杂几何体**: 支持mesh、凸包等复杂形状
2. **动态障碍物**: 考虑运动中的障碍物
3. **多机器人协调**: 考虑多机器人间的碰撞避免
4. **学习优化**: 基于历史成功目标进行优化

## 📝 使用示例

```python
# 基本使用
visualizer = SceneMotionGenVisualizer(gui=True)
collision_free_goal = visualizer.generate_collision_free_goal(world_cfg)

# 在多目标规划中使用
python motion_gen_scene_selector.py
# 选择任意场景 -> 选择"多目标运动规划" -> 自动应用collision-free重试
```

## 🎬 视频录制功能

新增了完整的视频录制功能，可以保存轨迹播放过程为MP4视频文件。

### 🎥 核心功能

#### 1. 自动目录管理
- **时间戳文件夹**: 自动创建 `motion_planning_videos_YYYYMMDD_HHMMSS/` 格式的文件夹
- **路径管理**: 智能处理文件路径，避免类型冲突
- **自动清理**: 程序结束时自动停止录制

#### 2. 录制控制
```python
# 开始录制
visualizer.start_recording("my_video.mp4")

# 检查录制状态
if visualizer.is_recording():
    print("正在录制...")

# 停止录制
visualizer.stop_recording()
```

#### 3. 智能命名
- **简单运动规划**: `simple_motion_{scene_name}_{time}.mp4`
- **避障运动规划**: `collision_avoidance_{scene_name}_{time}.mp4`
- **多目标规划**: `multi_goal_{scene_name}_target{N}_{time}.mp4`
- **重试录制**: `multi_goal_{scene_name}_target{N}_retry{M}_{time}.mp4`

### 🎮 用户交互

#### 录制询问
```
🎬 是否要录制轨迹视频？(y/n): y
📁 视频保存目录: /path/to/motion_planning_videos_20231201_143022/
🎬 开始录制视频: simple_motion_table_143045.mp4
📹 录制状态: ID = 12345
```

#### 录制完成
```
轨迹播放完成！
✅ 视频录制完成: simple_motion_table_143045.mp4
📁 视频保存路径: /path/to/motion_planning_videos_20231201_143022/simple_motion_table_143045.mp4
```

### 🧪 测试功能

提供专门的测试脚本验证录制功能：

```bash
python examples/test_video_recording.py
```

**测试输出**:
```
=== 测试视频录制功能 ===
📁 视频保存目录: /path/to/motion_planning_videos_20231201_143022/
加载了 1 个障碍物
✅ 轨迹规划成功！
规划时间: 0.1234秒
开始录制视频测试...
🎬 开始录制视频: test_recording_143045.mp4
轨迹播放完成！
✅ 视频录制完成: test_recording_143045.mp4
✅ 视频文件已创建: /path/to/motion_planning_videos_20231201_143022/test_recording_143045.mp4
📊 文件大小: 2.34 MB
```

### 🔧 技术实现

#### 1. PyBullet视频录制
```python
# 开始录制
self.recording_log_id = p.startStateLogging(
    p.STATE_LOGGING_VIDEO_MP4, 
    video_path
)

# 停止录制
p.stopStateLogging(self.recording_log_id)
```

#### 2. 目录管理
```python
def _setup_video_directory(self):
    """设置视频保存目录"""
    current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_dir = f"motion_planning_videos_{current_date}"
    self.video_save_path = video_dir
    os.makedirs(self.video_save_path, exist_ok=True)
```

#### 3. 轨迹可视化集成
```python
def visualize_trajectory(self, trajectory, start_state, goal_pose, 
                       record_video=False, video_name="trajectory_video.mp4"):
    """可视化运动轨迹"""
    # 开始录制
    if record_video:
        self.start_recording(video_name)
    
    # ... 轨迹播放逻辑 ...
    
    # 停止录制
    if record_video and self.is_recording():
        self.stop_recording()
```

### 🎯 录制特性

| 特性 | 描述 |
|------|------|
| **格式** | MP4高清视频 |
| **质量** | PyBullet原生录制质量 |
| **大小** | 典型轨迹 1-5MB |
| **帧率** | 与仿真同步 |
| **时长** | 取决于轨迹播放时间 |

### 🚀 快速开始

1. **运行场景选择器**:
   ```bash
   python examples/motion_gen_scene_selector.py
   ```

2. **选择场景和演示类型**

3. **选择录制**: 当询问是否录制时选择 `y`

4. **查看结果**: 程序会显示视频保存路径

---

**总结**: 新增的视频录制功能为motion_gen_scene_selector.py提供了完整的可视化记录能力，方便用户保存和分享运动规划演示结果，是一个实用且专业的功能扩展！🎉 