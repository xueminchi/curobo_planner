# Pick and Place 简化版本 - 使用指南

## 简介

`pick_and_place_simple.py` 是基于原始 `pick_and_place_fixed.py` 的简化版本，专注于提供流畅的演示体验，减少了调试输出和冗余代码。

## 新功能特点

### 🎥 视频录制功能
- **自动文件夹创建**: 基于当前日期时间创建专用文件夹
- **分段录制**: 每个演示阶段单独录制视频
- **时间戳标记**: 每个视频文件都包含精确的时间戳
- **完整场景录制**: 录制整个PyBullet场景，包括机器人、物体和环境

### 🚀 性能优化
- **2倍播放速度**: 相比原版本提升了演示效率
- **简化输出**: 减少了不必要的调试信息
- **流畅执行**: 取消了大部分用户交互等待

### 💎 可视化增强
- **球体标记**: 显示CuRobo内部的物体几何表示
- **实时更新**: 球体随机器人末端执行器移动
- **颜色编码**: 不同阶段使用不同颜色的位置标记

## 使用方法

### 基本使用
```bash
python pick_and_place_simple.py
```

### 启用视频录制
运行脚本后，程序会询问是否启用视频录制功能：
```
是否启用视频录制功能？(y/n): y
```

### 视频文件组织
启用视频录制后，会自动创建以下文件夹结构：
```
pick_and_place_videos_20250101_143052/
├── approach_143055.mp4      # 接近阶段
├── grasp_143102.mp4         # 抓取阶段  
├── place_143115.mp4         # 放置阶段
└── return_143125.mp4        # 返回阶段
```

## 演示流程

### 阶段1: 🚀 移动到接近位置
- 从起始位置移动到目标物体上方安全距离
- 视频文件: `approach_HHMMSS.mp4`

### 阶段2: 🎯 移动到抓取位置  
- 从接近位置下降到抓取高度
- 视频文件: `grasp_HHMMSS.mp4`

### 阶段3: 🤏 抓取物体
- 将物体附加到机器人末端执行器
- 创建球体标记显示物体几何表示

### 阶段4: 🚚 移动到放置位置
- 携带物体移动到目标放置位置
- 球体标记实时跟随机器人移动
- 视频文件: `place_HHMMSS.mp4`

### 阶段5: 📤 放置物体
- 将物体从机器人分离
- 清除球体标记

### 阶段6: 🏠 返回起始位置
- 机器人返回到安全起始位置
- 视频文件: `return_HHMMSS.mp4`

## 视频录制技术细节

### 文件命名规则
- 文件夹格式: `pick_and_place_videos_YYYYMMDD_HHMMSS`
- 视频文件格式: `{阶段名}_{时间戳}.mp4`
- 时间戳精确到秒: `HHMMSS`

### 录制参数
- 格式: MP4
- 内容: 完整PyBullet场景
- 帧率: 跟随PyBullet仿真步长
- 录制范围: 整个演示窗口

### 存储位置
- 默认位置: 脚本运行目录
- 可通过修改 `_setup_video_recording()` 方法自定义路径

## 代码对比

### 原版本 vs 简化版本
| 特性 | 原版本 | 简化版本 |
|------|--------|----------|
| 代码行数 | 880行 | 511行 |
| 减少比例 | - | 42% |
| 调试输出 | 详细 | 精简 |
| 播放速度 | 1x | 2x |
| 用户交互 | 多次等待 | 最少等待 |
| 球体可视化 | 复杂实现 | 简化实现 |
| 视频录制 | 无 | ✅ 自动录制 |

### 性能提升
- **执行时间**: 减少约50%
- **代码复杂度**: 显著降低
- **用户体验**: 更流畅的演示流程
- **功能完整性**: 保持所有核心功能

## 配置选项

### 世界配置
```python
world_config = {
    "cuboid": {
        "table": {"dims": [1.2, 1.2, 0.05], "pose": [0.4, 0.0, -0.025, 1, 0, 0, 0.0]},
        "target_cube": {"dims": [0.05, 0.05, 0.05], "pose": [0.45, 0.35, 0.025, 1, 0, 0, 0.0]},
        "obstacle1": {"dims": [0.08, 0.08, 1.2], "pose": [-0.2, -0.3, 0.6, 1, 0, 0, 0.0]},
        "obstacle2": {"dims": [0.35, 0.1, 1.1], "pose": [0.6, 0.0, 0.55, 1, 0, 0, 0.0]}
    }
}
```

### 运动规划配置
```python
motion_gen_config = MotionGenConfig.load_from_robot_config(
    robot_file="franka.yml",
    world_config=world_config,
    interpolation_dt=0.02,
    collision_checker_type=CollisionCheckerType.PRIMITIVE,
    use_cuda_graph=True,
    num_trajopt_seeds=4,
    num_graph_seeds=4,
)
```

### 视频录制配置
```python
# 在PickAndPlaceVisualizer.__init__()中
enable_video=True  # 启用视频录制
```

## 故障排除

### 常见问题

1. **视频文件夹创建失败**
   - 检查脚本运行目录的写入权限
   - 确保磁盘空间充足

2. **视频录制失败**
   - 确认PyBullet版本支持视频录制
   - 检查系统是否安装了必要的编解码器

3. **规划失败**
   - 调整障碍物位置
   - 增加规划尝试次数（`max_attempts`）

4. **球体标记不显示**
   - 检查物体是否成功附加到机器人
   - 确认`motion_gen`对象正确设置

### 调试选项
```python
# 启用详细日志
setup_curobo_logger("debug")

# 增加规划尝试次数
MotionGenPlanConfig(max_attempts=10)

# 调整播放速度
visualizer.play_trajectory(trajectory, speed=1.0)  # 降低速度便于观察
```

## 扩展功能

### 自定义视频格式
可以在`start_video_recording()`方法中修改视频参数：
```python
p.startStateLogging(
    p.STATE_LOGGING_VIDEO_MP4,
    video_path,
    objectUniqueIds=[]  # 可指定特定物体ID
)
```

### 添加更多阶段
可以轻松添加新的演示阶段：
```python
# 新增阶段示例
print(f"\n🔄 阶段7: 自定义动作")
# ... 规划和执行代码 ...
visualizer.play_trajectory(trajectory, stage_name="custom_action")
```

### 自定义标记
```python
# 添加自定义位置标记
visualizer.add_position_marker(
    position=[x, y, z],
    size=0.03,
    color=[1, 0, 0, 1]  # 红色标记
)
```

## 系统要求

- Python 3.8+
- PyBullet 3.2.0+
- CuRobo (最新版本)
- PyTorch 1.12+
- CUDA 11.6+ (推荐)
- 足够的磁盘空间用于视频存储

## 许可证

本项目遵循与CuRobo相同的许可证。 