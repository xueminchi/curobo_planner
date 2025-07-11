# Pick and Place 演示脚本

这里包含了完整的机械臂Pick and Place演示，展示了CuRobo的物体附加和分离功能。

## 🎯 功能特性

### 核心技术
- **物体附加**: 使用 `attach_objects_to_robot()` 将世界中的物体附加到机器人
- **避障规划**: 附加物体后自动成为机器人的一部分，参与碰撞检测
- **物体分离**: 使用 `detach_object_from_robot()` 将物体从机器人分离
- **智能路径规划**: 自动考虑附加物体的尺寸进行避障

### 可视化特性
- **PyBullet实时可视化**: 完整的3D可视化环境
- **物体跟随效果**: 附加物体实时跟随末端执行器
- **分阶段演示**: 用户可控制演示进度
- **状态指示器**: 清晰的视觉标记显示抓取位置、目标位置等

## 📁 文件说明

### 1. `pick_and_place_demo.py` - 完整演示脚本
**功能**: 包含PyBullet可视化的完整Pick and Place演示

**演示流程**:
1. 🚀 从起始位置移动到抓取位置
2. 🤏 抓取物体（附加到机器人）
3. 🚚 携带物体移动到放置位置
4. 📤 放置物体（从机器人分离）
5. 🏠 返回起始位置

**运行方式**:
```bash
python pick_and_place_demo.py
```

**特点**:
- 交互式演示，用户按回车键控制进度
- PyBullet 3D可视化
- 详细的中文提示和状态反馈
- 物体跟随末端执行器的视觉效果

### 2. `test_pick_and_place.py` - 功能测试脚本
**功能**: 验证attach/detach功能的基本工作原理（无可视化）

**测试内容**:
- 📋 世界模型初始化测试
- 🤏 `attach_objects_to_robot()` 功能测试
- 🚚 携带物体的运动规划测试
- 📤 `detach_object_from_robot()` 功能测试
- 📊 世界状态变化验证

**运行方式**:
```bash
python test_pick_and_place.py
```

**特点**:
- 快速功能验证（无GUI）
- 详细的测试报告
- 错误诊断信息

## 🚀 快速开始

### 1. 环境要求
- CuRobo已正确安装
- PyBullet（用于可视化演示）
- CUDA支持的GPU（推荐）

### 2. 运行演示
```bash
# 运行完整可视化演示
python pick_and_place_demo.py

# 或者先运行功能测试
python test_pick_and_place.py
```

### 3. 预期结果
- ✅ 机器人成功抓取红色立方体
- ✅ 携带物体避开蓝色/绿色障碍物
- ✅ 在目标位置放置立方体
- ✅ 安全返回起始位置

## 🧠 技术原理

### attach_objects_to_robot()
```python
success = motion_gen.attach_objects_to_robot(
    joint_state,                              # 当前机器人状态
    ["target_cube"],                          # 要附加的物体名称
    surface_sphere_radius=0.01,               # 表面球体半径
    link_name="attached_cube",                # 附加链接名称
    remove_obstacles_from_world_config=True   # 从障碍物中移除
)
```

**工作原理**:
1. 将指定物体从世界障碍物列表中移除
2. 在物体表面生成碰撞检测球体
3. 将碰撞球体附加到机器人的指定链接
4. 更新机器人的碰撞模型

### detach_object_from_robot()
```python
motion_gen.detach_object_from_robot("attached_cube")
```

**工作原理**:
1. 从机器人模型中移除指定的附加物体
2. 恢复机器人原始的碰撞模型
3. 物体不再参与运动规划的碰撞检测

## 🎨 自定义配置

### 修改物体属性
在 `create_pick_and_place_world()` 函数中修改：
```python
"target_cube": {
    "dims": [0.05, 0.05, 0.05],           # 立方体尺寸
    "pose": [0.5, 0.2, 0.025, 1, 0, 0, 0.0]  # 位置和姿态
}
```

### 调整演示参数
- 播放速度: 修改 `playback_speed` 参数
- 规划尝试次数: 修改 `max_attempts` 参数
- 碰撞检测精度: 修改 `surface_sphere_radius` 参数

## 🔧 故障排除

### 常见问题

**1. IK求解失败**
- 问题: `MotionGenStatus.IK_FAIL`
- 解决: 调整目标位置到机器人工作空间内
- 检查: 目标姿态是否可达

**2. 碰撞检测失败**
- 问题: `INVALID_START_STATE_WORLD_COLLISION`
- 解决: 增大物体间距，调整物体位置
- 检查: 附加物体是否与环境冲突

**3. 附加物体失败**
- 问题: `attach_objects_to_robot()` 返回 False
- 解决: 检查物体名称是否正确
- 检查: 机器人状态是否有效

### 调试技巧
1. 运行 `test_pick_and_place.py` 进行基本功能验证
2. 检查世界模型中的物体数量变化
3. 观察PyBullet可视化中的碰撞情况

## 📈 扩展功能

### 支持更多物体类型
- 球体: 修改世界配置添加 `sphere` 类型
- 胶囊体: 添加 `capsule` 类型物体
- 网格: 使用 `mesh` 类型加载复杂几何体

### 多物体抓取
```python
# 同时附加多个物体
motion_gen.attach_objects_to_robot(
    joint_state,
    ["cube1", "cube2", "sphere1"],  # 多个物体
    link_name="multi_attached"
)
```

### 自定义抓取策略
- 实现不同的抓取姿态
- 添加抓取约束
- 集成力传感器反馈

## 🎉 总结

这个Pick and Place演示完整展示了CuRobo在机器人抓取任务中的强大功能：

**核心价值**:
- ✓ 简化了复杂的抓取规划流程
- ✓ 自动化的碰撞检测和避障
- ✓ 高效的运动规划算法
- ✓ 灵活的物体附加/分离机制

**应用场景**:
- 工业装配线
- 仓储物流
- 服务机器人
- 研究和教育

**学习价值**:
- 理解现代机器人运动规划
- 掌握CuRobo API使用
- 学习PyBullet可视化技术
- 体验完整的机器人应用开发流程

---

🤖 **开始您的Pick and Place之旅吧！** 