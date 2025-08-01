# Isaac Lab 遥操作数据录制脚本

## 概述

`teleop_se3_agent_with_recording.py` 是一个基于 Isaac Lab 的遥操作脚本，支持多种输入设备并具备数据录制功能。该脚本允许用户通过键盘、SpaceMouse、游戏手柄或手部跟踪设备来控制机器人，同时可以录制演示数据用于机器学习训练。

## 主要功能

### 1. 多设备支持
- **键盘控制**: 使用 WASD 等按键进行 SE(3) 控制
- **SpaceMouse**: 3D 鼠标进行直观的空间控制
- **游戏手柄**: 使用手柄摇杆和按键控制
- **手部跟踪**: 支持 OpenXR 手部跟踪和 GR1T2 人形机器人控制

### 2. 数据录制功能
- **HDF5 格式**: 将演示数据保存为 HDF5 文件格式
- **实时录制**: 在遥操作过程中实时录制观测、动作、奖励等数据
- **回合管理**: 支持标记成功回合并自动保存
- **自动完成检测**: 可选的自动检测环境done并完成episode
- **元数据**: 自动保存时间戳、任务信息、FPS 等元数据

### 3. 控制功能
- **环境重置**: 随时重置环境到初始状态
- **录制控制**: 开始/停止录制，标记成功回合
- **自动完成检测**: 检测环境done信号并自动保存演示
- **频率控制**: 可配置的仿真频率和录制 FPS
- **优雅退出**: 支持键盘中断并安全保存数据

### 4. 数据转换功能
- **LeRobot兼容**: 将Isaac Lab数据转换为LeRobot训练格式
- **智能预处理**: 自动处理SE(3)到关节空间的动作转换
- **多文件支持**: 批量转换多个录制文件
- **HuggingFace集成**: 直接上传到HuggingFace Hub

### 5. 策略评估功能
- **LeRobot策略兼容**: 支持评估LeRobot训练的策略
- **Isaac Lab环境**: 在真实的物理仿真环境中测试
- **自动格式转换**: 智能处理观测和动作数据格式
- **详细指标**: 成功率、奖励、回合长度等评估指标
- **视频录制**: 自动保存评估过程视频
- **批量评估**: 支持多回合并行评估

## 命令行参数

### 基本参数
```bash
--num_envs          # 并行环境数量 (默认: 1)
--teleop_device     # 遥操作设备类型 (默认: "keyboard")
--task              # Isaac Lab 任务名称
--sensitivity       # 控制灵敏度 (默认: 3.0)
--seed              # 随机种子 (默认: 42)
--enable_pinocchio  # 启用 Pinocchio 物理引擎
--enable_cameras    # 启用相机渲染和录制
```

### 录制参数
```bash
--record            # 启用数据录制功能
--step_hz           # 环境步进频率 (默认: 60 Hz)
--dataset_file      # 数据集保存路径 (默认: "./datasets/isaac_dataset.hdf5")
--num_demos         # 录制演示数量，0 表示无限制 (默认: 0)
--fps               # 视频录制 FPS (默认: 30)
--auto_success      # 自动检测任务成功并完成episode
```

## 使用方法

### 1. 基本遥操作（无录制）
```bash
# 使用键盘控制 Lift 任务（带相机）
python src/lerobot/scripts/isaac/teleop_se3_agent_with_recording.py \
    --task Isaac-Lift-Cube-Franka-IK-Rel-visumotor-v0 \
    --teleop_device keyboard \
    --enable_cameras

# 使用 SpaceMouse 控制
python src/lerobot/scripts/isaac/teleop_se3_agent_with_recording.py \
    --task Isaac-Lift-Cube-Franka-v0 \
    --teleop_device spacemouse

# 使用游戏手柄控制
python src/lerobot/scripts/isaac/teleop_se3_agent_with_recording.py \
    --task Isaac-Lift-Cube-Franka-v0 \
    --teleop_device gamepad
```

### 2. 带数据录制的遥操作
```bash
# 录制 10 个演示到指定文件（带相机）
python src/lerobot/scripts/isaac/teleop_se3_agent_with_recording.py \
    --task Isaac-Lift-Cube-Franka-IK-Rel-visumotor-v0 \
    --teleop_device keyboard \
    --enable_cameras \
    --record \
    --num_demos 10 \
    --dataset_file ./datasets/lift_demos.hdf5 \
    --step_hz 30

# 无限录制模式
python src/lerobot/scripts/isaac/teleop_se3_agent_with_recording.py \
    --task Isaac-Lift-Cube-Franka-v0 \
    --teleop_device spacemouse \
    --record \
    --dataset_file ./datasets/lift_demos.hdf5

# 自动完成检测模式 - 环境done时自动完成episode
python src/lerobot/scripts/isaac/teleop_se3_agent_with_recording.py \
    --task Isaac-Lift-Cube-Franka-IK-Rel-visumotor-v0 \
    --teleop_device keyboard \
    --record \
    --auto_success \
    --num_demos 5 \
    --dataset_file ./datasets/auto_demos.hdf5
```

### 3. 手部跟踪控制
```bash
# 单手相对控制
python src/lerobot/scripts/isaac/teleop_se3_agent_with_recording.py \
    --task Isaac-Lift-Cube-Franka-v0 \
    --teleop_device handtracking \
    --record

# 双手绝对控制（GR1T2 任务）
python src/lerobot/scripts/isaac/teleop_se3_agent_with_recording.py \
    --task Isaac-PickPlace-GR1T2-v0 \
    --teleop_device dualhandtracking_abs \
    --enable_pinocchio \
    --record
```

## 控制说明

### 键盘控制
- **R**: 重置环境
- **M**: 标记当前回合为成功并保存录制（仅录制模式）
- **T**: 开始/停止录制（仅录制模式）
- **WASD**: 平移控制
- **QE**: 垂直移动
- **箭头键**: 旋转控制
- **空格键**: 夹爪控制

### 自动完成检测模式
启用 `--auto_success` 参数后，系统会自动检测任务完成并结束episode：

- **环境done信号检测**: 当环境返回done=True时自动完成当前演示
- **手动开始录制**: 仍需手动按T键开始录制
- **自动保存**: 检测到完成时自动保存演示
- **自动重置**: 完成后自动重置环境开始新的演示

### 手部跟踪控制
- **手势识别**: 自动检测开始/停止/重置手势
- **实时跟踪**: 手部位置和姿态直接映射到机器人末端执行器
- **夹爪控制**: 通过手指姿态控制夹爪开合

## 数据格式

### HDF5 文件结构
```
dataset.hdf5
├── episode_000000/
│   ├── observations/
│   │   ├── policy/ (嵌套观测组)
│   │   │   ├── state (机器人状态数据)
│   │   │   ├── actions (上一步动作)
│   │   │   ├── main_cam (主视角相机图像)
│   │   │   └── wrist_cam (手腕相机图像)
│   │   └── other_obs (其他非嵌套观测)
│   ├── actions
│   ├── rewards
│   ├── dones
│   ├── timestamps
│   └── attrs (元数据)
├── episode_000001/
│   └── ...
└── attrs (数据集元数据)
```

### 数据内容和类型
- **observations**: 包含机器人状态、相机图像等所有观测信息
  - **嵌套观测组**：如`policy`组包含多个子观测（state, actions, cameras等）
  - **数值数据**：直接保存为HDF5数组
  - **图像数据**：相机观测保存为图像数组
  - **复杂数据**：序列化为JSON字符串
  - **变长数据**：每个时间步单独保存
  - **数据类型标记**：通过`data_type`属性区分('numeric', 'nested_observations', 'serialized', 'variable_shape')
- **actions**: 执行的动作序列（SE(3) 位姿 + 夹爪命令）
- **rewards**: 每步的奖励值
- **dones**: 回合结束标志
- **timestamps**: 每步的时间戳
- **metadata**: 任务名称、FPS、回合长度等元信息

### 数据兼容性
- **嵌套观测结构**：自动处理Isaac Lab的观测组（ObsGroup）结构
- **递归数据保存**：支持任意层次的嵌套观测字典
- **PyTorch张量转换**：自动处理PyTorch张量到NumPy的转换
- **多种数据类型**：支持不同形状和类型的观测数据
- **错误处理**：对无法直接保存的数据类型提供警告信息
- **存储优化**：自动将float64转换为float32以节省存储空间

## 核心类说明

### RateLimiter 类
```python
class RateLimiter:
    """控制仿真循环频率的工具类"""
    def __init__(self, hz: int)
    def sleep(self, env)
```
- 确保仿真以指定频率运行
- 在等待期间继续渲染画面
- 自动检测和处理时间跳跃

### DataCollector 类
```python
class DataCollector:
    """数据录制和管理类"""
    def start_recording()          # 开始录制新回合
    def stop_recording(success)    # 停止录制并可选保存
    def record_step(obs, actions, rewards, dones, infos)  # 录制单步数据
    def close()                    # 关闭并保存最终数据
```
- 实时收集和缓存演示数据
- 支持成功/失败回合的区别处理
- 自动处理 PyTorch 张量到 NumPy 的转换
- 线程安全的文件操作

## 支持的任务

### 单臂任务
- `Isaac-Lift-Cube-Franka-v0`: Franka 机器人举起立方体
- `Isaac-Reach-Franka-v0`: Franka 机器人到达目标位置
- `Isaac-Stack-Cube-Franka-v0`: 堆叠立方体任务

### 双臂任务
- `Isaac-PickPlace-GR1T2-v0`: GR1T2 人形机器人拾取放置任务
- 需要启用 `--enable_pinocchio` 选项

### 自定义任务
脚本支持任何符合 Isaac Lab 接口的自定义任务。

## 故障排除

### 常见问题

1. **Gymnasium 版本兼容性**
   ```
   ValueError: too many values to unpack (expected 4)
   ```
   - 脚本已自动处理新旧版本的 Gymnasium
   - 新版本返回 5 个值：(obs, reward, terminated, truncated, info)
   - 旧版本返回 4 个值：(obs, reward, done, info)
   - 脚本会自动检测并适配不同版本

2. **HDF5 数据保存问题**
   ```
   TypeError: Object dtype dtype('O') has no native HDF5 equivalent
   警告：跳过无法保存的观测数据 'policy'，数据类型：<class 'numpy.ndarray'>
   ```
   - 脚本已自动处理不同数据类型和嵌套结构
   - **嵌套观测**：自动处理Isaac Lab的ObsGroup结构
   - **复杂数据**：自动序列化为JSON字符串
   - **变长数据**：分别保存每个时间步
   - **递归保存**：支持任意层次的观测嵌套
   - 无法保存的数据会显示警告而不会崩溃

3. **设备连接问题**
   ```bash
   # 检查 SpaceMouse 连接
   ls /dev/input/
   
   # 检查权限
   sudo chmod 666 /dev/input/event*
   ```

2. **录制文件权限**
   ```bash
   # 确保输出目录存在且可写
   mkdir -p ./datasets
   chmod 755 ./datasets
   ```

3. **内存不足**
   - 减少 `--step_hz` 频率
   - 定期保存数据（设置较小的 `--num_demos`）
   - 监控系统内存使用

4. **仿真性能问题**
   - 降低 `--num_envs` 数量
   - 调整渲染设置
   - 使用 GPU 加速

### 调试模式
```bash
# 启用详细日志
export OMNI_LOG_LEVEL=info

# 检查 Isaac Sim 状态
python -c "import omni.isaac.sim; print('Isaac Sim available')"
```

## 数据转换

录制完成后，您可以使用 `convert_isaac_to_lerobot.py` 将Isaac Lab数据转换为LeRobot格式，用于策略训练。

### 转换脚本使用

```bash
# 基本转换
python src/lerobot/scripts/isaac/convert_isaac_to_lerobot.py \
    --input_files ./datasets/lift_demos.hdf5 \
    --output_repo_id "cyx/franka-lift-dataset" \
    --task "Lift cube to 20cm height"

# 多文件转换
python src/lerobot/scripts/isaac/convert_isaac_to_lerobot.py \
    --input_files ./datasets/demo1.hdf5 ./datasets/demo2.hdf5 \
    --output_repo_id "username/franka-lift-dataset" \
    --task "Lift cube to 20cm height" \
    --fps 30 \
    --skip_frames 5

# 转换并上传到HuggingFace Hub
python src/lerobot/scripts/isaac/convert_isaac_to_lerobot.py \
    --input_files ./datasets/lift_demos.hdf5 \
    --output_repo_id "username/franka-lift-dataset" \
    --task "Lift cube to 20cm height" \
    --push_to_hub
```

### 转换参数说明

- `--input_files`: 输入的HDF5文件路径（支持多个文件）
- `--output_repo_id`: 输出数据集的仓库ID
- `--task`: 任务描述
- `--fps`: 视频编码帧率（默认：30）
- `--robot_type`: 机器人类型标识（默认：franka_panda）
- `--skip_frames`: 每个episode跳过的初始帧数（默认：5）
- `--push_to_hub`: 转换完成后推送到HuggingFace Hub

### 转换后的数据格式

转换后的LeRobot数据集包含：

- **动作空间**: 7维（6个SE(3)位姿增量 + 1个夹爪命令）
  - `[delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw, gripper_cmd]`
- **观测空间**: 
  - `observation.state`: 8维机器人状态（7个关节角度 + 夹爪状态）
  - `observation.images.main_cam`: 主视角相机图像 (224×224×3)
  - `observation.images.wrist_cam`: 手腕相机图像 (224×224×3)

### 数据预处理

转换脚本会自动进行以下预处理：

1. **动作保持**: 保持原始的SE(3)位姿增量和夹爪命令
2. **关节状态**: 保持Isaac Lab的关节角度（弧度制）
3. **图像格式**: 保持原始的224×224 RGB格式
4. **数据类型**: 统一转换为float32格式
5. **帧同步**: 确保观测和动作的时间对齐

## 扩展和定制

### 添加新的遥操作设备
1. 继承 `Se3Device` 基类
2. 实现 `advance()` 方法
3. 在 `main()` 函数中添加设备选项

### 自定义数据格式
1. 修改 `DataCollector` 类的 `_save_episode()` 方法
2. 调整观测数据的预处理逻辑
3. 添加额外的元数据字段

### 自定义转换逻辑
1. 修改 `convert_isaac_to_lerobot.py` 中的 `FRANKA_FEATURES` 定义
2. 调整 `preprocess_se3_actions()` 函数来处理不同的动作格式
3. 添加新的观测数据处理

### 集成新的任务
1. 确保任务符合 Isaac Lab ManagerBasedEnv 接口
2. 根据需要修改 `pre_process_actions()` 函数
3. 添加任务特定的终止条件

## 策略评估

录制和训练完成后，您可以使用 `eval_policy_isaac.py` 在Isaac Lab环境中评估训练的策略。

### 评估脚本使用

```bash
# 基本评估
python src/lerobot/scripts/isaac/eval_policy_isaac.py \
    --policy_path "cyx/franka-lift-dataset" \
    --task Isaac-Lift-Cube-Franka-IK-Rel-visumotor-v0 \
    --n_episodes 10 \
    --seed 42

# 带视频录制的评估
python src/lerobot/scripts/isaac/eval_policy_isaac.py \
    --policy_path "./outputs/train/my_policy/checkpoints/final" \
    --task Isaac-Lift-Cube-Franka-IK-Rel-visumotor-v0 \
    --n_episodes 20 \
    --save_videos \
    --max_videos 5 \
    --output_dir ./eval_results

# 自定义评估设置
python src/lerobot/scripts/isaac/eval_policy_isaac.py \
    --policy_path "username/my-policy" \
    --task Isaac-Lift-Cube-Franka-IK-Rel-visumotor-v0 \
    --n_episodes 50 \
    --step_hz 30 \
    --max_episode_length 500 \
    --policy_device cuda \
    --output_dir ./detailed_eval
```

### 评估参数说明

- `--policy_path`: 策略路径（HuggingFace repo或本地checkpoint）
- `--task`: Isaac Lab任务名称
- `--n_episodes`: 评估回合数（默认：10）
- `--save_videos`: 保存评估视频
- `--max_videos`: 最大视频数量（默认：5）
- `--step_hz`: 环境步进频率（默认：60）
- `--max_episode_length`: 最大回合长度
- `--output_dir`: 结果保存目录

### 评估结果

评估完成后会生成：
- `eval_results.json`: 详细的评估指标和每回合结果
- `episode_XXX.mp4`: 评估视频（如果启用）
- 控制台输出包含成功率、平均奖励等关键指标

## 快速开始指南

### 完整工作流程：从录制到训练到评估

```bash
# 1. 录制演示数据
python src/lerobot/scripts/isaac/teleop_se3_agent_with_recording.py \
    --task Isaac-Lift-Cube-Franka-IK-Rel-visumotor-v0 \
    --teleop_device keyboard \
    --record \
    --auto_success \
    --num_demos 10 \
    --dataset_file ./datasets/franka_lift_demos.hdf5

# 2. 转换为LeRobot格式
python src/lerobot/scripts/isaac/convert_isaac_to_lerobot.py \
    --input_files ./datasets/franka_lift_demos.hdf5 \
    --output_repo_id "your-username/franka-lift-dataset" \
    --task "Lift cube to 20cm height" \
    --push_to_hub

# 3. 使用LeRobot训练策略
python lerobot/scripts/train.py \
    dataset_repo_id=your-username/franka-lift-dataset \
    policy=act \
    env=isaac_lab

# 4. 评估训练的策略
python src/lerobot/scripts/isaac/eval_policy_isaac.py \
    --policy_path "your-username/franka-lift-dataset" \
    --task Isaac-Lift-Cube-Franka-IK-Rel-visumotor-v0 \
    --n_episodes 20 \
    --save_videos
```

### 最佳实践建议

1. **录制质量**
   - 每个演示保持动作流畅和一致
   - 确保任务成功完成（使用 `--auto_success` 或手动按M键）
   - 录制10-50个高质量演示比100个低质量演示更有效

2. **数据多样性**
   - 变化起始位置和物体位置
   - 包含不同的成功路径
   - 避免重复完全相同的动作序列

3. **转换设置**
   - 使用 `--skip_frames 5` 跳过不稳定的初始帧
   - 设置合适的 `--fps` 匹配录制频率
   - 为数据集选择描述性的 `--task` 名称

## 许可证

本脚本基于 Isaac Lab 项目，遵循 BSD-3-Clause 许可证。

## 贡献

欢迎提交问题报告和功能请求。请确保：
1. 代码符合项目风格指南
2. 添加适当的文档和注释
3. 包含必要的测试用例