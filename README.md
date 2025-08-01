# LeRobot: LeRobot + Isaac

🤖 **LeRobot** 集成了LeRobot和Isaac Lab，提供从数据收集、策略训练到部署的Isaac环境中的完整解决方案。

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


## 快速开始

### 安装

1. **克隆仓库**
```bash
git clone https://github.com/Elycyx/IeRobot.git
cd ierobot
```

2. **安装IsaacLab**
```bash
git clone https://github.com/Elycyx/IsaacLab.git
```
然后按照IsaacLab官方的[安装教程](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html)进行安装。

3. **安装依赖**
```bash
cd ierobot
pip install -e .

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

## 许可证

本项目基于 [LeRobot](https://github.com/huggingface/lerobot) 和 [Isaac Lab](https://github.com/isaac-sim/IsaacLab) 项目，参考[LeIsaac](https://github.com/LightwheelAI/leisaac) 项目，遵循 BSD-3-Clause 许可证。

## 贡献

欢迎提交问题报告和功能请求。请确保：
1. 代码符合项目风格指南
2. 添加适当的文档和注释
3. 包含必要的测试用例