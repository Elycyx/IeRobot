# 🚀 IeRobot: LeRobot + Isaac

🤖 **IeRobot** 集成了LeRobot和Isaac Lab，提供从数据收集、策略训练到部署的Isaac环境中的完整解决方案。

https://github.com/user-attachments/assets/a6e338a7-ccce-4278-92e5-08bb4c253ea9

## 📝 更新日志

- v0.1.0: 开源初始版本，目前功能较为简陋。
(当前只测试了`Isaac-Lift-Cube-Franka-IK-Rel-visumotor-v0`任务、`keyboard`遥操作、ACT, DP, SmolVLA三种policy)


## 🚦 快速开始

### 🔧 安装

1. **克隆仓库**
```bash
git clone https://github.com/Elycyx/IeRobot.git
cd ierobot
```

2. **安装IsaacLab**
```bash
cd ierobot
git clone https://github.com/Elycyx/IsaacLab.git
```
然后按照IsaacLab官方的[安装教程](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html)进行安装。

3. **安装依赖**
```bash
cd ierobot
pip install -e .

```

## 📖 使用方法

### 1. 基本遥操作（无录制）
```bash
# 使用键盘控制 Lift 任务（带相机）
python src/lerobot/scripts/isaac/teleop_se3_agent_with_recording.py \
    --task Isaac-Lift-Cube-Franka-IK-Rel-visumotor-v0 \
    --teleop_device keyboard \
    --enable_cameras

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


# 自动完成检测模式 - 环境done时自动完成episode
python src/lerobot/scripts/isaac/teleop_se3_agent_with_recording.py \
    --task Isaac-Lift-Cube-Franka-IK-Rel-visumotor-v0 \
    --teleop_device keyboard \
    --record \
    --auto_success \
    --num_demos 5 \
    --dataset_file ./datasets/auto_demos.hdf5
```


## 🎯 控制说明

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



## 🔄 数据转换

录制完成后，您可以使用 `convert_isaac_to_lerobot.py` 将Isaac Lab数据转换为LeRobot格式，用于策略训练。

### 转换脚本使用

```bash
# 基本转换
python src/lerobot/scripts/isaac/convert_isaac_to_lerobot.py \
    --input_files ./datasets/lift_demos.hdf5 \
    --output_repo_id "username/franka-lift-dataset" \
    --task "Lift cube to 20cm height"

# 多文件转换
python src/lerobot/scripts/isaac/convert_isaac_to_lerobot.py \
    --input_files ./datasets/demo1.hdf5 ./datasets/demo2.hdf5 \
    --output_repo_id "username/franka-lift-dataset" \
    --task "Lift cube to 20cm height" \
    --fps 30 \
    --skip_frames 5

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

## 🏋️‍♂️ 策略训练
使用lerobot训练脚本对policy进行训练。
```bash
python src/lerobot/scripts/train.py \
  --dataset.repo_id=${HF_USER}/franka-lift-dataset \
  --policy.type=act \    # 选择任意lerobot支持的policy
  --output_dir=outputs/train/act_franka-lift-dataset \
  --job_name=act_franka-lift-dataset \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id=${HF_USER}/my_policy
```


## 📊 策略评估

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

## 🔧 扩展和定制

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

## 📄 许可证

本项目基于 [LeRobot](https://github.com/huggingface/lerobot) 和 [Isaac Lab](https://github.com/isaac-sim/IsaacLab) 项目，参考[LeIsaac](https://github.com/LightwheelAI/leisaac) 项目，遵循 BSD-3-Clause 许可证。
