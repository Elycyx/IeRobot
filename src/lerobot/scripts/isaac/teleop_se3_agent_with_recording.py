#!/usr/bin/env python3

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run a keyboard teleoperation with Isaac Lab manipulation environments with data recording capabilities."""

"""Launch Isaac Sim Simulator first."""

import multiprocessing
if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

import argparse
import json
import os
import time
import h5py
import numpy as np
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Keyboard teleoperation for Isaac Lab environments with data recording.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--teleop_device", type=str, default="keyboard", help="Device for interacting with environment")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--sensitivity", type=float, default=3.0, help="Sensitivity factor.")
parser.add_argument("--seed", type=int, default=42, help="Seed for the environment.")
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio.",
)
# parser.add_argument(
#     "--enable_cameras",
#     action="store_true",
#     default=False,
#     help="Enable camera rendering and recording.",
# )

# Data recording parameters
parser.add_argument("--record", action="store_true", default=False, help="whether to enable record function")
parser.add_argument("--step_hz", type=int, default=60, help="Environment stepping rate in Hz.")
parser.add_argument("--dataset_file", type=str, default="./datasets/isaac_dataset.hdf5", help="File path to export recorded demos.")
parser.add_argument("--num_demos", type=int, default=0, help="Number of demonstrations to record. Set to 0 for infinite.")
parser.add_argument("--fps", type=int, default=30, help="FPS for video recording.")
parser.add_argument("--auto_success", action="store_true", default=False, help="Automatically detect task success and complete episodes.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)

if args_cli.enable_pinocchio:
    import pinocchio  # noqa: F401
if "handtracking" in args_cli.teleop_device.lower():
    app_launcher_args["xr"] = True

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import omni.log

if "handtracking" in args_cli.teleop_device.lower():
    from isaacsim.xr.openxr import OpenXRSpec

from isaaclab.devices import OpenXRDevice, Se3Gamepad, Se3Keyboard, Se3SpaceMouse

if args_cli.enable_pinocchio:
    from isaaclab.devices.openxr.retargeters.humanoid.fourier.gr1t2_retargeter import GR1T2Retargeter
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401
from isaaclab.devices.openxr.retargeters.manipulator import GripperRetargeter, Se3AbsRetargeter, Se3RelRetargeter
from isaaclab.managers import TerminationTermCfg as DoneTerm

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.utils import parse_env_cfg


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz):
        """
        Args:
            hz (int): frequency to enforce
        """
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.0166, self.sleep_duration)

    def sleep(self, env):
        """Attempt to sleep at the specified rate in hz."""
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration

        # detect time jumping forwards (e.g. loop is too slow)
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


class DataCollector:
    """Data collector for recording demonstrations."""
    
    def __init__(self, dataset_file: str, fps: int = 30):
        self.dataset_file = dataset_file
        self.fps = fps
        self.current_episode_data = []
        self.episode_count = 0
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(dataset_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Initialize HDF5 file
        self.h5_file = None
        self.is_recording = False
        
    def start_recording(self):
        """Start recording a new episode."""
        if self.h5_file is None:
            self.h5_file = h5py.File(self.dataset_file, 'w')
            
        self.current_episode_data = []
        self.is_recording = True
        print(f"开始录制第 {self.episode_count + 1} 个演示...")
        
    def stop_recording(self, success: bool = True):
        """Stop recording and save episode if successful."""
        if not self.is_recording:
            return
            
        self.is_recording = False
        
        if success and len(self.current_episode_data) > 0:
            self._save_episode()
            self.episode_count += 1
            print(f"成功保存第 {self.episode_count} 个演示，包含 {len(self.current_episode_data)} 个步骤")
        else:
            print("录制取消或失败，数据未保存")
            
        self.current_episode_data = []
        
    def record_step(self, obs, actions, rewards, dones, infos):
        """Record a single step."""
        if not self.is_recording:
            return
            
        # Debug: print observation structure for first step
        if len(self.current_episode_data) == 0:
            print("首次录制步骤，观测数据结构：")
            self._print_obs_structure(obs, prefix="  ")
            
        step_data = {
            'observations': self._convert_tensor_dict(obs),
            'actions': self._convert_tensor(actions),
            'rewards': self._convert_tensor(rewards),
            'dones': self._convert_tensor(dones),
            'timestamp': time.time()
        }
        
        self.current_episode_data.append(step_data)
        
    def _print_obs_structure(self, obs, prefix="", max_depth=3, current_depth=0):
        """Print observation data structure for debugging."""
        if current_depth >= max_depth:
            return
            
        if isinstance(obs, dict):
            for key, value in obs.items():
                if isinstance(value, dict):
                    print(f"{prefix}{key}: 字典 ({len(value)} 项)")
                    self._print_obs_structure(value, prefix + "  ", max_depth, current_depth + 1)
                elif torch.is_tensor(value):
                    print(f"{prefix}{key}: 张量 {tuple(value.shape)} {value.dtype}")
                elif isinstance(value, np.ndarray):
                    print(f"{prefix}{key}: 数组 {value.shape} {value.dtype}")
                else:
                    print(f"{prefix}{key}: {type(value)}")
        else:
            print(f"{prefix}非字典类型: {type(obs)}")
        
    def _convert_tensor_dict(self, tensor_dict):
        """Convert tensor dictionary to numpy, handling nested dictionaries."""
        if isinstance(tensor_dict, dict):
            result = {}
            for k, v in tensor_dict.items():
                if isinstance(v, dict):
                    # Handle nested dictionaries (e.g., policy group observations)
                    result[k] = self._convert_tensor_dict(v)
                else:
                    result[k] = self._convert_tensor(v)
            return result
        else:
            return self._convert_tensor(tensor_dict)
            
    def _convert_tensor(self, tensor):
        """Convert tensor to numpy array with proper handling of different data types."""
        if torch.is_tensor(tensor):
            # Convert PyTorch tensor to numpy
            data = tensor.detach().cpu().numpy()
            # Handle different data types
            if data.dtype == np.float64:
                return data.astype(np.float32)  # Reduce precision to save space
            return data
        elif isinstance(tensor, np.ndarray):
            # Handle numpy arrays
            if tensor.dtype == np.float64:
                return tensor.astype(np.float32)
            return tensor
        elif isinstance(tensor, (list, tuple)):
            # Handle sequences
            try:
                return np.array(tensor)
            except (ValueError, TypeError):
                # If can't convert to homogeneous array, keep as list
                return tensor
        elif isinstance(tensor, (int, float, bool)):
            # Handle scalars
            return np.array(tensor)
        else:
            # For other types, try to convert or keep as-is
            try:
                return np.array(tensor)
            except (ValueError, TypeError):
                return tensor
            
    def _save_episode(self):
        """Save current episode to HDF5 file."""
        episode_group = self.h5_file.create_group(f'episode_{self.episode_count:06d}')
        
        # Stack all step data
        all_observations = {}
        all_actions = []
        all_rewards = []
        all_dones = []
        all_timestamps = []
        
        for step_data in self.current_episode_data:
            # Handle observations
            for key, value in step_data['observations'].items():
                if key not in all_observations:
                    all_observations[key] = []
                all_observations[key].append(value)
                
            all_actions.append(step_data['actions'])
            all_rewards.append(step_data['rewards'])
            all_dones.append(step_data['dones'])
            all_timestamps.append(step_data['timestamp'])
            
        # Save to HDF5 with proper data handling
        obs_group = episode_group.create_group('observations')
        self._save_observations_recursive(obs_group, all_observations)
            
        # Save other data
        try:
            episode_group.create_dataset('actions', data=np.array(all_actions))
        except Exception as e:
            print(f"警告：无法保存动作数据: {e}")
            
        try:
            episode_group.create_dataset('rewards', data=np.array(all_rewards))
        except Exception as e:
            print(f"警告：无法保存奖励数据: {e}")
            
        try:
            episode_group.create_dataset('dones', data=np.array(all_dones))
        except Exception as e:
            print(f"警告：无法保存完成标志数据: {e}")
            
        try:
            episode_group.create_dataset('timestamps', data=np.array(all_timestamps))
        except Exception as e:
            print(f"警告：无法保存时间戳数据: {e}")
        
        # Add metadata
        episode_group.attrs['episode_length'] = len(self.current_episode_data)
        episode_group.attrs['fps'] = self.fps
        episode_group.attrs['task'] = args_cli.task
        episode_group.attrs['timestamp'] = datetime.now().isoformat()
        
        self.h5_file.flush()
        
    def _save_observations_recursive(self, group, observations_dict):
        """Recursively save observations to HDF5, handling nested dictionaries."""
        for key, values in observations_dict.items():
            try:
                if len(values) == 0:
                    continue
                    
                # Check if the values contain nested dictionaries
                first_item = values[0]
                if isinstance(first_item, dict):
                    # Create a subgroup for nested observations
                    sub_group = group.create_group(key)
                    sub_group.attrs['data_type'] = 'nested_observations'
                    
                    # Reorganize nested data: from list of dicts to dict of lists
                    nested_obs = {}
                    for step_data in values:
                        for sub_key, sub_value in step_data.items():
                            if sub_key not in nested_obs:
                                nested_obs[sub_key] = []
                            nested_obs[sub_key].append(sub_value)
                    
                    # Recursively save nested observations
                    self._save_observations_recursive(sub_group, nested_obs)
                else:
                    # Handle regular data
                    self._save_single_observation(group, key, values)
                    
            except Exception as e:
                print(f"警告：无法保存观测数据 '{key}': {e}")
                continue
                
    def _save_single_observation(self, group, key, values):
        """Save a single observation (non-nested) to HDF5."""
        try:
            # Try to create a homogeneous array
            data_array = np.array(values)
            
            # Check if the array has object dtype (unsupported by HDF5)
            if data_array.dtype == np.object_:
                # Handle complex data types
                if len(values) > 0:
                    first_item = values[0]
                    if isinstance(first_item, (dict, list)):
                        # Serialize complex data as strings
                        serialized_data = [json.dumps(item, default=str) for item in values]
                        group.create_dataset(key, data=serialized_data, dtype=h5py.string_dtype())
                        group[key].attrs['data_type'] = 'serialized'
                    else:
                        print(f"警告：跳过无法保存的观测数据 '{key}'，数据类型：{type(first_item)}")
                        return
            else:
                # Normal numeric data
                group.create_dataset(key, data=data_array)
                group[key].attrs['data_type'] = 'numeric'
                
        except (ValueError, TypeError) as e:
            print(f"警告：无法保存观测数据 '{key}': {e}")
            # Try to save individual items with their shapes
            try:
                # Create a group for this observation key
                key_group = group.create_group(key)
                for i, item in enumerate(values):
                    item_array = np.array(item)
                    if item_array.dtype != np.object_:
                        key_group.create_dataset(f'step_{i:06d}', data=item_array)
                key_group.attrs['data_type'] = 'variable_shape'
            except Exception as e2:
                print(f"警告：完全无法保存观测数据 '{key}': {e2}")
                return
        
    def close(self):
        """Close the data collector."""
        if self.h5_file is not None:
            # Save dataset metadata
            self.h5_file.attrs['total_episodes'] = self.episode_count
            self.h5_file.attrs['task'] = args_cli.task
            self.h5_file.attrs['created_at'] = datetime.now().isoformat()
            
            self.h5_file.close()
            self.h5_file = None
            
        print(f"数据收集完成。总共录制了 {self.episode_count} 个演示，保存到 {self.dataset_file}")
        
        # Verify the saved data
        if self.episode_count > 0:
            self._verify_saved_data()
            
    def _verify_saved_data(self):
        """Verify that the saved data can be read correctly."""
        try:
            with h5py.File(self.dataset_file, 'r') as f:
                print(f"验证数据文件 {self.dataset_file}:")
                print(f"  - 总回合数: {f.attrs.get('total_episodes', 'Unknown')}")
                print(f"  - 任务: {f.attrs.get('task', 'Unknown')}")
                
                if self.episode_count > 0:
                    # Check the first episode
                    first_ep = f[f'episode_{0:06d}']
                    print(f"  - 第一个回合长度: {first_ep.attrs.get('episode_length', 'Unknown')}")
                    
                    if 'observations' in first_ep:
                        obs_group = first_ep['observations']
                        obs_keys = list(obs_group.keys())
                        print(f"  - 观测数据类型: {obs_keys}")
                        
                        # Show data types recursively
                        self._verify_observations_recursive(obs_group, prefix="    ")
                    
                print("数据验证完成，文件格式正确！")
                
        except Exception as e:
            print(f"警告：数据验证失败: {e}")
            print("这可能表示数据文件存在问题，请检查保存的数据")
            
    def _verify_observations_recursive(self, group, prefix="", max_items=3):
        """Recursively verify and display observation structure."""
        items = list(group.keys())[:max_items]
        for key in items:
            item = group[key]
            if hasattr(item, 'attrs') and 'data_type' in item.attrs:
                data_type = item.attrs['data_type']
                if isinstance(item, h5py.Dataset):
                    shape = item.shape
                    dtype = item.dtype
                    print(f"{prefix}- {key}: 类型={data_type}, 形状={shape}, 数据类型={dtype}")
                elif isinstance(item, h5py.Group):
                    if data_type == 'nested_observations':
                        print(f"{prefix}- {key}: 嵌套观测组")
                        # Recursively show nested observations
                        self._verify_observations_recursive(item, prefix + "  ", max_items=2)
                    else:
                        print(f"{prefix}- {key}: 类型={data_type} (分组数据)")
            else:
                if isinstance(item, h5py.Group):
                    print(f"{prefix}- {key}: 观测组 ({len(item.keys())} 项)")
                else:
                    print(f"{prefix}- {key}: 数据集")


def pre_process_actions(
    teleop_data: tuple[np.ndarray, bool] | list[tuple[np.ndarray, np.ndarray, np.ndarray]], num_envs: int, device: str
) -> torch.Tensor:
    """Convert teleop data to the format expected by the environment action space.

    Args:
        teleop_data: Data from the teleoperation device.
        num_envs: Number of environments.
        device: Device to create tensors on.

    Returns:
        Processed actions as a tensor.
    """
    # compute actions based on environment
    if "Reach" in args_cli.task:
        delta_pose, gripper_command = teleop_data
        # convert to torch
        delta_pose = torch.tensor(delta_pose, dtype=torch.float, device=device).repeat(num_envs, 1)
        # note: reach is the only one that uses a different action space
        # compute actions
        return delta_pose
    elif "PickPlace-GR1T2" in args_cli.task:
        (left_wrist_pose, right_wrist_pose, hand_joints) = teleop_data[0]
        # Reconstruct actions_arms tensor with converted positions and rotations
        actions = torch.tensor(
            np.concatenate([
                left_wrist_pose,  # left ee pose
                right_wrist_pose,  # right ee pose
                hand_joints,  # hand joint angles
            ]),
            device=device,
            dtype=torch.float32,
        ).unsqueeze(0)
        # Concatenate arm poses and hand joint angles
        return actions
    else:
        # resolve gripper command
        delta_pose, gripper_command = teleop_data
        # convert to torch
        delta_pose = torch.tensor(delta_pose, dtype=torch.float, device=device).repeat(num_envs, 1)
        gripper_vel = torch.zeros((delta_pose.shape[0], 1), dtype=torch.float, device=device)
        gripper_vel[:] = -1 if gripper_command else 1
        # compute actions
        return torch.concat([delta_pose, gripper_vel], dim=1)


def main():
    """Running keyboard teleoperation with Isaac Lab manipulation environment with data recording."""
    
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.env_name = args_cli.task
    env_cfg.seed = args_cli.seed
    
    # modify configuration
    env_cfg.terminations.time_out = None
    
    
    if "Lift" in args_cli.task:
        # set the resampling time range to large number to avoid resampling
        env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
        # add termination condition for reaching the goal otherwise the environment won't reset
        env_cfg.terminations.object_reached_goal = DoneTerm(func=mdp.object_reached_goal)
        
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    
    # check environment name (for reach , we don't allow the gripper)
    if "Reach" in args_cli.task:
        omni.log.warn(
            f"The environment '{args_cli.task}' does not support gripper control. The device command will be ignored."
        )

    # Initialize data collector
    data_collector = None
    if args_cli.record:
        data_collector = DataCollector(args_cli.dataset_file, args_cli.fps)

    # Flags for controlling teleoperation flow
    should_reset_recording_instance = False
    should_mark_success = False
    teleoperation_active = True
    recording_active = False

    # Callback handlers
    def reset_recording_instance():
        """Reset the environment to its initial state."""
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True

    def mark_success():
        """Mark current episode as successful and save recording."""
        nonlocal should_mark_success
        should_mark_success = True

    def start_teleoperation():
        """Activate teleoperation control of the robot."""
        nonlocal teleoperation_active
        teleoperation_active = True

    def stop_teleoperation():
        """Deactivate teleoperation control of the robot."""
        nonlocal teleoperation_active
        teleoperation_active = False

    def toggle_recording():
        """Toggle recording on/off."""
        nonlocal recording_active
        if args_cli.record:
            recording_active = not recording_active
            if recording_active:
                data_collector.start_recording()
                print("开始录制数据...")
            else:
                data_collector.stop_recording(success=False)
                print("停止录制数据")

    # create controller
    if args_cli.teleop_device.lower() == "keyboard":
        teleop_interface = Se3Keyboard(
            pos_sensitivity=0.05 * args_cli.sensitivity, rot_sensitivity=0.05 * args_cli.sensitivity
        )
    elif args_cli.teleop_device.lower() == "spacemouse":
        teleop_interface = Se3SpaceMouse(
            pos_sensitivity=0.05 * args_cli.sensitivity, rot_sensitivity=0.05 * args_cli.sensitivity
        )
    elif args_cli.teleop_device.lower() == "gamepad":
        teleop_interface = Se3Gamepad(
            pos_sensitivity=0.1 * args_cli.sensitivity, rot_sensitivity=0.1 * args_cli.sensitivity
        )
    elif "dualhandtracking_abs" in args_cli.teleop_device.lower() and "GR1T2" in args_cli.task:
        # Create GR1T2 retargeter with desired configuration
        gr1t2_retargeter = GR1T2Retargeter(
            enable_visualization=True,
            num_open_xr_hand_joints=2 * (int(OpenXRSpec.HandJointEXT.XR_HAND_JOINT_LITTLE_TIP_EXT) + 1),
            device=env.unwrapped.device,
            hand_joint_names=env.scene["robot"].data.joint_names[-22:],
        )

        # Create hand tracking device with retargeter
        teleop_interface = OpenXRDevice(
            env_cfg.xr,
            retargeters=[gr1t2_retargeter],
        )
        teleop_interface.add_callback("RESET", reset_recording_instance)
        teleop_interface.add_callback("START", start_teleoperation)
        teleop_interface.add_callback("STOP", stop_teleoperation)

        # Hand tracking needs explicit start gesture to activate
        teleoperation_active = False

    elif "handtracking" in args_cli.teleop_device.lower():
        # Create EE retargeter with desired configuration
        if "_abs" in args_cli.teleop_device.lower():
            retargeter_device = Se3AbsRetargeter(
                bound_hand=OpenXRDevice.TrackingTarget.HAND_RIGHT, zero_out_xy_rotation=True
            )
        else:
            retargeter_device = Se3RelRetargeter(
                bound_hand=OpenXRDevice.TrackingTarget.HAND_RIGHT, zero_out_xy_rotation=True
            )

        grip_retargeter = GripperRetargeter(bound_hand=OpenXRDevice.TrackingTarget.HAND_RIGHT)

        # Create hand tracking device with retargeter (in a list)
        teleop_interface = OpenXRDevice(
            env_cfg.xr,
            retargeters=[retargeter_device, grip_retargeter],
        )
        teleop_interface.add_callback("RESET", reset_recording_instance)
        teleop_interface.add_callback("START", start_teleoperation)
        teleop_interface.add_callback("STOP", stop_teleoperation)

        # Hand tracking needs explicit start gesture to activate
        teleoperation_active = False
    else:
        raise ValueError(
            f"Invalid device interface '{args_cli.teleop_device}'. Supported: 'keyboard', 'spacemouse', 'gamepad',"
            " 'handtracking', 'handtracking_abs'."
        )

    # add teleoperation callbacks for all devices
    teleop_interface.add_callback("R", reset_recording_instance)
    if args_cli.record:
        teleop_interface.add_callback("M", mark_success)  # S for success
        teleop_interface.add_callback("T", toggle_recording)  # T for toggle recording
    
    print(teleop_interface)
    print("\n控制说明:")
    print("  R - 重置环境")
    if args_cli.record:
        print("  M - 标记当前回合为成功并保存录制")
        print("  T - 开始/停止录制")
        if args_cli.auto_success:
            print("  自动完成检测: 启用 - 环境done时自动完成episode")
        else:
            print("  自动完成检测: 禁用 - 需要手动按 S 键标记成功")
    print()

    # Initialize rate limiter
    rate_limiter = RateLimiter(args_cli.step_hz) if args_cli.step_hz > 0 else None

    # reset environment
    env.reset()
    teleop_interface.reset()

    step_count = 0
    current_recorded_demo_count = 0

    # simulate environment
    try:
        while simulation_app.is_running():
            # run everything in inference mode
            with torch.inference_mode():
                # get device command
                teleop_data = teleop_interface.advance()

                # Handle success marking
                if should_mark_success:
                    should_mark_success = False
                    if recording_active and data_collector:
                        data_collector.stop_recording(success=True)
                        current_recorded_demo_count += 1
                        recording_active = False
                        print(f"标记成功！已录制 {current_recorded_demo_count} 个演示")
                        
                        # Check if we've reached the target number of demos
                        if args_cli.num_demos > 0 and current_recorded_demo_count >= args_cli.num_demos:
                            print(f"已完成 {args_cli.num_demos} 个演示录制。退出程序。")
                            break

                # Handle environment reset
                if should_reset_recording_instance:
                    env.reset()
                    step_count = 0
                    should_reset_recording_instance = False
                    
                    # Stop current recording if active
                    if recording_active and data_collector:
                        data_collector.stop_recording(success=False)
                        recording_active = False

                # Only apply teleop commands when active
                if teleoperation_active and teleop_data is not None:
                    # compute actions based on environment
                    actions = pre_process_actions(teleop_data, env.num_envs, env.device)
                    
                    # step environment
                    step_result = env.step(actions)
                    
                    # Handle different gymnasium versions
                    if len(step_result) == 4:
                        # Old gymnasium version: (obs, reward, done, info)
                        obs, rewards, dones, infos = step_result
                    elif len(step_result) == 5:
                        # New gymnasium version: (obs, reward, terminated, truncated, info)
                        obs, rewards, terminated, truncated, infos = step_result
                        dones = terminated | truncated  # Combine terminated and truncated into done
                    else:
                        raise ValueError(f"Unexpected number of return values from env.step(): {len(step_result)}")
                    
                    step_count += 1
                    
                    # Record data if recording is active
                    if recording_active and data_collector:
                        data_collector.record_step(obs, actions, rewards, dones, infos)
                    
                    # Check for automatic task completion (only if enabled)
                    if args_cli.auto_success and recording_active and data_collector:
                        # Simple check: if environment is done, task is completed
                        if torch.any(dones):
                            print("检测到任务完成！")
                            data_collector.stop_recording(success=True)
                            current_recorded_demo_count += 1
                            recording_active = False
                            print(f"自动保存演示！已录制 {current_recorded_demo_count} 个演示")
                            
                            # Check if we've reached the target number of demos
                            if args_cli.num_demos > 0 and current_recorded_demo_count >= args_cli.num_demos:
                                print(f"已完成 {args_cli.num_demos} 个演示录制。退出程序。")
                                break
                            
                            # Auto-reset environment for next demonstration
                            print("自动重置环境以开始新的演示...")
                            env.reset()
                            step_count = 0
                        
                else:
                    env.sim.render()

                # Rate limiting
                if rate_limiter:
                    rate_limiter.sleep(env)

    except KeyboardInterrupt:
        print("\n收到中断信号，正在保存数据...")
        
    finally:
        # Clean up
        if data_collector:
            if recording_active:
                data_collector.stop_recording(success=False)
            data_collector.close()
            
        # close the simulator
        env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()