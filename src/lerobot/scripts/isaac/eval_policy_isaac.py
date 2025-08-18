#!/usr/bin/env python3

"""
Evaluate a LeRobot policy on Isaac Lab environments.

This script loads a trained LeRobot policy and evaluates it in Isaac Lab simulation
environments, computing success rates and other metrics.

Usage examples:

```bash
# Evaluate a trained policy from the hub
python src/lerobot/scripts/isaac/eval_policy_isaac.py \
    --policy_path lerobot/diffusion_pusht \
    --task Isaac-Lift-Cube-Franka-IK-Rel-visumotor-v0 \
    --n_episodes 10 \
    --seed 42

# Evaluate a local checkpoint
python src/lerobot/scripts/isaac/eval_policy_isaac.py \
    --policy_path outputs/train/act_isaac_lift/checkpoints/last/pretrained_model \
    --task Isaac-Lift-Cube-Franka-IK-Rel-visumotor-v0 \
    --n_episodes 10 \
    --save_videos \
    --output_dir outputs/eval/act_isaac_lift --enable_cameras
```
"""

import multiprocessing
if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
import gymnasium as gym
from tqdm import trange

from isaaclab.app import AppLauncher

# Add Isaac Lab specific arguments
parser = argparse.ArgumentParser(description="Evaluate LeRobot policy in Isaac Lab environment.")

# Isaac Lab arguments
parser.add_argument("--task", type=str, required=True, help="Isaac Lab task name")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments")
parser.add_argument("--step_hz", type=int, default=30, help="Environment stepping rate in Hz")
parser.add_argument("--seed", type=int, default=42, help="Random seed")

# Policy arguments
parser.add_argument("--policy_path", type=str, required=True, 
                   help="Path to policy checkpoint or HuggingFace repo")
parser.add_argument("--policy_device", type=str, default="cuda", 
                   help="Device for policy inference")

# Evaluation arguments
parser.add_argument("--n_episodes", type=int, default=10, 
                   help="Number of episodes to evaluate")
parser.add_argument("--max_episode_length", type=int, default=250,
                   help="Maximum episode length (defaults to env max)")

# Output arguments
parser.add_argument("--output_dir", type=str, default="./eval_isaac_results",
                   help="Directory to save evaluation results")
parser.add_argument("--save_videos", action="store_true", 
                   help="Save evaluation videos")
parser.add_argument("--max_videos", type=int, default=5,
                   help="Maximum number of videos to save")

# AppLauncher arguments
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher_args = vars(args_cli)
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

# Import after Isaac Sim launch
import isaaclab_tasks  # noqa: F401
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils import parse_env_cfg

# Import LeRobot components
from lerobot.policies.factory import make_policy, get_policy_class
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.configs.policies import PreTrainedConfig
from lerobot.utils.io_utils import write_video
from lerobot.utils.random_utils import set_seed
import json


def get_task_description_for_environment(task_name: str) -> str:
    """
    Get appropriate task description for VLA models based on Isaac Lab environment.
    
    Args:
        task_name: Isaac Lab task name
        
    Returns:
        Task description string
    """
    task_descriptions = {
        "Isaac-Lift-Cube-Franka-IK-Rel-visumotor-v0": "Lift up the cube.",
    }
    
    # Return specific description if available, otherwise a generic one
    if task_name in task_descriptions:
        return task_descriptions[task_name]
    else:
        # Extract the main task from the name
        if "Lift" in task_name:
            return "Pick up the object and lift it"
        elif "Reach" in task_name:
            return "Reach to the target position"
        elif "Push" in task_name:
            return "Push the object to the target"
        elif "Stack" in task_name:
            return "Stack the objects"
        else:
            return "Complete the robotic manipulation task"


def load_policy_from_path(policy_path: str, device: str) -> PreTrainedPolicy:
    """
    直接使用具体策略类的from_pretrained方法加载策略。
    
    Args:
        policy_path: 策略路径，可以是Hugging Face模型名或本地checkpoint路径
        device: 设备（cuda或cpu）
        
    Returns:
        加载的策略实例
    """
    policy_path = Path(policy_path)
    
    try:
        if policy_path.is_dir():
            # 本地checkpoint路径
            config_file = policy_path / "config.json"
            if config_file.exists():
                # 从配置文件加载策略类型
                with open(config_file) as f:
                    config_data = json.load(f)
                policy_type = config_data.get("type", "act")  # 默认为act
                
                # 获取策略类
                policy_cls = get_policy_class(policy_type)
                
                # 直接使用具体策略类的from_pretrained方法
                policy = policy_cls.from_pretrained(str(policy_path))
                
                # 将策略移到指定设备并设置为评估模式
                policy.to(device)
                policy.eval()
                return policy
            else:
                raise FileNotFoundError(f"配置文件 {config_file} 不存在")
        else:
            # Hugging Face模型路径 - 这种情况下我们可能需要额外的逻辑
            # 但现在先假设是本地路径
            raise ValueError(f"路径 {policy_path} 不是一个目录")
            
    except Exception as e:
        raise ValueError(f"无法从 {policy_path} 加载策略: {e}")


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz: int):
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.0166, self.sleep_duration)

    def sleep(self, env):
        """Sleep to maintain target frequency."""
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration

        # Detect time jumping forwards
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


class IsaacLabToLeRobotObsAdapter:
    """Convert Isaac Lab observations to LeRobot format."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
    
    def convert_observations(self, isaac_obs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Convert Isaac Lab observation format to LeRobot format.
        
        Args:
            isaac_obs: Isaac Lab observation dict with 'policy' key containing nested obs
            
        Returns:
            LeRobot format observation dict
        """
        if 'policy' in isaac_obs:
            policy_obs = isaac_obs['policy']
        else:
            policy_obs = isaac_obs
            
        lerobot_obs = {}
        
        for key, value in policy_obs.items():
            # Convert tensor to appropriate device
            if torch.is_tensor(value):
                tensor_value = value.to(self.device)
            else:
                tensor_value = torch.tensor(value, device=self.device)
            
            # Remove batch dimension if present and batch size is 1
            if tensor_value.ndim > 1 and tensor_value.shape[0] == 1:
                tensor_value = tensor_value.squeeze(0)
            
            # Map observation names to LeRobot format and ensure proper dimensions
            if key == 'state':
                # State should be 1D vector, will add batch dim later
                lerobot_obs['observation.state'] = tensor_value
            elif key in ['main_cam', 'wrist_cam']:
                # Handle image format conversion
                image_tensor = self._process_image(tensor_value, key)
                # Images should be (C, H, W), will add batch dim later
                lerobot_obs[f'observation.images.{key}'] = image_tensor
            elif key not in ['actions', 'action']:  # Skip action fields from observations
                # Keep other relevant observations
                lerobot_obs[f'observation.{key}'] = tensor_value
                
        return lerobot_obs
    
    def _process_image(self, image_tensor: torch.Tensor, cam_name: str) -> torch.Tensor:
        """
        Process image tensor to ensure correct format for LeRobot.
        
        Args:
            image_tensor: Input image tensor
            cam_name: Camera name for debugging
            
        Returns:
            Processed image tensor in LeRobot format (C, H, W)
        """
        original_shape = image_tensor.shape
        
        # Debug: print original shape
        # print(f"Processing {cam_name} image, original shape: {original_shape}")
        
        # Isaac Lab images are typically (H, W, C) but we need (C, H, W) for LeRobot
        if len(original_shape) == 3:
            if original_shape[-1] == 3:  # (H, W, C) format
                # Convert to (C, H, W)
                image_tensor = image_tensor.permute(2, 0, 1)
                #print(f"Converted {cam_name} from (H,W,C) to (C,H,W): {image_tensor.shape}")
            # elif original_shape[0] == 3:  # Already (C, H, W) format
            #     #print(f"{cam_name} already in (C,H,W) format: {image_tensor.shape}")
            # else:
                #print(f"Warning: Unexpected image shape for {cam_name}: {original_shape}")
        
        # Ensure image is float32 and normalized to [0, 1]
        if image_tensor.dtype == torch.uint8:
            image_tensor = image_tensor.float() / 255.0
        
        # Ensure correct target size (3, 224, 224)
        expected_shape = (3, 224, 224)
        if image_tensor.shape != expected_shape:
            print(f"Warning: {cam_name} shape {image_tensor.shape} != expected {expected_shape}")
            # Resize if needed (basic implementation)
            if len(image_tensor.shape) == 3 and image_tensor.shape[0] == 3:
                # Use interpolation to resize
                import torch.nn.functional as F
                image_tensor = F.interpolate(
                    image_tensor.unsqueeze(0), 
                    size=(224, 224), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
                print(f"Resized {cam_name} to: {image_tensor.shape}")
        
        return image_tensor


class LeRobotToIsaacLabActionAdapter:
    """Convert LeRobot actions to Isaac Lab format."""
    
    def __init__(self, action_dim: int = 7):
        self.action_dim = action_dim
    
    def convert_actions(self, lerobot_actions: torch.Tensor) -> torch.Tensor:
        """
        Convert LeRobot action format to Isaac Lab format.
        
        Args:
            lerobot_actions: Actions from LeRobot policy [action_dim] or [batch, action_dim]
            
        Returns:
            Isaac Lab format actions [1, action_dim]
        """
        # Ensure tensor has batch dimension
        if lerobot_actions.ndim == 1:
            isaac_actions = lerobot_actions.unsqueeze(0)  # Add batch dimension
        else:
            isaac_actions = lerobot_actions
            
        # Ensure correct action dimension
        if isaac_actions.shape[-1] != self.action_dim:
            print(f"Warning: Expected {self.action_dim} actions, got {isaac_actions.shape[-1]}")
            
        return isaac_actions


def run_isaac_episode(
    env: ManagerBasedRLEnv,
    policy: PreTrainedPolicy,
    obs_adapter: IsaacLabToLeRobotObsAdapter,
    action_adapter: LeRobotToIsaacLabActionAdapter,
    rate_limiter: Optional[RateLimiter] = None,
    max_steps: Optional[int] = None,
    save_video: bool = False,
    seed: Optional[int] = None,
    task_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run a single episode with the policy in Isaac Lab environment.
    
    Returns:
        Dictionary containing episode data and metrics
    """
    # Reset environment
    if seed is not None:
        obs, _ = env.reset(seed=seed)
    else:
        obs, _ = env.reset()
    
    # Reset policy
    policy.reset()
    
    # Episode data
    episode_rewards = []
    episode_actions = []
    episode_observations = []
    episode_frames = []
    
    step_count = 0
    done = False
    total_reward = 0.0
    success = False
    
    # Get max steps from environment if not specified
    if max_steps is None:
        max_steps = getattr(env, '_max_episode_steps', 1000)
    
    while not done and step_count < max_steps:
        # Convert observations to LeRobot format
        lerobot_obs = obs_adapter.convert_observations(obs)
        
        # Debug: print observation shapes
        if step_count == 0:  # Only print for first step
            print("LeRobot observation shapes:")
            for key, value in lerobot_obs.items():
                print(f"  {key}: {value.shape}")
        
        # Ensure all observations have batch dimension
        batch_obs = {}
        for key, value in lerobot_obs.items():
            if value.ndim == 1:  # Add batch dimension for 1D tensors
                batch_obs[key] = value.unsqueeze(0)
            elif value.ndim == 3 and 'images' in key:  # Images should have batch dim
                if value.shape[0] != 1:  # If not already batched
                    batch_obs[key] = value.unsqueeze(0)
                else:
                    batch_obs[key] = value
            else:
                batch_obs[key] = value
        
        # Add task description for VLA models (like SmolVLA)
        if hasattr(policy, 'config') and hasattr(policy.config, 'vlm_model_name'):
            # This is a VLA model that needs task description
            # Use provided task_name or fallback to a generic task  
            current_task_name = task_name or 'Isaac-Lift-Cube-Franka-IK-Rel-visumotor-v0'
            task_description = get_task_description_for_environment(current_task_name)
            batch_obs["task"] = task_description
            
            if step_count == 0:  # Log the task description once per episode
                print(f"VLA Task description: {task_description}")
        
        # Debug: print batch observation shapes
        if step_count == 0:
            print("Batched observation shapes:")
            for key, value in batch_obs.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape}")
                else:
                    print(f"  {key}: {type(value)} - {value}")
        
        # Get action from policy
        with torch.inference_mode():
            action = policy.select_action(batch_obs)
        
        # Convert action to Isaac Lab format
        isaac_action = action_adapter.convert_actions(action)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(isaac_action)
        done = terminated or truncated
        
        # Track data
        episode_rewards.append(reward.cpu().numpy())
        episode_actions.append(action.cpu().numpy())
        episode_observations.append(lerobot_obs)
        total_reward += reward.item()
        
        # Save frame for video if requested
        if save_video:
            try:
                frame = env.render()
                if frame is not None and frame.size > 0:
                    # Ensure frame is uint8 and in correct format (H, W, C)
                    if frame.dtype != np.uint8:
                        frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
                    episode_frames.append(frame)
            except Exception as e:
                if step_count == 0:  # Only warn once per episode
                    print(f"Warning: Failed to capture frame: {e}")
                    print("Video recording may not work properly. Ensure --enable_cameras is set if running headless.")
        
        # Check for success
        if hasattr(info, 'get') and info.get('is_success', False):
            success = True
        elif terminated:  # Assume termination without truncation is success
            success = True
            
        step_count += 1
        
        # Rate limiting
        if rate_limiter:
            rate_limiter.sleep(env)
    
    return {
        'episode_length': step_count,
        'total_reward': total_reward,
        'mean_reward': total_reward / max(step_count, 1),
        'success': success,
        'rewards': episode_rewards,
        'actions': episode_actions,
        'observations': episode_observations,
        'frames': episode_frames if save_video else None
    }


def evaluate_policy_isaac(
    policy_path: str,
    task: str,
    n_episodes: int,
    output_dir: str,
    save_videos: bool = False,
    max_videos: int = 5,
    **kwargs
) -> Dict[str, Any]:
    """
    Evaluate a LeRobot policy on Isaac Lab environment.
    
    Returns:
        Dictionary containing evaluation metrics and results
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Evaluating policy: {policy_path}")
    logger.info(f"Task: {task}")
    logger.info(f"Episodes: {n_episodes}")
    
    # Set random seed
    set_seed(args_cli.seed)
    
    # Create Isaac Lab environment
    env_cfg = parse_env_cfg(task, device=args_cli.policy_device, num_envs=args_cli.num_envs)
    
    # Disable termination timeout for evaluation
    if hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None
        
    # Disable recorders
    env_cfg.recorders = None
    
    # Configure rendering for video saving
    if save_videos:
        # Set render mode to rgb_array to capture frames
        env_cfg.viewer.eye = (7.5, 7.5, 7.5)  # Adjust camera position if needed
        env_cfg.viewer.lookat = (0.0, 0.0, 0.0)
        env_cfg.viewer.resolution = (640, 480)  # Set video resolution
    
    # Create environment
    env: ManagerBasedRLEnv = gym.make(task, cfg=env_cfg, render_mode="rgb_array" if save_videos else None).unwrapped
    logger.info(f"Created environment: {task}")
    
    # Load policy
    try:
        policy = load_policy_from_path(policy_path, device=args_cli.policy_device)
        logger.info(f"Successfully loaded policy from: {policy_path}")
        logger.info(f"Policy type: {policy.__class__.__name__}")
    except Exception as e:
        logger.error(f"Failed to load policy: {e}")
        raise ValueError(f"Could not load policy from {policy_path}: {e}")
    
    policy.eval()
    
    # Create adapters
    obs_adapter = IsaacLabToLeRobotObsAdapter(device=args_cli.policy_device)
    action_adapter = LeRobotToIsaacLabActionAdapter(action_dim=7)  # SE(3) + gripper
    
    # Create rate limiter
    rate_limiter = RateLimiter(args_cli.step_hz) if args_cli.step_hz > 0 else None
    
    # Run evaluation episodes
    episode_results = []
    video_paths = []
    
    logger.info("Starting evaluation...")
    start_time = time.time()
    
    for episode_idx in trange(n_episodes, desc="Evaluating episodes"):
        save_video_this_episode = save_videos and episode_idx < max_videos
        
        episode_result = run_isaac_episode(
            env=env,
            policy=policy,
            obs_adapter=obs_adapter,
            action_adapter=action_adapter,
            rate_limiter=rate_limiter,
            max_steps=args_cli.max_episode_length,
            save_video=save_video_this_episode,
            seed=args_cli.seed + episode_idx,
            task_name=task
        )
        
        episode_results.append(episode_result)
        
        # Save video if recorded
        if save_video_this_episode and episode_result['frames'] and len(episode_result['frames']) > 0:
            video_path = output_path / f"episode_{episode_idx:03d}.mp4"
            try:
                frames_array = np.array(episode_result['frames'])
                logger.info(f"Saving video with {len(episode_result['frames'])} frames, shape: {frames_array.shape}")
                
                write_video(
                    str(video_path),
                    frames_array,
                    fps=30  # Default FPS
                )
                video_paths.append(str(video_path))
                logger.info(f"Successfully saved video: {video_path}")
            except Exception as e:
                logger.warning(f"Failed to save video: {e}")
                logger.warning(f"Frames info: count={len(episode_result['frames']) if episode_result['frames'] else 0}")
                if len(episode_result['frames']) > 0:
                    logger.warning(f"First frame shape: {episode_result['frames'][0].shape if hasattr(episode_result['frames'][0], 'shape') else 'unknown'}")
        elif save_video_this_episode:
            logger.warning(f"No frames captured for episode {episode_idx}, video not saved")
        
        # Log progress
        if (episode_idx + 1) % 5 == 0:
            current_success_rate = np.mean([r['success'] for r in episode_results])
            logger.info(f"Episodes {episode_idx + 1}/{n_episodes}, "
                       f"Success rate: {current_success_rate:.2%}")
    
    total_time = time.time() - start_time
    
    # Compute metrics
    successes = [r['success'] for r in episode_results]
    total_rewards = [r['total_reward'] for r in episode_results]
    episode_lengths = [r['episode_length'] for r in episode_results]
    
    metrics = {
        'success_rate': float(np.mean(successes)),
        'avg_total_reward': float(np.mean(total_rewards)),
        'std_total_reward': float(np.std(total_rewards)),
        'avg_episode_length': float(np.mean(episode_lengths)),
        'std_episode_length': float(np.std(episode_lengths)),
        'evaluation_time_s': total_time,
        'episodes_per_second': n_episodes / total_time,
    }
    
    # Create results dictionary
    results = {
        'policy_path': policy_path,
        'task': task,
        'n_episodes': n_episodes,
        'seed': args_cli.seed,
        'metrics': metrics,
        'per_episode': [
            {
                'episode_idx': i,
                'success': r['success'],
                'total_reward': r['total_reward'],
                'episode_length': r['episode_length'],
            }
            for i, r in enumerate(episode_results)
        ]
    }
    
    if video_paths:
        results['video_paths'] = video_paths
    
    # Save results
    results_file = output_path / "eval_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Evaluation completed!")
    logger.info(f"Success rate: {metrics['success_rate']:.2%}")
    logger.info(f"Average reward: {metrics['avg_total_reward']:.3f}")
    logger.info(f"Results saved to: {results_file}")
    
    # Close environment
    env.close()
    
    return results


def main():
    """Main evaluation function."""
    try:
        results = evaluate_policy_isaac(
            policy_path=args_cli.policy_path,
            task=args_cli.task,
            n_episodes=args_cli.n_episodes,
            output_dir=args_cli.output_dir,
            save_videos=args_cli.save_videos,
            max_videos=args_cli.max_videos,
        )
        
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Policy: {args_cli.policy_path}")
        print(f"Task: {args_cli.task}")
        print(f"Episodes: {args_cli.n_episodes}")
        print(f"Success Rate: {results['metrics']['success_rate']:.2%}")
        print(f"Average Reward: {results['metrics']['avg_total_reward']:.3f}")
        print(f"Average Episode Length: {results['metrics']['avg_episode_length']:.1f}")
        print("="*50)
        
    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        raise
    finally:
        # Clean shutdown
        simulation_app.close()


if __name__ == "__main__":
    main()