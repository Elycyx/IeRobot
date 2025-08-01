#!/usr/bin/env python3

"""
Convert Isaac Lab teleoperation data to LeRobot dataset format.

This script converts HDF5 files recorded with teleop_se3_agent_with_recording.py
to the LeRobot dataset format for training robot policies.

NOTE: Please use the lerobot environment for this conversion.
"""

import os
import h5py
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Feature definition for Franka Panda arm in Isaac Lab
FRANKA_FEATURES = {
    "action": {
        "dtype": "float32",
        "shape": (7,),  # 6 delta eef pose + 1 gripper
        "names": [
            "x",
            "y", 
            "z",
            "roll",
            "pitch",
            "yaw",
            "gripper",
        ]
    },
    "observation.state": {
        "dtype": "float32", 
        "shape": (8,),  # 7 joint positions + 1 gripper state
        "names": [
            "panda_joint1",
            "panda_joint2",
            "panda_joint3", 
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
            "gripper",
        ]
    },
    "observation.images.main_cam": {
        "dtype": "video",
        "shape": [224, 224, 3],
        "names": ["height", "width", "channels"],
        "video_info": {
            "video.height": 224,
            "video.width": 224,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p", 
            "video.is_depth_map": False,
            "video.fps": 30.0,
            "video.channels": 3,
            "has_audio": False,
        },
    },
    "observation.images.wrist_cam": {
        "dtype": "video",
        "shape": [224, 224, 3],
        "names": ["height", "width", "channels"],
        "video_info": {
            "video.height": 224,
            "video.width": 224,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": 30.0,
            "video.channels": 3,
            "has_audio": False,
        },
    }
}

def preprocess_se3_actions(se3_actions: np.ndarray) -> np.ndarray:
    """
    Preprocess SE(3) pose commands and gripper actions.
    
    Keep the original SE(3) pose deltas + gripper commands as actions.
    
    Args:
        se3_actions: SE(3) pose commands + gripper (shape: [T, 7])
        
    Returns:
        Preprocessed SE(3) actions (shape: [T, 7])
    """
    # Ensure float32 type
    se3_actions = se3_actions.astype(np.float32)
    
    # SE(3) actions: [dx, dy, dz, droll, dpitch, dyaw, gripper_cmd]
    # No additional preprocessing needed - keep original delta commands
    
    return se3_actions

def preprocess_joint_positions(joint_pos: np.ndarray) -> np.ndarray:
    """
    Preprocess joint positions for LeRobot format.
    
    Isaac Lab uses radians for joints, which is fine for LeRobot.
    We'll just ensure the data is in the right format.
    """
    # Ensure float32 type
    return joint_pos.astype(np.float32)

def load_episode_data(episode_group: h5py.Group) -> tuple:
    """
    Load data from a single episode group.
    
    Returns:
        tuple: (observations, actions, success) where observations is a dict
    """
    try:
        # Load observations from nested structure
        obs_group = episode_group['observations']
        
        if 'policy' in obs_group:
            # Nested observation structure
            policy_group = obs_group['policy']
            
            observations = {
                'state': np.array(policy_group['state']),
                'main_cam': np.array(policy_group['main_cam']),
                'wrist_cam': np.array(policy_group['wrist_cam']),
            }
        else:
            # Flat observation structure (fallback)
            observations = {}
            for key in obs_group.keys():
                observations[key] = np.array(obs_group[key])
        
        # Load actions
        actions = np.array(episode_group['actions'])
        
        # Check if episode was successful (from metadata)
        success = episode_group.attrs.get('success', True)
        
        return observations, actions, success
        
    except KeyError as e:
        print(f"Missing required data in episode: {e}")
        return None, None, False

def process_episode_data(dataset: LeRobotDataset, task: str, episode_group: h5py.Group, 
                        episode_name: str, skip_frames: int = 0) -> bool:
    """
    Process a single episode and add frames to the dataset.
    
    Args:
        dataset: LeRobot dataset to add frames to
        task: Task description
        episode_group: HDF5 group for this episode
        episode_name: Name of the episode
        skip_frames: Number of initial frames to skip
        
    Returns:
        bool: True if episode was processed successfully
    """
    observations, se3_actions, success = load_episode_data(episode_group)
    
    if observations is None:
        print(f'Episode {episode_name} could not be loaded, skipping')
        return False
    
    if not success:
        print(f'Episode {episode_name} was not successful, skipping')
        return False
    
    # Extract data
    try:
        joint_states = observations['state']  # [T, 8]
        main_cam_images = observations['main_cam']  # [T, 224, 224, 3]
        wrist_cam_images = observations['wrist_cam']  # [T, 224, 224, 3] 
        
    except KeyError as e:
        print(f'Episode {episode_name} missing required observation: {e}')
        return False
    
    # Preprocess data
    joint_states = preprocess_joint_positions(joint_states)
    
    # Keep SE(3) actions as is (no conversion to joint space)
    se3_actions = preprocess_se3_actions(se3_actions)
    
    # Print data shapes for debugging
    print(f'Episode {episode_name} data shapes:')
    print(f'  joint_states: {joint_states.shape}')
    print(f'  se3_actions: {se3_actions.shape}') 
    print(f'  main_cam: {main_cam_images.shape}')
    print(f'  wrist_cam: {wrist_cam_images.shape}')
    
    # Verify data consistency
    T = joint_states.shape[0]
    if not (se3_actions.shape[0] == T and 
            main_cam_images.shape[0] == T and 
            wrist_cam_images.shape[0] == T):
        print(f'Episode {episode_name} has inconsistent data shapes, skipping')
        return False
    
    # Add frames to dataset (skip first few frames)
    total_frames = T
    start_frame = max(skip_frames, 0)
    
    for frame_idx in tqdm(range(start_frame, total_frames), 
                         desc=f'Processing {episode_name}', leave=False):
        # Remove batch dimension if present
        action = se3_actions[frame_idx]
        if action.ndim > 1 and action.shape[0] == 1:
            action = action.squeeze(0)  # Remove batch dimension
            
        state = joint_states[frame_idx]
        if state.ndim > 1 and state.shape[0] == 1:
            state = state.squeeze(0)  # Remove batch dimension
            
        main_cam = main_cam_images[frame_idx]
        if main_cam.ndim > 3 and main_cam.shape[0] == 1:
            main_cam = main_cam.squeeze(0)  # Remove batch dimension
            
        wrist_cam = wrist_cam_images[frame_idx]
        if wrist_cam.ndim > 3 and wrist_cam.shape[0] == 1:
            wrist_cam = wrist_cam.squeeze(0)  # Remove batch dimension
        
        # Debug: print shapes for first frame
        if frame_idx == start_frame:
            print(f'  After squeeze - action: {action.shape}, state: {state.shape}')
            print(f'  After squeeze - main_cam: {main_cam.shape}, wrist_cam: {wrist_cam.shape}')
        
        frame = {
            "action": action,  # SE(3) delta pose + gripper action
            "observation.state": state,  # Current joint state
            "observation.images.main_cam": main_cam,
            "observation.images.wrist_cam": wrist_cam,
        }
        dataset.add_frame(frame=frame, task=task)
    
    return True

def convert_isaac_to_lerobot(
    input_files: list[str],
    output_repo_id: str, 
    task_description: str,
    fps: int = 30,
    robot_type: str = "franka_panda",
    skip_frames: int = 5,
    push_to_hub: bool = False
):
    """
    Convert Isaac Lab HDF5 files to LeRobot dataset format.
    
    Args:
        input_files: List of input HDF5 file paths
        output_repo_id: Repository ID for the output dataset
        task_description: Description of the task
        fps: Frames per second for video encoding
        robot_type: Type of robot (for metadata)
        skip_frames: Number of initial frames to skip per episode
        push_to_hub: Whether to push the dataset to HuggingFace Hub
    """
    print(f"Converting {len(input_files)} Isaac Lab HDF5 files to LeRobot format...")
    print(f"Output repository: {output_repo_id}")
    print(f"Task: {task_description}")
    
    # Create LeRobot dataset
    dataset = LeRobotDataset.create(
        repo_id=output_repo_id,
        fps=fps,
        robot_type=robot_type,
        features=FRANKA_FEATURES,
    )
    
    total_episodes = 0
    successful_episodes = 0
    
    # Process each input file
    for file_idx, input_file in enumerate(input_files):
        print(f"\n[{file_idx+1}/{len(input_files)}] Processing: {input_file}")
        
        if not os.path.exists(input_file):
            print(f"Warning: File {input_file} does not exist, skipping")
            continue
            
        try:
            with h5py.File(input_file, 'r') as f:
                # Get all episode names
                episode_names = [key for key in f.keys() if key.startswith('episode_')]
                episode_names.sort()  # Ensure consistent ordering
                
                print(f"Found {len(episode_names)} episodes: {episode_names[:5]}{'...' if len(episode_names) > 5 else ''}")
                
                # Process each episode
                for episode_name in tqdm(episode_names, desc="Processing episodes"):
                    episode_group = f[episode_name]
                    total_episodes += 1
                    
                    success = process_episode_data(
                        dataset=dataset,
                        task=task_description, 
                        episode_group=episode_group,
                        episode_name=episode_name,
                        skip_frames=skip_frames
                    )
                    
                    if success:
                        successful_episodes += 1
                        dataset.save_episode()
                        print(f"✓ Saved episode {successful_episodes}: {episode_name}")
                    else:
                        print(f"✗ Skipped episode: {episode_name}")
                        
        except Exception as e:
            print(f"Error processing file {input_file}: {e}")
            continue
    
    print(f"\n=== Conversion Summary ===")
    print(f"Total episodes processed: {total_episodes}")
    print(f"Successful episodes: {successful_episodes}")
    print(f"Conversion rate: {successful_episodes/total_episodes*100:.1f}%" if total_episodes > 0 else "No episodes processed")
    
    if successful_episodes > 0:
        print(f"\nDataset created with {successful_episodes} episodes")
        
        if push_to_hub:
            print("Pushing dataset to HuggingFace Hub...")
            dataset.push_to_hub()
            print("✓ Dataset pushed to Hub successfully!")
        else:
            print("Dataset saved locally. Use --push_to_hub to upload to HuggingFace Hub.")
    else:
        print("No episodes were successfully converted!")

def main():
    parser = argparse.ArgumentParser(description="Convert Isaac Lab data to LeRobot format")
    
    # Input/Output arguments
    parser.add_argument("--input_files", nargs="+", required=True,
                       help="Input HDF5 files to convert")
    parser.add_argument("--output_repo_id", required=True,
                       help="Output repository ID (e.g., 'username/dataset-name')")
    parser.add_argument("--task", required=True,
                       help="Task description (e.g., 'Lift cube to 20cm height')")
    
    # Processing arguments
    parser.add_argument("--fps", type=int, default=30,
                       help="Frames per second for video encoding (default: 30)")
    parser.add_argument("--robot_type", default="franka_panda",
                       help="Robot type identifier (default: franka_panda)")
    parser.add_argument("--skip_frames", type=int, default=0,
                       help="Number of initial frames to skip per episode (default: 5)")
    
    # Output arguments
    parser.add_argument("--push_to_hub", action="store_true",
                       help="Push dataset to HuggingFace Hub after conversion")
    
    args = parser.parse_args()
    
    # Validate input files
    for input_file in args.input_files:
        if not os.path.exists(input_file):
            print(f"Error: Input file does not exist: {input_file}")
            exit(1)
    
    # Run conversion
    convert_isaac_to_lerobot(
        input_files=args.input_files,
        output_repo_id=args.output_repo_id,
        task_description=args.task,
        fps=args.fps,
        robot_type=args.robot_type,
        skip_frames=args.skip_frames,
        push_to_hub=args.push_to_hub
    )

if __name__ == "__main__":
    main()