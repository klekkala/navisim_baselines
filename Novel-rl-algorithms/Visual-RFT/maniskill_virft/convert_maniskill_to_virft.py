#!/usr/bin/env python3
"""
Convert ManiSkill demonstrations to Visual-RFT format
Reads HDF5 trajectory files and converts them to JSON + images format
"""

import os
import json
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse


def format_action(action):
    """
    Format action array to string for Visual-RFT
    action: [x, y, z, gripper] with values in [-1, 1]
    """
    # Round to 4 decimal places for readability
    action_str = "[" + ", ".join([f"{a:.4f}" for a in action]) + "]"
    return action_str


def convert_trajectory(h5_file, output_dir, num_episodes=None, prompt_template=None):
    """
    Convert ManiSkill HDF5 trajectory to Visual-RFT format

    Args:
        h5_file: Path to HDF5 file
        output_dir: Output directory for images and JSON
        num_episodes: Number of episodes to convert (None = all)
        prompt_template: Custom prompt template
    """
    if prompt_template is None:
        prompt_template = "Given the current observation, predict the next robot action. The action format is [x, y, z, gripper], where x, y, z are position deltas in [-1, 1] and gripper is -1 (open) or 1 (close)."

    # Create output directories
    image_dir = os.path.join(output_dir, 'images')
    os.makedirs(image_dir, exist_ok=True)

    dataset = []

    with h5py.File(h5_file, 'r') as f:
        # Get all trajectory keys
        traj_keys = sorted([k for k in f.keys() if k.startswith('traj_')],
                          key=lambda x: int(x.split('_')[1]))

        if num_episodes is not None:
            traj_keys = traj_keys[:num_episodes]

        print(f"Converting {len(traj_keys)} trajectories from {h5_file}")

        for traj_key in tqdm(traj_keys, desc="Converting trajectories"):
            traj = f[traj_key]

            # Extract actions and observations
            actions = traj['actions'][:]  # Shape: (T, 4)

            # Check if RGB observations exist
            if 'obs' not in traj:
                print(f"Warning: {traj_key} has no observations, skipping...")
                continue

            obs = traj['obs']
            if 'sensor_data' not in obs or 'base_camera' not in obs['sensor_data']:
                print(f"Warning: {traj_key} has no camera data, skipping...")
                continue

            rgb_data = obs['sensor_data']['base_camera']['rgb'][:]  # Shape: (T+1, H, W, 3)

            # Process each timestep
            num_steps = actions.shape[0]
            traj_id = traj_key.split('_')[1]

            for step in range(num_steps):
                # Get current observation (image at step t)
                rgb = rgb_data[step]  # Shape: (H, W, 3)

                # Get action at this timestep
                action = actions[step]  # Shape: (4,)

                # Save image
                img_name = f'traj{traj_id}_step{step}.png'
                img_path = os.path.join(image_dir, img_name)
                Image.fromarray(rgb.astype('uint8')).save(img_path)

                # Format solution
                solution = format_action(action)

                # Add to dataset
                dataset.append({
                    "image": f"images/{img_name}",
                    "prompt": prompt_template,
                    "solution": solution
                })

    # Save JSON dataset
    json_path = os.path.join(output_dir, 'dataset.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Conversion complete!")
    print(f"   Total samples: {len(dataset)}")
    print(f"   Images saved to: {image_dir}")
    print(f"   Dataset saved to: {json_path}")

    return dataset


def main():
    parser = argparse.ArgumentParser(description='Convert ManiSkill demos to Visual-RFT format')
    parser.add_argument('--h5-file', type=str, required=True,
                       help='Path to ManiSkill HDF5 trajectory file')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for converted dataset')
    parser.add_argument('--num-episodes', type=int, default=None,
                       help='Number of episodes to convert (default: all)')
    parser.add_argument('--prompt', type=str, default=None,
                       help='Custom prompt template')

    args = parser.parse_args()

    convert_trajectory(
        h5_file=args.h5_file,
        output_dir=args.output_dir,
        num_episodes=args.num_episodes,
        prompt_template=args.prompt
    )


if __name__ == '__main__':
    main()
