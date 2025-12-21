"""
Process ManiSkill PickCube demonstrations to NaiviBridger dataset format

Usage:
    python process_maniskill_pickcube.py --demo-path ~/.maniskill/demos/PickCube-v1/motionplanning --output-dir dataset/pickcube_maniskill --num-trajs 100
"""

import os
import h5py
import numpy as np
import pickle
import argparse
from PIL import Image
from pathlib import Path
import json


def convert_maniskill_to_navibridge(demo_h5_path, output_dir, num_trajectories=100, traj_name_prefix="pickcube_traj"):
    """
    Convert ManiSkill PickCube demonstrations to NaiviBridger format

    Args:
        demo_h5_path: Path to the ManiSkill h5 file (e.g., trajectory.rgbd.pd_ee_delta_pos.physx_cpu.h5)
        output_dir: Output directory for processed trajectories
        num_trajectories: Number of trajectories to process
        traj_name_prefix: Prefix for trajectory folder names
    """

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading demonstrations from {demo_h5_path}...")

    # Load the h5 file
    with h5py.File(demo_h5_path, 'r') as f:
        # Check available trajectories
        traj_keys = list(f.keys())
        print(f"Found {len(traj_keys)} trajectories in the h5 file")
        print(f"Processing {min(num_trajectories, len(traj_keys))} trajectories...")

        processed_count = 0
        traj_names = []

        for traj_idx, traj_key in enumerate(traj_keys[:num_trajectories]):
            traj_data = f[traj_key]

            # Create trajectory folder
            traj_name = f"{traj_name_prefix}{traj_idx}"
            traj_dir = os.path.join(output_dir, traj_name)
            os.makedirs(traj_dir, exist_ok=True)
            traj_names.append(traj_name)

            # Extract data
            # Actions: end-effector delta positions
            if 'actions' in traj_data:
                actions = traj_data['actions'][:]
            else:
                print(f"Warning: No actions found in trajectory {traj_key}")
                continue

            # Extract RGB images
            if 'obs' in traj_data and 'sensor_data' in traj_data['obs']:
                sensor_data = traj_data['obs']['sensor_data']

                # Check for RGB camera
                if 'base_camera' in sensor_data and 'rgb' in sensor_data['base_camera']:
                    images = sensor_data['base_camera']['rgb'][:]
                elif 'hand_camera' in sensor_data and 'rgb' in sensor_data['hand_camera']:
                    images = sensor_data['hand_camera']['rgb'][:]
                else:
                    # Try to find any RGB image
                    images = None
                    for cam_key in sensor_data.keys():
                        if 'rgb' in sensor_data[cam_key]:
                            images = sensor_data[cam_key]['rgb'][:]
                            break

                    if images is None:
                        print(f"Warning: No RGB images found in trajectory {traj_key}")
                        continue
            else:
                print(f"Warning: No sensor data found in trajectory {traj_key}")
                continue

            # Extract robot state (end-effector pose)
            if 'obs' in traj_data and 'extra' in traj_data['obs'] and 'tcp_pose' in traj_data['obs']['extra']:
                tcp_pose = traj_data['obs']['extra']['tcp_pose'][:]
            elif 'env_states' in traj_data:
                # Try to extract from env_states
                env_states = traj_data['env_states'][:]
                # Assume first 7 dimensions are TCP pose (position + quaternion)
                tcp_pose = env_states[:, :7] if env_states.shape[1] >= 7 else None
            else:
                tcp_pose = None

            traj_len = len(actions)

            # Save images
            for t in range(traj_len + 1):  # +1 because observations have one more than actions
                if t < len(images):
                    img = images[t]

                    # Convert to RGB if necessary
                    if img.dtype == np.float32 or img.dtype == np.float64:
                        img = (img * 255).astype(np.uint8)

                    # Save image
                    img_path = os.path.join(traj_dir, f"{t}.jpg")
                    Image.fromarray(img).save(img_path)

            # Process trajectory data for NaiviBridger format
            # Extract 2D positions from end-effector positions (use x, y coordinates)
            if tcp_pose is not None:
                positions = tcp_pose[:, :2]  # Use x, y positions
            else:
                # If no TCP pose, use cumulative sum of actions as positions
                positions = np.cumsum(actions[:, :2], axis=0)
                # Prepend zero position
                positions = np.vstack([np.zeros((1, 2)), positions])

            # Extract yaw from quaternion or use zeros
            if tcp_pose is not None and tcp_pose.shape[1] >= 7:
                # Convert quaternion to yaw
                # quaternion format: [x, y, z, w]
                quat = tcp_pose[:, 3:7]
                # Convert to yaw (rotation around z-axis)
                yaw = np.arctan2(2.0 * (quat[:, 3] * quat[:, 2] + quat[:, 0] * quat[:, 1]),
                                1.0 - 2.0 * (quat[:, 1]**2 + quat[:, 2]**2))
            else:
                yaw = np.zeros(len(positions))

            # Create trajectory data dictionary
            traj_data_dict = {
                'position': positions.astype(np.float32),  # [T, 2] - x, y positions
                'yaw': yaw.astype(np.float32),             # [T] - yaw angles
            }

            # Save trajectory data
            traj_data_path = os.path.join(traj_dir, "traj_data.pkl")
            with open(traj_data_path, 'wb') as pkl_file:
                pickle.dump(traj_data_dict, pkl_file)

            processed_count += 1

            if (processed_count) % 10 == 0:
                print(f"Processed {processed_count} trajectories...")

    # Save trajectory names list
    traj_names_path = os.path.join(output_dir, "traj_names.txt")
    with open(traj_names_path, 'w') as txt_file:
        txt_file.write('\n'.join(traj_names))

    print(f"\n✅ Successfully processed {processed_count} trajectories!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📝 Trajectory names saved to: {traj_names_path}")

    return processed_count


def main():
    parser = argparse.ArgumentParser(description="Process ManiSkill PickCube demos to NaiviBridger format")
    parser.add_argument(
        "--demo-path",
        type=str,
        default="~/.maniskill/demos/PickCube-v1/motionplanning",
        help="Path to ManiSkill demo directory"
    )
    parser.add_argument(
        "--h5-file",
        type=str,
        default="trajectory.rgbd.pd_ee_delta_pos.physx_cpu.h5",
        help="Name of the h5 file to use (default: trajectory.rgbd.pd_ee_delta_pos.physx_cpu.h5)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="dataset/pickcube_maniskill",
        help="Output directory for processed trajectories"
    )
    parser.add_argument(
        "--num-trajs",
        type=int,
        default=100,
        help="Number of trajectories to process"
    )
    parser.add_argument(
        "--traj-prefix",
        type=str,
        default="pickcube_traj",
        help="Prefix for trajectory folder names"
    )

    args = parser.parse_args()

    # Expand user path
    demo_path = os.path.expanduser(args.demo_path)
    h5_path = os.path.join(demo_path, args.h5_file)

    if not os.path.exists(h5_path):
        print(f"❌ Error: h5 file not found at {h5_path}")
        print(f"\nAvailable files in {demo_path}:")
        if os.path.exists(demo_path):
            for f in os.listdir(demo_path):
                if f.endswith('.h5'):
                    print(f"  - {f}")
        else:
            print(f"  Directory does not exist: {demo_path}")
        return

    convert_maniskill_to_navibridge(
        demo_h5_path=h5_path,
        output_dir=args.output_dir,
        num_trajectories=args.num_trajs,
        traj_name_prefix=args.traj_prefix
    )


if __name__ == "__main__":
    main()
