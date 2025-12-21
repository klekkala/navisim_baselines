import os
import matplotlib.pyplot as plt
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from mani_skill.utils.wrappers import RecordEpisode

# ===========================================
# 1. Create directories (if they don’t exist)
# ===========================================
os.makedirs("result", exist_ok=True)  # To save GIFs and key frames during testing
os.makedirs("logs", exist_ok=True)    # To save training logs and model checkpoints
os.makedirs("videos", exist_ok=True)  # RecordEpisode will output videos here

# ===========================================
# 2. Define make_env(): create and wrap ManiSkill environment
# ===========================================
def make_env(env_id="PickCube-v1", max_episode_steps=None):
    """
    Create ManiSkill environment with gym.make, then wrap with Monitor and RecordEpisode.

    Args:
        env_id (str): Environment ID, e.g., "PickCube-v1"
        max_episode_steps (int or None): If not None, wrap with TimeLimit to cap max steps
    Returns:
        env (Gym environment): Environment wrapped with Monitor and RecordEpisode
    """
    # 2.1 Initialize with render_mode="rgb_array" for rendering
    env = gym.make(
        env_id,
        obs_mode="state",                 # Observation mode: 'state', 'rgbd', or 'pointcloud'
        control_mode="pd_ee_delta_pose",  # Control mode
        reward_mode="dense",              # Reward mode
        render_mode="rgb_array"           # Must include this to use env.render()
    )

    # 2.2 Wrap with TimeLimit if max_episode_steps is specified
    if max_episode_steps is not None:
        from gymnasium.wrappers import TimeLimit
        env = TimeLimit(env, max_episode_steps=max_episode_steps)

    # 2.3 Wrap with RecordEpisode for recording videos
    env = RecordEpisode(
        env,
        output_dir="videos",    # Directory for saving videos
        save_trajectory=False,  # Only record video, not trajectory data
        video_fps=20            # Frames per second
    )

    # 2.4 Wrap with Monitor to log reward, episode length, etc.
    env = Monitor(env, "logs")
    return env

# ===========================================
# 3. Train PPO model
# ===========================================
def train_model():
    """
    Train ManiSkill environment using PPO from Stable-Baselines3,
    with checkpoint saving enabled.
    """
    # 3.1 Create training environment with max 200 steps per episode
    train_env = make_env(env_id="PickCube-v1", max_episode_steps=200)

    # 3.2 Initialize PPO agent
    model = PPO(
        "MlpPolicy",                            # MLP policy (for 'state' vector observations)
        train_env,
        verbose=1,                              # Display training logs
        tensorboard_log="logs/ppo_tensorboard", # TensorBoard log directory
        batch_size=256,
        n_steps=1024,                           # Rollout length
        gamma=0.95,                             # Discount factor
        gae_lambda=0.9,                         # GAE parameter
        ent_coef=0.01,                          # Entropy coefficient (encourage exploration)
        learning_rate=3e-4,                     # Learning rate
        n_epochs=10,                            # Number of epochs per update
        clip_range=0.2,                         # PPO clipping range
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # 3.3 Define checkpoint callback: save model every 10k steps
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="logs/",
        name_prefix="ppo_model"
    )

    # 3.4 Start training for 1M steps
    model.learn(
        total_timesteps=1000_000,
        callback=checkpoint_callback,
        tb_log_name="ppo_pickcube_run"  # Tag for TensorBoard
    )

    # 3.5 Save final trained model
    model.save("logs/ppo_maniskill_pickcube_final")

    # Close environment
    train_env.close()
    return model

# ===========================================
# 4. Test the trained model and save results
# ===========================================
def test_and_record(model, num_episodes=5):
    """
    Run the trained model for several episodes, save rendered frames as GIFs,
    and also extract key frames as PNGs.
    """
    # 4.1 Create test environment
    test_env = make_env(env_id="PickCube-v1", max_episode_steps=200)

    for ep in range(num_episodes):
        obs, _ = test_env.reset()
        frames = []
        done = False

        while not done:
            # 4.2 Render current frame
            frame = test_env.render().cpu().numpy()  
            while frame.ndim > 3:
                frame = frame.squeeze(0)

            frames.append(frame)

            # 4.3 Predict action
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated

        # 4.4 Save episode as GIF
        gif_path = f"result/episode_{ep}.gif"
        save_gif(frames, gif_path)

        # 4.5 Save 5 key frames (0, 1/4, 1/2, 3/4, last)
        save_key_frames(frames, ep)

    test_env.close()

def save_gif(frames, filename, duration=50):
    """
    Save multiple frames into a GIF file.
    - frames: list of numpy arrays (H, W, 3)
    - filename: output file path
    - duration: frame duration in ms
    """
    from PIL import Image

    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(
        filename,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )

def save_key_frames(frames, ep_index):
    """
    Save 5 key frames as PNG:
      - First frame
      - Quarter frame
      - Middle frame
      - Three-quarter frame
      - Last frame
    """
    key_indices = [
        0,
        len(frames) // 4,
        len(frames) // 2,
        (len(frames) * 3) // 4,
        len(frames) - 1
    ]
    for idx in key_indices:
        if 0 <= idx < len(frames):
            plt.figure(figsize=(6, 4))
            plt.imshow(frames[idx])
            plt.axis("off")
            plt.title(f"Episode {ep_index} - Step {idx}")
            save_path = f"result/ep{ep_index}_step{idx}.png"
            plt.savefig(
                save_path,
                bbox_inches="tight",
                pad_inches=0.1
            )
            plt.close()

# ===========================================
# 5. Main: train then test
# ===========================================
if __name__ == "__main__":
    # 5.1 Train PPO agent
    trained_model = train_model()

    # 5.2 Test trained model and save results
    test_and_record(trained_model, num_episodes=5)

    print("Training and testing completed. GIFs and key frames saved in 'result/' directory.")