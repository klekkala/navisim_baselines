import os
import matplotlib.pyplot as plt
import torch
import gymnasium as gym
from stable_baselines3 import DDPG
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from mani_skill.utils.wrappers import RecordEpisode

# ===========================================
# ===========================================
os.makedirs("result_ddpg", exist_ok=True)
os.makedirs("logs_ddpg", exist_ok=True)
os.makedirs("videos_ddpg", exist_ok=True)

# ===========================================
# ===========================================
def make_env(env_id="PickCube-v1", max_episode_steps=200):
    env = gym.make(
        env_id,
        obs_mode="state",
        control_mode="pd_ee_delta_pose",
        reward_mode="dense",
        render_mode="rgb_array"
    )

    if max_episode_steps is not None:
        from gymnasium.wrappers import TimeLimit
        env = TimeLimit(env, max_episode_steps=max_episode_steps)

    env = RecordEpisode(
        env,
        output_dir="videos_ddpg",
        save_trajectory=False,
        video_fps=20
    )

    env = Monitor(env, "logs_ddpg")
    return env

# ===========================================
# ===========================================
def train_model():
    env = make_env()

    model = DDPG(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="logs_ddpg/ddpg_tensorboard",
        learning_rate=1e-3,
        buffer_size=1000000,
        batch_size=256,
        tau=0.005,
        gamma=0.95,
        train_freq=1,
        gradient_steps=1,
        learning_starts=10000,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="logs_ddpg/",
        name_prefix="ddpg_model"
    )

    model.learn(
        total_timesteps=1000000,
        callback=checkpoint_callback,
        tb_log_name="ddpg_pickcube_run"
    )

    model.save("logs_ddpg/ddpg_maniskill_pickcube_final")
    env.close()
    return model

# ===========================================
# ===========================================
def test_and_record(model, num_episodes=5):
    env = make_env()
    for ep in range(num_episodes):
        obs, _ = env.reset()
        frames = []
        done = False

        while not done:
            frame = env.render().cpu().numpy()
            while frame.ndim > 3:
                frame = frame.squeeze(0)
            frames.append(frame)

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        gif_path = f"result_ddpg/episode_{ep}.gif"
        save_gif(frames, gif_path)
        save_key_frames(frames, ep)

    env.close()

def save_gif(frames, filename, duration=50):
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
            save_path = f"result_ddpg/ep{ep_index}_step{idx}.png"
            plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
            plt.close()

# ===========================================
# ===========================================
if __name__ == "__main__":
    trained_model = train_model()
    test_and_record(trained_model, num_episodes=5)
    print("✅ DDPG训练与测试完成，结果保存在 result_ddpg/")
