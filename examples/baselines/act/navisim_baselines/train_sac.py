import os
import matplotlib.pyplot as plt
import torch
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from mani_skill.utils.wrappers import RecordEpisode

# ===========================================
# ===========================================
os.makedirs("result_sac", exist_ok=True)
os.makedirs("logs_sac", exist_ok=True)
os.makedirs("videos_sac", exist_ok=True)

# ===========================================
# ===========================================
def make_env(env_id="PickCube-v1", max_episode_steps=None):
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
        output_dir="videos_sac",     # 
        save_trajectory=False,
        video_fps=20
    )

    env = Monitor(env, "logs_sac")   # logs_sac
    return env

# ===========================================
# 3. 训练 SAC 模型
# ===========================================
def train_model():
    train_env = make_env(env_id="PickCube-v1", max_episode_steps=200)

    model = SAC(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log="logs_sac/sac_tensorboard",  # ✅ logs_sac
        learning_rate=3e-4,
        buffer_size=1000000,
        batch_size=256,
        gamma=0.95,
        tau=0.005,
        ent_coef="auto",
        train_freq=1,
        gradient_steps=1,
        learning_starts=10000,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="logs_sac/",         # ✅ logs_sac
        name_prefix="sac_model"
    )

    model.learn(
        total_timesteps=1000000,
        callback=checkpoint_callback,
        tb_log_name="sac_pickcube_run"
    )

    model.save("logs_sac/sac_maniskill_pickcube_final")  # ✅ logs_sac
    train_env.close()
    return model

# ===========================================
# ===========================================
def test_and_record(model, num_episodes=5):
    test_env = make_env(env_id="PickCube-v1", max_episode_steps=200)

    for ep in range(num_episodes):
        obs, _ = test_env.reset()
        frames = []
        done = False

        while not done:
            frame = test_env.render().cpu().numpy()
            while frame.ndim > 3:
                frame = frame.squeeze(0)
            frames.append(frame)

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated

        gif_path = f"result_sac/episode_{ep}.gif"         # ✅ result_sac
        save_gif(frames, gif_path)
        save_key_frames(frames, ep)

    test_env.close()

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
            save_path = f"result_sac/ep{ep_index}_step{idx}.png"  # ✅ result_sac
            plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
            plt.close()

# ===========================================
# 5. 主函数
# ===========================================
if __name__ == "__main__":
    trained_model = train_model()
    test_and_record(trained_model, num_episodes=5)
    print("训练与测试已完成，生成的 GIF 和关键帧已保存到 result_sac/ 目录中。")
