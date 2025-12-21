import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from PIL import Image
from torchvision import transforms as T

import gymnasium as gym
import mani_skill.envs

from r3m import load_r3m

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- R3M loading and preprocessing ----
r3m_model = load_r3m("resnet50")
r3m_model.eval().to(device)

transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
])

def encode_rgb(rgb_np):
    if isinstance(rgb_np, torch.Tensor):
        rgb_np = rgb_np.cpu().numpy()
    if rgb_np.dtype != np.uint8:
        rgb_np = (rgb_np * 255).astype(np.uint8)
    img = Image.fromarray(rgb_np)
    t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = r3m_model(t * 255.0)
    return emb

def extract_proprioception(obs):
    """Improved proprioception feature extraction."""
    proprio_features = []

    # ManiSkill standard observation structure
    # 1) agent qpos (joint positions)
    if 'agent' in obs:
        agent_dict = obs['agent']
        if 'qpos' in agent_dict:
            qpos = agent_dict['qpos']
            if isinstance(qpos, torch.Tensor):
                proprio_features.append(qpos.flatten().cpu().numpy())
            else:
                proprio_features.append(np.array(qpos).flatten())

        # 2) agent qvel (joint velocities)
        if 'qvel' in agent_dict:
            qvel = agent_dict['qvel']
            if isinstance(qvel, torch.Tensor):
                proprio_features.append(qvel.flatten().cpu().numpy())
            else:
                proprio_features.append(np.array(qvel).flatten())

    # 3) extra (often contains TCP pose, etc.)
    if 'extra' in obs:
        extra_dict = obs['extra']
        for key in sorted(extra_dict.keys()):  # keep ordering deterministic
            value = extra_dict[key]
            if isinstance(value, torch.Tensor):
                proprio_features.append(value.flatten().cpu().numpy())
            elif isinstance(value, (np.ndarray, list)):
                proprio_features.append(np.array(value).flatten())

    if len(proprio_features) == 0:
        print("Warning: No standard proprioception fields were found.")
        print("Observation keys:", list(obs.keys()))
        # Return a small placeholder vector instead of an empty array
        return np.zeros(8, dtype=np.float32)

    return np.concatenate(proprio_features).astype(np.float32)

# ---- Improved network architectures ----
class ImprovedMultiModalActor(nn.Module):
    def __init__(self, visual_dim, proprio_dim, action_dim, hidden_dim=512):
        super().__init__()
        self.visual_dim = visual_dim
        self.proprio_dim = proprio_dim

        # Visual feature branch (deeper)
        self.visual_net = nn.Sequential(
            nn.Linear(visual_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )

        # Proprioception branch
        self.proprio_net = nn.Sequential(
            nn.Linear(proprio_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )

        # Fusion network
        self.fusion_net = nn.Sequential(
            nn.Linear(256 + 128, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Output heads
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

        self.LOG_STD_MIN = -10
        self.LOG_STD_MAX = 2

        # Initialization
        self.apply(self._init_weights)
        # Small init for output layer
        nn.init.uniform_(self.mean_layer.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.mean_layer.bias, -3e-3, 3e-3)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)

    def forward(self, visual_x, proprio_x):
        visual_feat = self.visual_net(visual_x)
        proprio_feat = self.proprio_net(proprio_x)

        fused = torch.cat([visual_feat, proprio_feat], dim=-1)
        features = self.fusion_net(fused)

        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp()

        return mean, std

    def sample(self, visual_x, proprio_x):
        mean, std = self.forward(visual_x, proprio_x)
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def eval_action(self, visual_x, proprio_x):
        mean, _ = self.forward(visual_x, proprio_x)
        action = torch.tanh(mean)
        return action

class ImprovedMultiModalCritic(nn.Module):
    def __init__(self, visual_dim, proprio_dim, action_dim, hidden_dim=512):
        super().__init__()

        # Visual branch
        self.visual_net = nn.Sequential(
            nn.Linear(visual_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )

        # Proprio branch
        self.proprio_net = nn.Sequential(
            nn.Linear(proprio_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )

        # Q network
        self.q_net = nn.Sequential(
            nn.Linear(256 + 128 + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)

    def forward(self, visual_x, proprio_x, a):
        visual_feat = self.visual_net(visual_x)
        proprio_feat = self.proprio_net(proprio_x)

        fused = torch.cat([visual_feat, proprio_feat, a], dim=-1)
        q_val = self.q_net(fused)

        return q_val

# ---- Replay buffer ----
class MultiModalReplayBuffer:
    def __init__(self, max_size, visual_dim, proprio_dim, action_dim):
        self.max_size = max_size
        self.visual_buf = np.zeros((max_size, visual_dim), dtype=np.float32)
        self.proprio_buf = np.zeros((max_size, proprio_dim), dtype=np.float32)
        self.next_visual_buf = np.zeros((max_size, visual_dim), dtype=np.float32)
        self.next_proprio_buf = np.zeros((max_size, proprio_dim), dtype=np.float32)
        self.act_buf = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rew_buf = np.zeros((max_size, 1), dtype=np.float32)
        self.done_buf = np.zeros((max_size, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def store(self, visual_obs, proprio_obs, act, rew, next_visual_obs, next_proprio_obs, done):
        self.visual_buf[self.ptr] = visual_obs
        self.proprio_buf[self.ptr] = proprio_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.next_visual_buf[self.ptr] = next_visual_obs
        self.next_proprio_buf[self.ptr] = next_proprio_obs
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idxs = np.random.choice(self.size, batch_size, replace=False)
        batch = dict(
            visual_obs=torch.FloatTensor(self.visual_buf[idxs]).to(device),
            proprio_obs=torch.FloatTensor(self.proprio_buf[idxs]).to(device),
            act=torch.FloatTensor(self.act_buf[idxs]).to(device),
            rew=torch.FloatTensor(self.rew_buf[idxs]).to(device),
            next_visual_obs=torch.FloatTensor(self.next_visual_buf[idxs]).to(device),
            next_proprio_obs=torch.FloatTensor(self.next_proprio_buf[idxs]).to(device),
            done=torch.FloatTensor(self.done_buf[idxs]).to(device),
        )
        return batch

# ---- Reward normalization ----
class RewardScaler:
    def __init__(self, gamma=0.99):
        self.returns = []
        self.gamma = gamma
        self.mean = 0.0
        self.std = 1.0

    def update(self, reward):
        """Update return statistics."""
        if len(self.returns) > 0:
            self.returns[-1] = reward + self.gamma * self.returns[-1]
        else:
            self.returns.append(reward)

        if len(self.returns) > 100:
            self.mean = np.mean(self.returns[-100:])
            self.std = np.std(self.returns[-100:]) + 1e-8

    def scale(self, reward):
        """Scale reward."""
        return reward / (self.std + 1e-8)

    def new_episode(self):
        """Start a new episode."""
        self.returns.append(0.0)

# ---- Automatic entropy coefficient tuning ----
class AdaptiveAlpha:
    def __init__(self, target_entropy, initial_log_alpha=0.0, lr=3e-4):
        self.target_entropy = target_entropy
        self.log_alpha = torch.tensor(initial_log_alpha, requires_grad=True, device=device)
        self.optimizer = optim.Adam([self.log_alpha], lr=lr)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def update(self, log_prob):
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.optimizer.zero_grad()
        alpha_loss.backward()
        self.optimizer.step()
        return alpha_loss.item()

# ---- Main training loop ----
def main():
    env = gym.make(
        "PickCube-v1",
        obs_mode="rgbd",
        control_mode="pd_joint_delta_pos",
        render_mode="rgb_array",
        num_envs=1
    )
    action_dim = env.single_action_space.shape[0]

    # Reset once to infer feature dimensions
    obs, _ = env.reset(seed=0)

    # Visual feature dimension
    sensor_data = obs['sensor_data']
    sensor_uid = list(sensor_data.keys())[0]
    rgb = sensor_data[sensor_uid]['rgb']
    if isinstance(rgb, torch.Tensor):
        rgb_np = rgb.cpu().numpy()
    else:
        rgb_np = rgb
    if rgb_np.ndim == 4:
        rgb_np = rgb_np[0]
    emb = encode_rgb(rgb_np)
    visual_dim = emb.shape[-1]

    # Proprioception feature dimension
    proprio_vec = extract_proprioception(obs)
    proprio_dim = proprio_vec.shape[0]

    print("=== Environment Info ===")
    print(f"Action dimension: {action_dim}")
    print(f"Visual feature dimension: {visual_dim}")
    print(f"Proprioception feature dimension: {proprio_dim}")
    print(f"Observation keys: {list(obs.keys())}")

    # Networks
    actor = ImprovedMultiModalActor(visual_dim, proprio_dim, action_dim).to(device)
    critic1 = ImprovedMultiModalCritic(visual_dim, proprio_dim, action_dim).to(device)
    critic2 = ImprovedMultiModalCritic(visual_dim, proprio_dim, action_dim).to(device)
    critic1_target = ImprovedMultiModalCritic(visual_dim, proprio_dim, action_dim).to(device)
    critic2_target = ImprovedMultiModalCritic(visual_dim, proprio_dim, action_dim).to(device)
    critic1_target.load_state_dict(critic1.state_dict())
    critic2_target.load_state_dict(critic2.state_dict())

    # Optimizers (lower learning rates)
    actor_opt = optim.Adam(actor.parameters(), lr=1e-4)
    critic_opt = optim.Adam(list(critic1.parameters()) + list(critic2.parameters()), lr=3e-4)

    # Entropy tuning
    target_entropy = -action_dim * 0.5  # slightly smaller target entropy
    alpha_manager = AdaptiveAlpha(target_entropy, initial_log_alpha=-1.0)

    # Training settings
    gamma = 0.99
    tau = 0.005
    buffer = MultiModalReplayBuffer(
        max_size=200000,
        visual_dim=visual_dim,
        proprio_dim=proprio_dim,
        action_dim=action_dim
    )
    reward_scaler = RewardScaler(gamma=gamma)

    num_episodes = 1000
    max_steps = 200
    batch_size = 256
    start_steps = 10000
    update_after = 2000
    update_freq = 1
    total_steps = 0

    # Stats
    best_reward = -float('inf')
    reward_history = []
    success_history = []

    print("\n=== Start Training ===")
    print(f"Initial exploration steps: {start_steps}")
    print(f"Batch size: {batch_size}")
    print(f"Max steps per episode: {max_steps}\n")

    for ep in range(num_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        ep_steps = 0
        reward_scaler.new_episode()

        for step in range(max_steps):
            total_steps += 1

            # Extract features
            sensor_data = obs['sensor_data']
            sensor_uid = list(sensor_data.keys())[0]
            rgb = sensor_data[sensor_uid]['rgb']
            if isinstance(rgb, torch.Tensor):
                rgb_np = rgb.cpu().numpy()
            else:
                rgb_np = rgb
            if rgb_np.ndim == 4:
                rgb_np = rgb_np[0]
            emb_cur = encode_rgb(rgb_np)
            visual_obs = emb_cur.view(-1, visual_dim)

            proprio_obs_vec = extract_proprioception(obs)
            proprio_obs = torch.FloatTensor(proprio_obs_vec).unsqueeze(0).to(device)

            # Action selection (extra exploration noise)
            if total_steps < start_steps:
                action_np = env.single_action_space.sample()
            else:
                with torch.no_grad():
                    action, _ = actor.sample(visual_obs, proprio_obs)
                    # Add additional exploration noise early on
                    if np.random.rand() < max(0.1, 1.0 - total_steps / 50000):
                        noise = torch.randn_like(action) * 0.1
                        action = torch.clamp(action + noise, -1, 1)
                    action_np = action.cpu().numpy()[0]

            # Step env
            next_obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated

            # Reward processing
            raw_reward = float(reward)
            reward_scaler.update(raw_reward)
            scaled_reward = reward_scaler.scale(raw_reward)

            ep_reward += raw_reward
            ep_steps += 1

            # Next state features
            sensor_data2 = next_obs['sensor_data']
            sensor_uid2 = list(sensor_data2.keys())[0]
            rgb2 = sensor_data2[sensor_uid2]['rgb']
            if isinstance(rgb2, torch.Tensor):
                rgb2_np = rgb2.cpu().numpy()
            else:
                rgb2_np = rgb2
            if rgb2_np.ndim == 4:
                rgb2_np = rgb2_np[0]
            emb_next = encode_rgb(rgb2_np)
            next_visual_obs = emb_next.view(-1, visual_dim)

            next_proprio_obs_vec = extract_proprioception(next_obs)

            # Store transition
            buffer.store(
                visual_obs.detach().cpu().numpy()[0],
                proprio_obs_vec,
                action_np,
                scaled_reward,  # scaled reward
                next_visual_obs.detach().cpu().numpy()[0],
                next_proprio_obs_vec,
                float(done)
            )

            obs = next_obs

            # Updates
            if buffer.size >= batch_size and total_steps >= update_after and total_steps % update_freq == 0:
                batch = buffer.sample(batch_size)

                with torch.no_grad():
                    a_next, logp_next = actor.sample(batch['next_visual_obs'], batch['next_proprio_obs'])
                    q1_next = critic1_target(batch['next_visual_obs'], batch['next_proprio_obs'], a_next)
                    q2_next = critic2_target(batch['next_visual_obs'], batch['next_proprio_obs'], a_next)
                    q_next_min = torch.min(q1_next, q2_next)
                    target_q = batch['rew'] + gamma * (1 - batch['done']) * (q_next_min - alpha_manager.alpha * logp_next)

                # Critic update
                q1 = critic1(batch['visual_obs'], batch['proprio_obs'], batch['act'])
                q2 = critic2(batch['visual_obs'], batch['proprio_obs'], batch['act'])
                critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

                critic_opt.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(list(critic1.parameters()) + list(critic2.parameters()), 0.5)
                critic_opt.step()

                # Actor update
                a_pi, logp_pi = actor.sample(batch['visual_obs'], batch['proprio_obs'])
                q1_pi = critic1(batch['visual_obs'], batch['proprio_obs'], a_pi)
                q2_pi = critic2(batch['visual_obs'], batch['proprio_obs'], a_pi)
                q_pi_min = torch.min(q1_pi, q2_pi)
                actor_loss = (alpha_manager.alpha * logp_pi - q_pi_min).mean()

                actor_opt.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                actor_opt.step()

                # Alpha update
                _alpha_loss = alpha_manager.update(logp_pi)

                # Soft update targets
                for param, target_param in zip(critic1.parameters(), critic1_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                for param, target_param in zip(critic2.parameters(), critic2_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            if done:
                break

        # Episode stats
        is_success = info.get('success', False)
        success_history.append(float(is_success))
        reward_history.append(ep_reward)

        # Save best model
        if ep_reward > best_reward:
            best_reward = ep_reward
            torch.save({
                'actor': actor.state_dict(),
                'critic1': critic1.state_dict(),
                'critic2': critic2.state_dict(),
                'visual_dim': visual_dim,
                'proprio_dim': proprio_dim,
                'action_dim': action_dim,
                'episode': ep,
                'best_reward': best_reward,
            }, "maniskill_r3m_best.pt")

        # Periodic logging
        if ep % 5 == 0:
            avg_reward = np.mean(reward_history[-20:]) if len(reward_history) >= 20 else np.mean(reward_history)
            avg_success = np.mean(success_history[-20:]) if len(success_history) >= 20 else np.mean(success_history)
            print(
                f"Ep {ep:4d} | Reward: {ep_reward:7.2f} | Avg: {avg_reward:7.2f} | "
                f"Success: {avg_success:.2%} | Steps: {ep_steps:3d} | "
                f"Alpha: {alpha_manager.alpha.item():.3f} | Buffer: {buffer.size:6d}"
            )

    # Save final model
    torch.save({
        'actor': actor.state_dict(),
        'critic1': critic1.state_dict(),
        'critic2': critic2.state_dict(),
        'reward_history': reward_history,
        'success_history': success_history,
        'visual_dim': visual_dim,
        'proprio_dim': proprio_dim,
        'action_dim': action_dim,
    }, "maniskill_r3m_final.pt")

    env.close()
    print("\nTraining finished!")
    print(f"Best reward: {best_reward:.2f}")
    print(f"Final 20-episode average reward: {np.mean(reward_history[-20:]):.2f}")
    print(f"Final 20-episode average success rate: {np.mean(success_history[-20:]):.2%}")

if __name__ == "__main__":
    main()