"""
NaviSim Navigation: Complete Training and Visualization Pipeline

This script provides a unified interface for:
1. Collecting data with scripted policies (baseline)
2. Training RL policies (SAC/PPO) with replay buffer and gradient updates
3. Testing trained policies with Jetbot first-person camera capture
4. Generating visualizations (GIFs, key frames, statistics)

Directory structure:
    result_navisim/    - Test episode GIFs and key frames
    logs_navisim/      - Training logs, checkpoints, TensorBoard
    videos_navisim/    - Recorded episodes during testing
    buffers_navisim/   - Replay buffers for RL training

Usage:
    # Collect baseline data with scripted policy
    ./run_isaac_lab.sh "python train_navisim_v2.py --mode collect --num_envs 4 --total_steps 50000"
    
    # Train SAC policy
    ./run_isaac_lab.sh "python train_navisim_v2.py --mode train --policy_type sac --num_envs 4 --total_steps 100000"
    
    # Test trained policy and generate visualizations
    ./run_isaac_lab.sh "python train_navisim_v2.py --mode test --policy_type sac --model_path logs_navisim/sac_policy_final.pth --num_episodes 5"
    
    # Train and then test
    ./run_isaac_lab.sh "python train_navisim_v2.py --mode both --policy_type sac --num_envs 4 --total_steps 100000 --num_episodes 5"
"""

import argparse
import os
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless mode
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# Import isaaclab.app first for proper initialization
from isaaclab.app import AppLauncher

# ===========================================
# 1. Parse Arguments FIRST
# ===========================================
def parse_args():
    parser = argparse.ArgumentParser(description="NaviSim Navigation Training and Visualization")
    
    # Mode selection
    parser.add_argument("--mode", type=str, default="both", 
                        choices=["collect", "train", "test", "both"],
                        help="Mode: collect (scripted baseline), train (RL), test, or both (train+test)")
    
    # Training arguments
    parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--total_steps", type=int, default=100000, help="Total training steps")
    parser.add_argument("--save_interval", type=int, default=10000, help="Steps between checkpoint saves")
    parser.add_argument("--log_interval", type=int, default=1000, help="Steps between logging")
    
    # RL-specific arguments
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate for RL")
    parser.add_argument("--buffer_size", type=int, default=1000000, help="Replay buffer size")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.005, help="Soft update coefficient")
    parser.add_argument("--learning_starts", type=int, default=10000, help="Steps before learning starts")
    
    # Testing arguments
    parser.add_argument("--num_episodes", type=int, default=5, help="Number of test episodes")
    parser.add_argument("--max_episode_steps", type=int, default=500, help="Maximum steps per test episode")
    parser.add_argument("--camera_save_interval", type=int, default=50, help="Steps between camera image saves")
    parser.add_argument("--model_path", type=str, default=None, help="Path to trained model (for testing)")
    
    # Directory configuration
    parser.add_argument("--result_dir", type=str, default="result_navisim", help="Directory for test results")
    parser.add_argument("--log_dir", type=str, default="logs_navisim", help="Directory for training logs")
    parser.add_argument("--video_dir", type=str, default="videos_navisim", help="Directory for videos")
    parser.add_argument("--buffer_dir", type=str, default="buffers_navisim", help="Directory for replay buffers")
    
    # Policy configuration
    parser.add_argument("--policy_type", type=str, default="scripted", 
                        choices=["scripted", "sac", "ppo"],
                        help="Policy type: scripted (baseline), sac (Soft Actor-Critic), or ppo (not implemented)")
    
    # AppLauncher args (--headless, --device, etc.)
    # Note: AppLauncher.add_app_launcher_args() adds --device automatically, so we don't add it here
    AppLauncher.add_app_launcher_args(parser)
    
    return parser.parse_args()


# ===========================================
# 2. Launch Isaac Sim Application
# ===========================================
def launch_simulation(args):
    """Launch Isaac Sim via AppLauncher."""
    args.enable_cameras = True  # Enable camera sensors
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    return simulation_app


# ===========================================
# 3. Import Modules AFTER App Launch
# ===========================================
def setup_imports():
    """Import project modules after Isaac Sim is initialized."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Import project modules
    from configs.navigation_env_cfg import NavisimNavigationEnvCfg
    from tasks.navigation_env import NavisimNavigationEnv
    from policies.base_policy import BasePolicy, PolicyConfig
    
    return NavisimNavigationEnvCfg, NavisimNavigationEnv, BasePolicy, PolicyConfig


# ===========================================
# 4. Directory Management
# ===========================================
def setup_directories(args):
    """Create necessary directories."""
    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.video_dir, exist_ok=True)
    os.makedirs(args.buffer_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"[Setup] Directories configured:")
    print(f"  Results:  {args.result_dir}")
    print(f"  Logs:     {args.log_dir}")
    print(f"  Videos:   {args.video_dir}")
    print(f"  Buffers:  {args.buffer_dir}")
    print(f"{'='*80}\n")


# ===========================================
# 5. Replay Buffer for RL
# ===========================================
class ReplayBuffer:
    """Simple replay buffer for off-policy RL algorithms."""
    
    def __init__(self, obs_dim: int, action_dim: int, capacity: int, device: str = "cuda:0"):
        self.capacity = capacity
        self.device = torch.device(device)
        self.ptr = 0
        self.size = 0
        
        # Pre-allocate memory
        self.observations = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=self.device)
        self.actions = torch.zeros((capacity, action_dim), dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=self.device)
        self.next_observations = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=self.device)
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32, device=self.device)
    
    def add(self, obs: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, 
            next_obs: torch.Tensor, done: torch.Tensor):
        """Add a batch of transitions to the buffer."""
        batch_size = obs.shape[0]
        
        for i in range(batch_size):
            self.observations[self.ptr] = obs[i]
            self.actions[self.ptr] = action[i]
            self.rewards[self.ptr] = reward[i]
            self.next_observations[self.ptr] = next_obs[i]
            self.dones[self.ptr] = done[i]
            
            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample a batch of transitions."""
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        
        return (
            self.observations[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_observations[indices],
            self.dones[indices]
        )
    
    def __len__(self) -> int:
        return self.size
    
    def save(self, path: str):
        """Save buffer to disk."""
        torch.save({
            "observations": self.observations[:self.size].cpu(),
            "actions": self.actions[:self.size].cpu(),
            "rewards": self.rewards[:self.size].cpu(),
            "next_observations": self.next_observations[:self.size].cpu(),
            "dones": self.dones[:self.size].cpu(),
            "ptr": self.ptr,
            "size": self.size,
        }, path)
        print(f"[Buffer] Saved {self.size} transitions to {path}")
    
    def load(self, path: str):
        """Load buffer from disk."""
        checkpoint = torch.load(path)
        size = checkpoint["size"]
        
        self.observations[:size] = checkpoint["observations"].to(self.device)
        self.actions[:size] = checkpoint["actions"].to(self.device)
        self.rewards[:size] = checkpoint["rewards"].to(self.device)
        self.next_observations[:size] = checkpoint["next_observations"].to(self.device)
        self.dones[:size] = checkpoint["dones"].to(self.device)
        self.ptr = checkpoint["ptr"]
        self.size = size
        
        print(f"[Buffer] Loaded {self.size} transitions from {path}")


# ===========================================
# 6. SAC Neural Networks
# ===========================================
class Actor(nn.Module):
    """SAC Actor network (Gaussian policy)."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        
        self.action_scale = 1.0  # Will be set based on action space
        self.action_bias = 0.0
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returns mean and log_std."""
        features = self.net(obs)
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)
        log_std = torch.clamp(log_std, -20, 2)  # Stabilize training
        return mean, log_std
    
    def sample(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy."""
        mean, log_std = self.forward(obs)
        
        if deterministic:
            action = torch.tanh(mean)
        else:
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()  # Reparameterization trick
            action = torch.tanh(x_t)
            
            # Compute log probability with squashing correction
            log_prob = normal.log_prob(x_t)
            log_prob -= torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(1, keepdim=True)
            return action * self.action_scale + self.action_bias, log_prob
        
        return action * self.action_scale + self.action_bias, None


class Critic(nn.Module):
    """SAC Critic network (twin Q-networks)."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # Q1
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        # Q2
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returns Q1 and Q2 values."""
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)


# ===========================================
# 7. Policy Implementations
# ===========================================
class ScriptedNavigationPolicy:
    """Simple scripted policy for navigation (forward + periodic turns)."""
    
    def __init__(self, num_envs: int, action_dim: int, device: str = "cuda:0"):
        self.num_envs = num_envs
        self.action_dim = action_dim
        self.device = torch.device(device)
        self.step_count = 0
        
        # Simple behavior: forward for 100 steps, turn for 20 steps
        self.forward_steps = 100
        self.turn_steps = 20
        self.cycle_length = self.forward_steps + self.turn_steps
    
    def compute_action(self, observations: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """Compute actions based on simple scripted behavior."""
        phase = self.step_count % self.cycle_length
        
        if phase < self.forward_steps:
            # Forward motion
            actions = torch.zeros((self.num_envs, self.action_dim), device=self.device)
            actions[:, 0] = 0.5  # Left wheel forward
            actions[:, 1] = 0.5  # Right wheel forward
        else:
            # Turn in place
            actions = torch.zeros((self.num_envs, self.action_dim), device=self.device)
            actions[:, 0] = 0.3   # Left wheel forward
            actions[:, 1] = -0.3  # Right wheel backward (turn)
        
        self.step_count += 1
        return actions
    
    def reset(self):
        """Reset policy state."""
        self.step_count = 0
    
    def save(self, path: str):
        """Save policy (minimal for scripted policy)."""
        save_dict = {
            "policy_type": "scripted",
            "step_count": self.step_count,
        }
        torch.save(save_dict, path)
        print(f"[Policy] Saved scripted policy to {path}")
    
    def load(self, path: str):
        """Load policy."""
        checkpoint = torch.load(path)
        self.step_count = checkpoint.get("step_count", 0)
        print(f"[Policy] Loaded scripted policy from {path}")


class SACPolicy:
    """Soft Actor-Critic policy for continuous control."""
    
    def __init__(self, obs_dim: int, action_dim: int, args, device: str = "cuda:0"):
        self.device = torch.device(device)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Hyperparameters
        self.gamma = args.gamma
        self.tau = args.tau
        self.learning_rate = args.learning_rate
        self.batch_size = args.batch_size
        
        # Networks
        self.actor = Actor(obs_dim, action_dim).to(self.device)
        self.critic = Critic(obs_dim, action_dim).to(self.device)
        self.critic_target = Critic(obs_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.learning_rate)
        
        # Automatic entropy tuning
        self.target_entropy = -action_dim  # Heuristic
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.learning_rate)
        
        # Training stats
        self.update_count = 0
    
    @property
    def alpha(self):
        return self.log_alpha.exp()
    
    def compute_action(self, observations: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Compute actions from observations."""
        with torch.no_grad():
            action, _ = self.actor.sample(observations, deterministic=deterministic)
        return action
    
    def update(self, replay_buffer: ReplayBuffer) -> Dict[str, float]:
        """Update SAC networks."""
        # Sample from replay buffer
        obs, actions, rewards, next_obs, dones = replay_buffer.sample(self.batch_size)
        
        # ==================== Update Critic ====================
        with torch.no_grad():
            # Sample actions from current policy
            next_actions, next_log_probs = self.actor.sample(next_obs, deterministic=False)
            
            # Compute target Q-values
            q1_target, q2_target = self.critic_target(next_obs, next_actions)
            q_target = torch.min(q1_target, q2_target) - self.alpha * next_log_probs
            target = rewards + (1 - dones) * self.gamma * q_target
        
        # Current Q-values
        q1, q2 = self.critic(obs, actions)
        
        # Critic loss
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # ==================== Update Actor ====================
        # Sample actions from current policy
        new_actions, log_probs = self.actor.sample(obs, deterministic=False)
        
        # Compute Q-values for new actions
        q1_new, q2_new = self.critic(obs, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        # Actor loss
        actor_loss = (self.alpha * log_probs - q_new).mean()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # ==================== Update Alpha ====================
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # ==================== Soft Update Target Network ====================
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        self.update_count += 1
        
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha.item(),
            "q_value": q_new.mean().item(),
        }
    
    def reset(self):
        """Reset policy state (if needed)."""
        pass
    
    def save(self, path: str):
        """Save policy networks and optimizers."""
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha,
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "update_count": self.update_count,
        }, path)
        print(f"[Policy] Saved SAC policy to {path}")
    
    def load(self, path: str):
        """Load policy networks and optimizers."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.log_alpha = checkpoint["log_alpha"]
        self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
        self.update_count = checkpoint["update_count"]
        
        print(f"[Policy] Loaded SAC policy from {path} (updates: {self.update_count})")


# ===========================================
# 8. Data Collection (Scripted Baseline)
# ===========================================
def collect_baseline_data(args, simulation_app):
    """Collect baseline data using scripted policy (no learning)."""
    NavisimNavigationEnvCfg, NavisimNavigationEnv, _, _ = setup_imports()
    
    print(f"\n{'='*80}")
    print(f"[Collect] Starting Baseline Data Collection")
    print(f"{'='*80}")
    print(f"  Policy type:      scripted (no learning)")
    print(f"  Num environments: {args.num_envs}")
    print(f"  Total steps:      {args.total_steps}")
    print(f"  Purpose:          Baseline performance evaluation")
    print(f"{'='*80}\n")
    
    # Create environment
    env_cfg = NavisimNavigationEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    
    # Device is managed by AppLauncher
    print(f"[Setup] Using device: {env_cfg.sim.device}")
    
    render_mode = None if args.headless else "human"
    env = NavisimNavigationEnv(cfg=env_cfg, render_mode=render_mode)
    
    # Reset environment first to get actual observations
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    
    # Get observation and action dimensions from actual tensors (more robust than using observation_space)
    obs_dim = obs.shape[-1]
    action_dim = env.action_space.shape[0]
    
    print(f"[Setup] Observation dim: {obs_dim}, Action dim: {action_dim}")
    
    # Create scripted policy
    policy = ScriptedNavigationPolicy(
        num_envs=args.num_envs,
        action_dim=action_dim,
        device=str(env.device)
    )
    
    # Statistics tracking
    episode_rewards = torch.zeros(args.num_envs, device=env.device)
    episode_lengths = torch.zeros(args.num_envs, device=env.device)
    completed_episodes = 0
    all_episode_rewards = []
    all_episode_lengths = []
    
    print(f"[Collect] Starting data collection loop...\n")
    
    for step in range(args.total_steps):
        # Compute actions
        actions = policy.compute_action(obs)
        
        # Step environment
        obs_dict, rewards, terminated, truncated, info = env.step(actions)
        obs = obs_dict["policy"]
        
        episode_rewards += rewards
        episode_lengths += 1
        
        # Handle episode completion
        dones = terminated | truncated
        if dones.any():
            done_indices = torch.where(dones)[0]
            for idx in done_indices:
                completed_episodes += 1
                ep_reward = episode_rewards[idx].item()
                ep_length = episode_lengths[idx].item()
                
                all_episode_rewards.append(ep_reward)
                all_episode_lengths.append(ep_length)
                
                print(f"[Step {step:6d}] Episode {completed_episodes:4d} | "
                      f"Env {idx} | Reward: {ep_reward:+.3f} | Length: {ep_length:.0f}")
            
            episode_rewards[dones] = 0.0
            episode_lengths[dones] = 0
        
        # Periodic logging
        if step % args.log_interval == 0 and step > 0:
            if len(all_episode_rewards) > 0:
                mean_reward = np.mean(all_episode_rewards[-100:])  # Last 100 episodes
                mean_length = np.mean(all_episode_lengths[-100:])
                print(f"[Step {step:6d}] Progress | "
                      f"Episodes: {completed_episodes} | "
                      f"Mean Reward (last 100): {mean_reward:+.3f} | "
                      f"Mean Length: {mean_length:.1f}")
        
        # Check if simulation is still running
        if not simulation_app.is_running():
            print(f"\n[Collect] Simulation stopped by user")
            break
    
    # Save baseline statistics
    stats_path = os.path.join(args.log_dir, "baseline_stats.txt")
    with open(stats_path, "w") as f:
        f.write(f"Baseline Data Collection Statistics\n")
        f.write(f"{'='*80}\n")
        f.write(f"Total steps:          {step + 1}\n")
        f.write(f"Episodes completed:   {completed_episodes}\n")
        f.write(f"Mean episode reward:  {np.mean(all_episode_rewards):.3f} ± {np.std(all_episode_rewards):.3f}\n")
        f.write(f"Mean episode length:  {np.mean(all_episode_lengths):.1f} ± {np.std(all_episode_lengths):.1f}\n")
    
    print(f"\n{'='*80}")
    print(f"[Collect] Baseline data collection completed!")
    print(f"  Total steps:          {step + 1}")
    print(f"  Episodes completed:   {completed_episodes}")
    print(f"  Mean episode reward:  {np.mean(all_episode_rewards):.3f} ± {np.std(all_episode_rewards):.3f}")
    print(f"  Mean episode length:  {np.mean(all_episode_lengths):.1f} ± {np.std(all_episode_lengths):.1f}")
    print(f"  Statistics saved:     {stats_path}")
    print(f"{'='*80}\n")
    
    env.close()


# ===========================================
# 9. RL Training Function
# ===========================================
def train_rl_policy(args, simulation_app):
    """Train RL policy (SAC) with replay buffer and gradient updates."""
    NavisimNavigationEnvCfg, NavisimNavigationEnv, _, _ = setup_imports()
    
    print(f"\n{'='*80}")
    print(f"[Training] Starting RL Policy Training")
    print(f"{'='*80}")
    print(f"  Policy type:        {args.policy_type.upper()}")
    print(f"  Num environments:   {args.num_envs}")
    print(f"  Total steps:        {args.total_steps}")
    print(f"  Learning rate:      {args.learning_rate}")
    print(f"  Batch size:         {args.batch_size}")
    print(f"  Buffer size:        {args.buffer_size}")
    print(f"  Learning starts:    {args.learning_starts}")
    print(f"  Gamma:              {args.gamma}")
    print(f"  Tau:                {args.tau}")
    print(f"{'='*80}\n")
    
    # Create environment
    env_cfg = NavisimNavigationEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    
    # Device is managed by AppLauncher
    print(f"[Setup] Using device: {env_cfg.sim.device}")
    
    render_mode = None if args.headless else "human"
    env = NavisimNavigationEnv(cfg=env_cfg, render_mode=render_mode)
    
    # Reset environment first to get actual observations
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    
    # Get observation and action dimensions from actual tensors (more robust than using observation_space)
    obs_dim = obs.shape[-1]
    action_dim = env.action_space.shape[0]
    
    print(f"[Setup] Observation dim: {obs_dim}, Action dim: {action_dim}")
    
    # Create RL policy
    if args.policy_type == "sac":
        policy = SACPolicy(obs_dim, action_dim, args, device=str(env.device))
    else:
        raise NotImplementedError(f"Policy type '{args.policy_type}' not implemented")
    
    # Create replay buffer
    replay_buffer = ReplayBuffer(
        obs_dim=obs_dim,
        action_dim=action_dim,
        capacity=args.buffer_size,
        device=str(env.device)
    )
    
    print(f"[Setup] Created replay buffer with capacity {args.buffer_size}")
    
    # Statistics tracking
    episode_rewards = torch.zeros(args.num_envs, device=env.device)
    episode_lengths = torch.zeros(args.num_envs, device=env.device)
    completed_episodes = 0
    all_episode_rewards = []
    all_episode_lengths = []
    
    # Training stats
    recent_losses = deque(maxlen=100)
    
    print(f"[Training] Starting RL training loop...\n")
    
    for step in range(args.total_steps):
        # Select action (exploration mode before learning starts)
        if step < args.learning_starts:
            # Random actions for exploration
            actions = torch.rand((args.num_envs, action_dim), device=env.device) * 2 - 1  # [-1, 1]
        else:
            # Policy actions (with exploration noise)
            actions = policy.compute_action(obs, deterministic=False)
        
        # Step environment
        next_obs_dict, rewards, terminated, truncated, info = env.step(actions)
        next_obs = next_obs_dict["policy"]
        
        # Store transitions in replay buffer
        dones = (terminated | truncated).float().unsqueeze(-1)
        replay_buffer.add(obs, actions, rewards.unsqueeze(-1), next_obs, dones)
        
        obs = next_obs
        episode_rewards += rewards
        episode_lengths += 1
        
        # Update policy
        if step >= args.learning_starts and len(replay_buffer) >= args.batch_size:
            update_info = policy.update(replay_buffer)
            recent_losses.append(update_info)
        
        # Handle episode completion
        dones_bool = terminated | truncated
        if dones_bool.any():
            done_indices = torch.where(dones_bool)[0]
            for idx in done_indices:
                completed_episodes += 1
                ep_reward = episode_rewards[idx].item()
                ep_length = episode_lengths[idx].item()
                
                all_episode_rewards.append(ep_reward)
                all_episode_lengths.append(ep_length)
                
                print(f"[Step {step:6d}] Episode {completed_episodes:4d} | "
                      f"Env {idx} | Reward: {ep_reward:+.3f} | Length: {ep_length:.0f}")
            
            episode_rewards[dones_bool] = 0.0
            episode_lengths[dones_bool] = 0
        
        # Periodic logging
        if step % args.log_interval == 0 and step > 0:
            log_msg = f"[Step {step:6d}] "
            
            if len(all_episode_rewards) > 0:
                mean_reward = np.mean(all_episode_rewards[-100:])
                mean_length = np.mean(all_episode_lengths[-100:])
                log_msg += f"Episodes: {completed_episodes} | Reward: {mean_reward:+.3f} | Length: {mean_length:.1f}"
            
            if len(recent_losses) > 0 and step >= args.learning_starts:
                mean_critic_loss = np.mean([d["critic_loss"] for d in recent_losses])
                mean_actor_loss = np.mean([d["actor_loss"] for d in recent_losses])
                mean_alpha = np.mean([d["alpha"] for d in recent_losses])
                log_msg += f" | C-Loss: {mean_critic_loss:.3f} | A-Loss: {mean_actor_loss:.3f} | Alpha: {mean_alpha:.3f}"
            
            if step < args.learning_starts:
                log_msg += " | [EXPLORATION PHASE]"
            
            print(log_msg)
        
        # Save checkpoints
        if step % args.save_interval == 0 and step > 0:
            checkpoint_path = os.path.join(args.log_dir, f"{args.policy_type}_policy_step_{step}.pth")
            policy.save(checkpoint_path)
            
            buffer_path = os.path.join(args.buffer_dir, f"buffer_step_{step}.pth")
            replay_buffer.save(buffer_path)
            
            print(f"[Step {step:6d}] ✓ Checkpoint saved: {checkpoint_path}")
        
        # Check if simulation is still running
        if not simulation_app.is_running():
            print(f"\n[Training] Simulation stopped by user")
            break
    
    # Save final model and buffer
    final_policy_path = os.path.join(args.log_dir, f"{args.policy_type}_policy_final.pth")
    policy.save(final_policy_path)
    
    final_buffer_path = os.path.join(args.buffer_dir, "buffer_final.pth")
    replay_buffer.save(final_buffer_path)
    
    # Save training statistics
    stats_path = os.path.join(args.log_dir, f"{args.policy_type}_training_stats.txt")
    with open(stats_path, "w") as f:
        f.write(f"{args.policy_type.upper()} Training Statistics\n")
        f.write(f"{'='*80}\n")
        f.write(f"Total steps:          {step + 1}\n")
        f.write(f"Episodes completed:   {completed_episodes}\n")
        f.write(f"Buffer size:          {len(replay_buffer)}\n")
        f.write(f"Policy updates:       {policy.update_count}\n")
        if len(all_episode_rewards) > 0:
            f.write(f"Mean episode reward:  {np.mean(all_episode_rewards):.3f} ± {np.std(all_episode_rewards):.3f}\n")
            f.write(f"Mean episode length:  {np.mean(all_episode_lengths):.1f} ± {np.std(all_episode_lengths):.1f}\n")
    
    print(f"\n{'='*80}")
    print(f"[Training] RL training completed!")
    print(f"  Total steps:          {step + 1}")
    print(f"  Episodes completed:   {completed_episodes}")
    print(f"  Buffer size:          {len(replay_buffer)}")
    print(f"  Policy updates:       {policy.update_count}")
    if len(all_episode_rewards) > 0:
        print(f"  Mean episode reward:  {np.mean(all_episode_rewards):.3f} ± {np.std(all_episode_rewards):.3f}")
        print(f"  Mean episode length:  {np.mean(all_episode_lengths):.1f} ± {np.std(all_episode_lengths):.1f}")
    print(f"  Final model:          {final_policy_path}")
    print(f"  Statistics saved:     {stats_path}")
    print(f"{'='*80}\n")
    
    env.close()
    return policy


# ===========================================
# 10. Visualization Helper Functions
# ===========================================
def save_gif(frames: List[np.ndarray], filename: str, duration: int = 50):
    """Save frames as animated GIF."""
    if len(frames) == 0:
        print(f"[Warning] No frames to save for {filename}")
        return
    
    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(
        filename,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )
    print(f"[Save] GIF saved: {filename}")


def save_key_frames(frames: List[np.ndarray], episode_idx: int, output_dir: str):
    """Save key frames from episode."""
    if len(frames) == 0:
        return
    
    key_indices = [
        0,
        len(frames) // 4,
        len(frames) // 2,
        (len(frames) * 3) // 4,
        len(frames) - 1
    ]
    
    for idx in key_indices:
        if 0 <= idx < len(frames):
            plt.figure(figsize=(8, 6))
            plt.imshow(frames[idx])
            plt.axis("off")
            plt.title(f"Episode {episode_idx} - Step {idx}", fontsize=14)
            
            save_path = os.path.join(output_dir, f"ep{episode_idx}_step{idx:04d}.png")
            plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1, dpi=100)
            plt.close()


def save_camera_image(rgb_array: np.ndarray, output_dir: str, 
                      episode_idx: int, step: int) -> str:
    """Save camera RGB array to image file."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure uint8 format
    if rgb_array.dtype != np.uint8:
        if rgb_array.max() <= 1.0:
            rgb_array = (rgb_array * 255).astype(np.uint8)
        else:
            rgb_array = rgb_array.astype(np.uint8)
    
    filename = f"ep{episode_idx}_camera_step{step:04d}.png"
    filepath = os.path.join(output_dir, filename)
    
    img = Image.fromarray(rgb_array)
    img.save(filepath)
    
    return filepath


# ===========================================
# 11. Testing and Visualization Function
# ===========================================
def test_and_visualize(args, simulation_app, policy=None):
    """Test policy and generate visualizations."""
    NavisimNavigationEnvCfg, NavisimNavigationEnv, _, _ = setup_imports()
    
    print(f"\n{'='*80}")
    print(f"[Testing] Starting Policy Evaluation with Visualization")
    print(f"{'='*80}")
    print(f"  Policy type:          {args.policy_type}")
    print(f"  Num episodes:         {args.num_episodes}")
    print(f"  Max steps per episode: {args.max_episode_steps}")
    print(f"  Camera save interval: {args.camera_save_interval}")
    print(f"{'='*80}\n")
    
    # Create environment (single env for testing)
    env_cfg = NavisimNavigationEnvCfg()
    env_cfg.scene.num_envs = 1
    
    # Device is managed by AppLauncher
    
    render_mode = None if args.headless else "human"
    env = NavisimNavigationEnv(cfg=env_cfg, render_mode=render_mode)
    
    # Reset environment first to get actual observations
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    
    # Get dimensions from actual tensors (more robust than using observation_space)
    obs_dim = obs.shape[-1]
    action_dim = env.action_space.shape[0]
    
    # Load or create policy
    if policy is None:
        if args.model_path and os.path.exists(args.model_path):
            if args.policy_type == "scripted":
                policy = ScriptedNavigationPolicy(num_envs=1, action_dim=action_dim, device=str(env.device))
                policy.load(args.model_path)
            elif args.policy_type == "sac":
                policy = SACPolicy(obs_dim, action_dim, args, device=str(env.device))
                policy.load(args.model_path)
            print(f"[Policy] Loaded from {args.model_path}")
        else:
            if args.policy_type == "scripted":
                policy = ScriptedNavigationPolicy(num_envs=1, action_dim=action_dim, device=str(env.device))
            elif args.policy_type == "sac":
                policy = SACPolicy(obs_dim, action_dim, args, device=str(env.device))
            print(f"[Policy] Using fresh {args.policy_type} policy")
    
    # Access camera sensor
    camera = env.scene["jetbot_camera"]
    print(f"[Camera] ✓ Camera accessed: {camera.cfg.prim_path}")
    print(f"[Camera]   Resolution: {camera.image_shape}\n")
    
    # Set viewport camera if not headless
    if not args.headless:
        try:
            jetbot_pos = env.scene["jetbot"].data.root_pos_w[0].cpu().numpy()
            eye = jetbot_pos + np.array([3.0, 3.0, 2.0])
            target = jetbot_pos
            env.sim.set_camera_view(eye=eye.tolist(), target=target.tolist())
            print(f"[Viewport] ✓ Camera positioned\n")
        except Exception as e:
            print(f"[Viewport] Warning: {e}\n")
    
    # Test episodes
    episode_stats = []
    
    for ep in range(args.num_episodes):
        print(f"{'='*80}")
        print(f"[Episode {ep+1}/{args.num_episodes}] Starting...")
        print(f"{'='*80}")
        
        obs_dict, _ = env.reset()
        obs = obs_dict["policy"]
        policy.reset()
        
        frames = []  # For GIF
        episode_reward = 0.0
        episode_length = 0
        
        done = False
        step = 0
        
        while not done and step < args.max_episode_steps:
            # Capture viewport frame (for GIF) - only if not headless
            if not args.headless:
                try:
                    frame = env.sim.render()
                    if frame is not None:
                        if isinstance(frame, torch.Tensor):
                            frame = frame.cpu().numpy()
                        while frame.ndim > 3:
                            frame = frame.squeeze(0)
                        # Ensure uint8
                        if frame.dtype != np.uint8:
                            if frame.max() <= 1.0:
                                frame = (frame * 255).astype(np.uint8)
                            else:
                                frame = frame.astype(np.uint8)
                        frames.append(frame)
                except Exception as e:
                    if step == 0:  # Only warn once
                        print(f"[Warning] Could not capture viewport frame: {e}")
            
            # Capture camera image (Jetbot POV)
            if step % args.camera_save_interval == 0:
                camera_data = camera.data
                rgb_tensor = camera_data.output["rgb"][0]
                rgb_array = rgb_tensor.cpu().numpy()
                
                cam_pos = camera_data.pos_w[0].cpu().numpy()
                
                # Save camera image
                filepath = save_camera_image(
                    rgb_array, 
                    os.path.join(args.result_dir, f"camera_ep{ep}"),
                    ep, 
                    step
                )
                print(f"  [Step {step:4d}] 📸 Camera | Pos: [{cam_pos[0]:.2f}, {cam_pos[1]:.2f}, {cam_pos[2]:.2f}]")
            
            # Compute action and step (deterministic for testing)
            actions = policy.compute_action(obs, deterministic=True)
            obs_dict, rewards, terminated, truncated, info = env.step(actions)
            obs = obs_dict["policy"]
            
            episode_reward += rewards[0].item()
            episode_length += 1
            done = terminated[0] or truncated[0]
            step += 1
        
        # Save episode GIF (only if we have frames)
        if len(frames) > 0:
            gif_path = os.path.join(args.result_dir, f"episode_{ep}.gif")
            save_gif(frames, gif_path, duration=50)
            
            # Save key frames
            save_key_frames(frames, ep, args.result_dir)
        elif not args.headless:
            print(f"[Warning] No frames captured for episode {ep}")
        
        # Record stats
        episode_stats.append({
            "episode": ep,
            "reward": episode_reward,
            "length": episode_length,
            "completed": done
        })
        
        print(f"\n[Episode {ep+1}] Completed!")
        print(f"  Reward: {episode_reward:+.3f}")
        print(f"  Length: {episode_length} steps")
        print(f"  Status: {'✓ Done' if done else '✗ Truncated'}\n")
    
    # Print and save summary statistics
    print(f"\n{'='*80}")
    print(f"[Testing] Summary Statistics")
    print(f"{'='*80}")
    
    total_reward = sum(s["reward"] for s in episode_stats)
    mean_reward = total_reward / len(episode_stats)
    std_reward = np.std([s["reward"] for s in episode_stats])
    mean_length = sum(s["length"] for s in episode_stats) / len(episode_stats)
    completion_rate = sum(s["completed"] for s in episode_stats) / len(episode_stats)
    
    print(f"  Episodes:         {args.num_episodes}")
    print(f"  Mean reward:      {mean_reward:+.3f} ± {std_reward:.3f}")
    print(f"  Mean length:      {mean_length:.1f} steps")
    print(f"  Completion rate:  {completion_rate*100:.1f}%")
    print(f"  Results saved to: {args.result_dir}")
    print(f"{'='*80}\n")
    
    # Save test statistics
    stats_path = os.path.join(args.result_dir, f"{args.policy_type}_test_stats.txt")
    with open(stats_path, "w") as f:
        f.write(f"{args.policy_type.upper()} Test Statistics\n")
        f.write(f"{'='*80}\n")
        f.write(f"Episodes:         {args.num_episodes}\n")
        f.write(f"Mean reward:      {mean_reward:+.3f} ± {std_reward:.3f}\n")
        f.write(f"Mean length:      {mean_length:.1f} steps\n")
        f.write(f"Completion rate:  {completion_rate*100:.1f}%\n")
        f.write(f"\nPer-episode results:\n")
        for stat in episode_stats:
            f.write(f"  Episode {stat['episode']}: Reward={stat['reward']:+.3f}, Length={stat['length']}, {'✓' if stat['completed'] else '✗'}\n")
    
    env.close()


# ===========================================
# 12. Main Function
# ===========================================
def main():
    args = parse_args()
    
    # Setup directories
    setup_directories(args)
    
    # Launch Isaac Sim
    simulation_app = launch_simulation(args)
    
    try:
        trained_policy = None
        
        # Data collection phase (scripted baseline)
        if args.mode == "collect":
            collect_baseline_data(args, simulation_app)
        
        # Training phase (RL with learning)
        elif args.mode == "train":
            if args.policy_type == "scripted":
                print(f"[Error] Cannot 'train' scripted policy (no learning). Use --mode collect instead.")
            else:
                trained_policy = train_rl_policy(args, simulation_app)
        
        # Testing phase
        elif args.mode == "test":
            test_and_visualize(args, simulation_app, policy=trained_policy)
        
        # Both training and testing
        elif args.mode == "both":
            if args.policy_type == "scripted":
                print(f"[Error] Cannot 'train' scripted policy. Use --mode collect or --mode test instead.")
            else:
                trained_policy = train_rl_policy(args, simulation_app)
                test_and_visualize(args, simulation_app, policy=trained_policy)
        
        print(f"\n{'='*80}")
        print(f"[Complete] All operations finished successfully!")
        print(f"{'='*80}")
        print(f"  Check results in:  {args.result_dir}")
        print(f"  Check logs in:     {args.log_dir}")
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"\n[Error] {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()