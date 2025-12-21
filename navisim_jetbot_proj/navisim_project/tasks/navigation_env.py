"""
Navigation Environment Implementation for NaviSim

This implements the actual environment logic for robot navigation.
"""

from __future__ import annotations

import torch
from typing import Any, Dict

# Isaac Lab imports（注意：现在全部来自 isaaclab 包）
from isaaclab.envs import DirectRLEnv
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import Camera, ContactSensor


class NavisimNavigationEnv(DirectRLEnv):
    """
    Navigation environment for differential drive robot (Jetbot).

    The robot receives observations and must navigate to a goal position
    while avoiding obstacles and maximizing forward motion.
    """

    cfg: Any  # Will be NavisimNavigationEnvCfg

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        """Initialize the navigation environment.

        Args:
            cfg: Configuration for the environment.
            render_mode: Render mode for the environment. Defaults to None.
        """
        super().__init__(cfg, render_mode, **kwargs)

    # =====================================================
    # Scene setup
    # =====================================================

    def _setup_scene(self):
        """Setup the scene with robot, terrain, sensors."""
        # Add robot
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        # Add camera sensor (with error handling)
        try:
            self._camera = Camera(self.cfg.jetbot_camera)
            self.scene.sensors["jetbot_camera"] = self._camera
        except Exception as e:
            print(f"[Env] Warning: Camera setup failed: {e}")
            self._camera = None

        # Add contact sensor (with error handling)
        try:
            self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
            self.scene.sensors["contact_sensor"] = self._contact_sensor
        except Exception as e:
            print(f"[Env] Warning: Contact sensor setup failed: {e}")
            self._contact_sensor = None

        # Clone environments
        self.scene.clone_environments(copy_from_source=False)

        # Add lighting
        light_cfg = sim_utils.DomeLightCfg(
            intensity=2000.0,
            color=(0.75, 0.75, 0.75),
        )
        light_cfg.func("/World/skyLight", light_cfg)

        # Call parent to finish scene setup
        super()._setup_scene()

        # Initialize buffers AFTER scene is ready
        self._init_buffers()

    def _init_buffers(self):
        """Initialize internal buffers for tracking state."""
        # Goal positions (randomized per episode)
        self.goal_positions = torch.zeros((self.num_envs, 3), device=self.device)

        # Episode tracking
        self.previous_distance_to_goal = torch.zeros(self.num_envs, device=self.device)

        # Actions buffer
        self.actions = torch.zeros(
            (self.num_envs, self.cfg.num_actions), device=self.device
        )

    # =====================================================
    # RL interface
    # =====================================================

    def _pre_physics_step(self, actions: torch.Tensor):
        """Process actions before physics step.

        Args:
            actions: Actions from policy [left_wheel, right_wheel] in range [-1, 1]
        """
        self.actions = actions.clone()

        # Scale actions to velocity limits
        scaled_actions = actions * self.cfg.max_linear_velocity

        # Set wheel velocities
        try:
            self._robot.set_joint_velocity_target(scaled_actions)
        except Exception:
            # Silently handle if robot doesn't support velocity control
            pass

    def _apply_action(self):
        """Apply actions to the robot (called by parent class)."""
        # We already applied actions in _pre_physics_step
        pass

    def _get_observations(self) -> Dict[str, torch.Tensor]:
        """Compute observations.

        Returns:
            Dictionary with "policy" key containing observations for each environment.
        """
        # Get robot state
        root_state = self._robot.data.root_state_w
        robot_positions = root_state[:, :3]
        robot_velocities = root_state[:, 7:10]
        robot_orientations = root_state[:, 3:7]

        # Compute relative goal position
        goal_vector = self.goal_positions - robot_positions
        goal_distance = torch.norm(goal_vector[:, :2], dim=-1, keepdim=True)

        # Compute goal angle in robot frame
        robot_yaw = self._compute_yaw_from_quat(robot_orientations)
        goal_angle = torch.atan2(goal_vector[:, 1], goal_vector[:, 0]) - robot_yaw
        goal_angle = torch.atan2(torch.sin(goal_angle), torch.cos(goal_angle))  # [-pi, pi]

        # Check for collisions
        if self._contact_sensor is not None:
            contact_forces = self._contact_sensor.data.net_forces_w
            collision_flag = (
                torch.norm(contact_forces, dim=-1) > 1.0
            ).float().unsqueeze(-1)
        else:
            collision_flag = torch.zeros((self.num_envs, 1), device=self.device)

        # Normalized time
        normalized_time = (
            self.episode_length_buf / self.max_episode_length
        ).unsqueeze(-1)

        # Compose observation vector
        obs = torch.cat(
            [
                robot_positions[:, :2],        # [0:2] x, y position
                robot_velocities[:, :2],       # [2:4] vx, vy velocity
                self.goal_positions[:, :2],    # [4:6] goal x, y
                goal_distance,                 # [6] distance to goal
                goal_angle.unsqueeze(-1),      # [7] angle to goal
                collision_flag,                # [8] collision flag
                normalized_time,               # [9] time
            ],
            dim=-1,
        )

        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards.

        Returns:
            Reward tensor for each environment.
        """
        # Get current robot position
        robot_positions = self._robot.data.root_state_w[:, :3]

        # Distance to goal
        goal_vector = self.goal_positions - robot_positions
        distance_to_goal = torch.norm(goal_vector[:, :2], dim=-1)

        # Reward for moving towards goal
        progress = self.previous_distance_to_goal - distance_to_goal
        progress_reward = progress * self.cfg.reward_scales["forward"]

        # Update previous distance
        self.previous_distance_to_goal = distance_to_goal.clone()

        # Penalty for angular motion (encourage straight movement)
        angular_velocity = self._robot.data.root_ang_vel_w[:, 2]  # yaw rate
        angular_penalty = (
            torch.abs(angular_velocity) * self.cfg.reward_scales["angular"]
        )

        # Collision penalty
        if self._contact_sensor is not None:
            contact_forces = self._contact_sensor.data.net_forces_w
            collision_penalty = (
                torch.norm(contact_forces, dim=-1) > 1.0
            ).float() * self.cfg.reward_scales["collision"]
        else:
            collision_penalty = torch.zeros(self.num_envs, device=self.device)

        # Time penalty (encourage efficiency)
        time_penalty = (
            torch.ones(self.num_envs, device=self.device)
            * self.cfg.reward_scales["time_penalty"]
        )

        # Goal reached bonus
        goal_reached = (distance_to_goal < self.cfg.goal_distance_threshold).float()
        goal_bonus = goal_reached * 50.0

        # Total reward
        total_reward = (
            progress_reward
            + angular_penalty
            + collision_penalty
            + time_penalty
            + goal_bonus
        )

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute done flags.

        Returns:
            Tuple of (terminated, truncated) tensors.
        """
        # Get current robot position
        robot_positions = self._robot.data.root_state_w[:, :3]

        # Goal reached
        distance_to_goal = torch.norm(
            self.goal_positions[:, :2] - robot_positions[:, :2], dim=-1
        )
        goal_reached = distance_to_goal < self.cfg.goal_distance_threshold

        # Collision
        if self._contact_sensor is not None:
            contact_forces = self._contact_sensor.data.net_forces_w
            collision = torch.norm(contact_forces, dim=-1) > 10.0
        else:
            collision = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )

        # Time limit
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Terminated if goal reached or collision
        terminated = goal_reached | collision

        # Truncated if time limit reached
        truncated = time_out

        return terminated, truncated

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset specified environments.

        Args:
            env_ids: Indices of environments to reset. If None, reset all.
        """
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Reset robot to default state
        super()._reset_idx(env_ids)

        # Randomize goal positions (within 10 meters)
        self.goal_positions[env_ids, 0] = (
            torch.rand(len(env_ids), device=self.device) * 20.0 - 10.0
        )
        self.goal_positions[env_ids, 1] = (
            torch.rand(len(env_ids), device=self.device) * 20.0 - 10.0
        )
        self.goal_positions[env_ids, 2] = 0.0  # Ground level

        # Reset tracking
        robot_positions = self._robot.data.root_state_w[env_ids, :3]
        distance_to_goal = torch.norm(
            self.goal_positions[env_ids, :2] - robot_positions[:, :2], dim=-1
        )
        self.previous_distance_to_goal[env_ids] = distance_to_goal

        # Clear actions
        self.actions[env_ids] = 0.0

    def _compute_yaw_from_quat(self, quat: torch.Tensor) -> torch.Tensor:
        """Compute yaw angle from quaternion.

        Args:
            quat: Quaternion tensor [N, 4] (w, x, y, z)

        Returns:
            Yaw angles [N]
        """
        # Extract yaw from quaternion
        # yaw = atan2(2(w*z + x*y), 1 - 2(y^2 + z^2))
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        yaw = torch.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )
        return yaw

    # ========================================
    # Helper methods for external use
    # ========================================

    def sample_forward_action(self) -> torch.Tensor:
        """Sample a forward action (both wheels same speed).

        Returns:
            Action tensor for moving forward.
        """
        forward_speed = 0.5
        return torch.ones((self.num_envs, 2), device=self.device) * forward_speed

    def sample_turn_action(self) -> torch.Tensor:
        """Sample a turning action (differential wheel speeds).

        Returns:
            Action tensor for turning.
        """
        actions = torch.zeros((self.num_envs, 2), device=self.device)
        actions[:, 0] = 0.3   # Left wheel
        actions[:, 1] = -0.3  # Right wheel (opposite direction)
        return actions
