# policies/base_policy.py

"""Base policy interface for NaviSim navigation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch


@dataclass
class PolicyConfig:
    """Base configuration for policies."""

    name: str = "base_policy"
    device: str = "cuda:0"
    deterministic: bool = True  # Use deterministic actions (for evaluation)

    # Optional metadata
    description: str = ""
    version: str = "1.0"


class BasePolicy(ABC):
    """
    Base policy interface for navigation agents.

    All policies (learned, scripted, hybrid) should inherit from this class
    and implement the required methods.
    """

    def __init__(self, config: PolicyConfig, num_envs: int, observation_space: Any, action_space: Any):
        """
        Initialize the policy.

        Args:
            config: Policy configuration.
            num_envs: Number of parallel environments.
            observation_space: Environment observation space.
            action_space: Environment action space.
        """
        self.config = config
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = torch.device(config.device)

        # Policy state
        self._step_count = 0
        self._episode_count = torch.zeros(num_envs, dtype=torch.long, device=self.device)

    @abstractmethod
    def compute_action(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute actions given observations.

        Args:
            observations: Dictionary of observations with 'policy' key.
                         Shape: (num_envs, obs_dim)

        Returns:
            Actions tensor. Shape: (num_envs, action_dim)
        """
        raise NotImplementedError

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> None:
        """
        Reset policy state for specified environments.

        Args:
            env_ids: Indices of environments to reset. If None, reset all.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        self._episode_count[env_ids] = 0

    def update(self, observations: Dict[str, torch.Tensor], actions: torch.Tensor,
               rewards: torch.Tensor, dones: torch.Tensor, infos: Dict[str, Any]) -> None:
        """
        Update policy with environment feedback (optional, for online learning).

        Args:
            observations: Current observations.
            actions: Actions taken.
            rewards: Rewards received.
            dones: Episode termination flags.
            infos: Additional information.
        """
        # Increment step counter
        self._step_count += 1

        # Increment episode counter for done environments
        done_indices = torch.where(dones)[0]
        if len(done_indices) > 0:
            self._episode_count[done_indices] += 1

    def save(self, path: str) -> None:
        """
        Save policy to file.

        Args:
            path: Path to save policy.
        """
        pass  # Optional, implement in subclasses if needed

    def load(self, path: str) -> None:
        """
        Load policy from file.

        Args:
            path: Path to load policy from.
        """
        pass  # Optional, implement in subclasses if needed

    def get_info(self) -> Dict[str, Any]:
        """
        Get policy information and statistics.

        Returns:
            Dictionary with policy info.
        """
        return {
            "name": self.config.name,
            "description": self.config.description,
            "version": self.config.version,
            "step_count": self._step_count,
            "total_episodes": self._episode_count.sum().item(),
            "device": str(self.device),
        }

    def __repr__(self) -> str:
        """String representation of policy."""
        return f"{self.__class__.__name__}(name='{self.config.name}', device='{self.device}')"