import warnings
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from src.drl4sao.stable_algos.rl_algos.dqn import DQN
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_linear_fn, get_parameters_by_name, polyak_update
from stable_baselines3.dqn.policies import CnnPolicy, DQNPolicy, MlpPolicy, MultiInputPolicy, QNetwork
from src.drl4sao.stable_algos.custom_policies.policies import SoftmaxDQNPolicy, BoltzmannDQNPolicy


class CustomDQN(DQN):
    def __init__(
        self,
        policy: Union[str, Type[DQNPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 50000,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 4,
        gradient_steps: int = 1,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        init_setup_model: bool = True,
        deterministic: bool = True,
        temprature: float = 1.0
    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            optimize_memory_usage=optimize_memory_usage,
        )

        self.deterministic = deterministic
        self.temprature = temprature

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Overrides the base_class predict function to include epsilon-greedy exploration.

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """

        if type(self.policy) == MlpPolicy:
            if not deterministic and np.random.rand() < self.exploration_rate:
                if self.policy.is_vectorized_observation(observation):
                    if isinstance(observation, dict):
                        n_batch = observation[next(
                            iter(observation.keys()))].shape[0]
                    else:
                        n_batch = observation.shape[0]
                    action = np.array([self.action_space.sample()
                                      for _ in range(n_batch)])
                else:
                    action = np.array(self.action_space.sample())
            else:
                action, state = self.policy.predict(
                    observation, state, episode_start, deterministic)
            return action, state
        elif type(self.policy) == SoftmaxDQNPolicy:
            q_values, state = self.policy.predict(
                observation, state, episode_start, deterministic)
            exp_values = np.exp(
                (q_values - np.max(q_values)) / self.exploration_rate)
            # Calculate softmax probabilities
            probabilities = exp_values / exp_values.sum()
            # Select action based on probabilities
            selected_action = np.array(
                [np.random.choice(len(q_values), p=probabilities)])
            return selected_action, state
        elif type(self.policy) == BoltzmannDQNPolicy:
            q_values, state = self.policy.predict(
                observation, state, episode_start, deterministic)
            # Ensure numerical stability by subtracting the maximum Q-value
            q_values = q_values - np.max(q_values)

            # Compute the probabilities using the Boltzmann distribution
            probabilities = np.exp(q_values / self.exploration_rate) / \
                np.sum(np.exp(q_values / self.exploration_rate))

            # Sample an action based on the probabilities
            selected_action = np.array(
                [np.random.choice(len(q_values), p=probabilities)])

            return selected_action, state
