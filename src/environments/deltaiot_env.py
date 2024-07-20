import glob
import os
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from sklearn.preprocessing import StandardScaler

from src.utility.utils import utility, return_next_item, pca_analysis, kmeans_analysis


class DeltaIotEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, n_actions, n_obs_space, data_dir, reward_type, goal=None,
                 energy_coef=0.8, packet_coef=0.2, latency_coef=0.0,
                 energy_thresh=13.2, packet_thresh=15, latency_thresh=10, timesteps=216, setpoint_thresh=0.1, use_dict_obs_space=False):
        super().__init__()
        self.action_space = spaces.Discrete(n_actions)
        self.use_dict_obs_space = use_dict_obs_space
        if self.use_dict_obs_space:
            self.observation_space = spaces.Dict({
                'observation': spaces.Box(low=np.full(shape=n_obs_space, fill_value=-15, dtype=np.float32),
                                          high=np.full(
                                              shape=n_obs_space, fill_value=100, dtype=np.float32),
                                          dtype=np.float32),
                'achieved_goal': spaces.Box(low=np.full(shape=n_obs_space, fill_value=-15, dtype=np.float32),
                                            high=np.full(
                                                shape=n_obs_space, fill_value=100, dtype=np.float32),
                                            dtype=np.float32),
                'desired_goal': spaces.Box(low=np.full(shape=n_obs_space, fill_value=-15, dtype=np.float32),
                                           high=np.full(
                                               shape=n_obs_space, fill_value=100, dtype=np.float32),
                                           dtype=np.float32)
            })
        else:
            self.observation_space = spaces.Box(low=np.full(shape=n_obs_space, fill_value=-15, dtype=np.float32),
                                                high=np.full(
                                                    shape=n_obs_space, fill_value=100, dtype=np.float32),
                                                dtype=np.float32)
        self.data_dir = data_dir
        self.info = {}
        self.data = return_next_item(self.data_dir, normalize=False)
        self.reward = 0
        self.reward_type = reward_type
        self.goal = goal
        self.terminated = False
        self.truncated = False
        self.energy_coef = energy_coef
        self.packet_coef = packet_coef
        self.latency_coef = latency_coef
        self.energy_thresh = energy_thresh
        self.packet_thresh = packet_thresh
        self.latency_thresh = latency_thresh
        self.init_time_steps = timesteps
        self.setpoint_thresh = setpoint_thresh
        self.time_steps = timesteps

    def step(self, action):
        self.obs = self.df.iloc[action][[
            'energyconsumption', 'packetloss', 'latency']].to_numpy(dtype=float).flatten()

        energy_consumption = self.obs[0].flatten()
        packet_loss = self.obs[1].flatten()
        latency = self.obs[2].flatten()

        ut = utility(self.energy_coef, self.packet_coef, self.latency_coef,
                     energy_consumption, packet_loss, latency)

        self.reward = self.reward_type.get_reward(util=ut, energy_consumption=energy_consumption,
                                                  packet_loss=packet_loss, latency=latency,
                                                  energy_thresh=self.energy_thresh, packet_thresh=self.packet_thresh,
                                                  latency_thresh=self.latency_thresh,
                                                  setpoint_thresh=self.setpoint_thresh, goal=self.goal)
        self.time_steps -= 1
        if self.time_steps == 0:
            self.terminated = True
            self.truncated = True

        obs_dict = {
            'observation': self.obs,
            'achieved_goal': self.obs,
            'desired_goal': self.her_goal
        }
        if self.use_dict_obs_space:
            return obs_dict, self.reward, self.terminated, self.truncated, self.info
        else:
            return self.obs, self.reward, self.terminated, self.truncated, self.info

    def reset(self, seed=None, options=None):
        self.time_steps = self.init_time_steps
        self.terminated = False
        self.truncated = False
        try:
            self.df = next(self.data).drop('verification_times', axis=1)
        except Exception as e:
            print(e)
            self.data = return_next_item(self.data_dir, normalize=False)
            self.df = next(self.data).drop('verification_times', axis=1)

        rand_num = np.random.randint(self.df.count().iloc[0])
        self.obs = self.df.iloc[rand_num][[
            'energyconsumption', 'packetloss', 'latency']].to_numpy(dtype=float).flatten()
        self.her_goal = self.obs  # For simplicity, use the same observation as the goal
        obs_dict = {
            'observation': self.obs,
            'achieved_goal': self.obs,
            'desired_goal': self.her_goal
        }
        if self.use_dict_obs_space:
            return obs_dict, self.info
        else:
            return self.obs, self.info

    def render(self):
        pass

    def close(self):
        pass

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Compute the reward for HER.

        Parameters:
        achieved_goal: The goal that was achieved during the episode
        desired_goal: The goal that we desired to achieve
        info: An info dictionary with additional information

        Returns:
        reward: The computed reward
        """
        # Example logic:
        reward = -np.linalg.norm(achieved_goal - desired_goal)
        return reward
