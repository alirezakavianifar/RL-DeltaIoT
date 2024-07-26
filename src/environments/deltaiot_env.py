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

        # Initialize storage for energy consumption history
        self.energy_history = []
        self.packet_history = []
        self.latency_history = []

        # Initialize min and max values
        self.initialize_metrics()

        # self.min_energy = float(12.90)
        # self.max_energy = float(13.80)
        # self.min_packet_loss = float(0.0)
        # self.max_packet_loss = float(100.0)
        # self.min_latency = float(0.0)
        # self.max_latency = float(100.0)

    def step(self, action):
        self.obs = self.df.iloc[action][['energyconsumption', 'packetloss', 'latency']].to_numpy(dtype=float).flatten()

        energy_consumption = self.obs[0].flatten()
        packet_loss = self.obs[1].flatten()
        latency = self.obs[2].flatten()

        # Update reward estimates
        self.update_reward_estimates(energy_consumption, packet_loss, latency)

        ut = utility(self.energy_coef, self.packet_coef, self.latency_coef,
                     energy_consumption, packet_loss, latency)

        raw_reward = self.reward_type.get_reward(util=ut, energy_consumption=energy_consumption,
                                                  packet_loss=packet_loss, latency=latency,
                                                  energy_thresh=self.energy_thresh, packet_thresh=self.packet_thresh,
                                                  latency_thresh=self.latency_thresh,
                                                  setpoint_thresh=self.setpoint_thresh, goal=self.goal)

        # Normalize the reward
        self.reward = self.reward_shape(raw_reward)

        # self.reward = normalized_reward
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
        self.initialize_metrics()
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
        self.obs = self.df.iloc[rand_num][['energyconsumption', 'packetloss', 'latency']].to_numpy(dtype=float).flatten()
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

    def initialize_metrics(self):
        self.min_energy = float('inf')
        self.max_energy = float('-inf')
        self.min_packet_loss = float('inf')
        self.max_packet_loss = float('-inf')
        self.min_latency = float('inf')
        self.max_latency = float('-inf')

    def reward_shape(self, data):
        
        energy_consumption, packet_loss, latency = data

        if self.goal in ['energy', 'packet', 'latency']:
            norm_energy = (self.max_energy - energy_consumption) / (self.max_energy - self.min_energy) if self.max_energy != self.min_energy else 1
            norm_packet_loss = (self.max_packet_loss - packet_loss) / (self.max_packet_loss - self.min_packet_loss) if self.max_packet_loss != self.min_packet_loss else 1
            norm_latency = (self.max_latency - latency) / (self.max_latency - self.min_latency) if self.max_latency != self.min_latency else 1

            # Combine normalized metrics
            reward = (norm_energy * self.energy_coef) \
                + (norm_packet_loss * self.packet_coef) \
                + (norm_latency * self.latency_coef)

        elif self.goal in ['energy_thresh', 'packet_thresh', 'latency_thresh']:
            reward = (energy_consumption * self.energy_coef) \
                + (packet_loss * self.packet_coef) \
                + (latency * self.latency_coef)

        return reward
    
    def reward_shape_with_variability(self, raw_reward):
        energy_consumption, packet_loss, latency = raw_reward

        # Add current energy consumption to history
        self.energy_history.append(energy_consumption)
        self.packet_history.append(packet_loss)
        self.latency_history.append(latency)
        if len(self.energy_history) > 1:
            # Calculate standard deviation of energy consumption
            energy_std = np.std(self.energy_history)
            packet_std = np.std(self.packet_history)
            latency_std = np.std(self.latency_history)
        else:
            energy_std = 0
            packet_std = 0
            latency_std = 0

        # Define weights for the reward function
        alpha = 1
        beta = 0.5

        # Calculate the reward
        reward_energy = -alpha * (energy_consumption - self.min_energy) - beta * energy_std
        reward_packet = -alpha * (packet_loss - self.min_packet_loss) - beta * packet_std
        reward_latency = -alpha * (latency - self.min_latency) - beta * latency_std
        return reward_energy, reward_packet, reward_latency
    


    def update_reward_estimates(self, energy_consumption, packet_loss, latency):
        self.min_energy = min(self.min_energy, energy_consumption)
        self.max_energy = max(self.max_energy, energy_consumption)
        self.min_packet_loss = min(self.min_packet_loss, packet_loss)
        self.max_packet_loss = max(self.max_packet_loss, packet_loss)
        self.min_latency = min(self.min_latency, latency)
        self.max_latency = max(self.max_latency, latency)

    def compute_reward(self, achieved_goal, desired_goal, info):
        reward = np.apply_along_axis(self.reward_shape, 1, achieved_goal, desired_goal)
        return reward
