import glob
import os
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from sklearn.preprocessing import StandardScaler

from src.utility.utils import utility, pca_analysis, kmeans_analysis, iterate_dataframe

class BDBC_AllNumeric(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, n_actions, n_obs_space, data_dir, reward_type,
                 energy_coef=0.8, packet_coef=0.2, latency_coef=0.0,
                 energy_thresh=13.2, packet_thresh=15, latency_thresh=10, timesteps=216, setpoint_thresh=0.1):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(n_actions)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=np.full(shape=n_obs_space, fill_value=-15, dtype=np.float32),
                                            high=np.full(
                                                shape=n_obs_space, fill_value=100, dtype=np.float32),
                                            dtype=float)
        self.data_dir = data_dir
        self.info = {}
        self.data = iterate_dataframe(pd.read_csv(os.path.join(self.data_dir, 'BDBC_AllNumeric.csv')))
        self.reward = 0
        self.reward_type = reward_type()
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

    def step(self, action):
        self.obs = self.data.iloc[action].iloc[-1]

        self.reward = self.reward_type.get_reward(ut=0, energy_consumption=self.obs,
                                                  packet_loss=0, latency=0,
                                                  energy_thresh=39.0, packet_thresh=self.packet_thresh,
                                                  latency_thresh=self.latency_thresh, 
                                                  setpoint_thresh=self.setpoint_thresh)
        self.time_steps -= 1
        if self.time_steps == 0:
            self.terminated = True
            self.truncated = True

        return self.obs, self.reward, self.terminated, self.truncated, self.info

    def reset(self, seed=None, options=None):
        self.time_steps = self.init_time_steps
        self.terminated = False
        self.truncated = False
        try:
            self.df = self.data.sample(n=1)
            # scaler = StandardScaler()
            # X = pd.DataFrame(scaler.fit_transform(
            #     self.df['features'].values.tolist()))
            # X, _ = pca_analysis(X)
            # self.info['clusters'] = kmeans_analysis(X)
        except Exception as e:
            print(e)
            self.data = iterate_dataframe(pd.read_csv(os.path.join(self.data_dir, 'BDBC_AllNumeric.csv')))
            self.df = next(self.data)
            # scaler = StandardScaler()
            # X = pd.DataFrame(scaler.fit_transform(
            #     self.df['features'].values.tolist()))
            # X, _ = pca_analysis(X)
            # self.info['clusters'] = kmeans_analysis(X)
            
        self.obs = self.df.iloc[:, -1].iloc[0]
        return self.obs, self.info

    def render(self):
        ...

    def close(self):
        ...
