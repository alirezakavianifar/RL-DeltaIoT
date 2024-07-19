import os
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from src.utility.utils import utility, iterate_dataframe

class BDBC_AllNumeric(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, n_actions, n_obs_space, data_dir,
                 timesteps=216, performance_thresh=0.1, reward_scaling=True):
        super().__init__()
        # Define action and observation space
        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Box(
            low=np.full(shape=n_obs_space, fill_value=-15, dtype=np.float32),
            high=np.full(shape=n_obs_space, fill_value=100, dtype=np.float32),
            dtype=float
        )
        self.data_dir = data_dir
        self.info = {}
        self.data = iterate_dataframe(pd.read_csv(os.path.join(self.data_dir, 'BDBC_AllNumeric.csv')))
        self.reward = 0
        self.terminated = False
        self.truncated = False
        self.init_time_steps = timesteps
        self.performance_thresh = performance_thresh
        self.reward_scaling = reward_scaling

    def step(self, action):
        self.obs = self.data.iloc[action].iloc[-1]

        self.reward = self.get_reward(
            performance_score=self.obs, 
        )
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
        except Exception as e:
            print(e)
            self.data = iterate_dataframe(pd.read_csv(os.path.join(self.data_dir, 'BDBC_AllNumeric.csv')))
            self.df = next(self.data)
            
        self.obs = self.df.iloc[:, -1].iloc[0]
        return self.obs, self.info

    def render(self):
        pass

    def close(self):
        pass

    def get_reward(self, performance_score):
        if performance_score >= self.performance_thresh:
            return 1.0
        else:
            reward = performance_score / self.performance_thresh
            if self.reward_scaling:
                reward = reward ** 2  
            return reward
