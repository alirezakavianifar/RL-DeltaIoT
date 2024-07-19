import os
import glob
import numpy as np
import streamlit as st
import shutil
from collections import defaultdict
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.wrappers.time_limit import TimeLimit
from stable_baselines3.common.monitor import Monitor
from src.utility.constantsv1 import MAX_EPISODE_STEPS
from src.environments.deltaiot_env import DeltaIotEnv
from src.environments.env_helpers import RewardStrategy
from src.environments.bdbc_allNumeric_env import BDBC_AllNumeric
from src.drl4sao.custom_dqn.eps_dec_types import EpsDecTypeTwo
from src.drl4sao.stable_algos.custom_stable import CustomDQN
from src.drl4sao.stable_algos.custom_policies.bayesian_ucb import BayesianUCB
from src.drl4sao.stable_algos.rl_algos.a2c import A2C
from concurrent.futures import ProcessPoolExecutor

def main():
    st.title("RL Environment Configuration")

    # Input parameters
    env_name = st.selectbox("Environment Name", ['DeltaIoTv1', 'DeltaIoTv2', 'BDBC_AllNumeric'])
    # Additional parameters for DeltaIoT environments
    additional_params = {}
    additional_params['max_episode_steps'] = MAX_EPISODE_STEPS[env_name]
    if env_name in ['DeltaIoTv1', 'DeltaIoTv2']:
        strategy_type = st.selectbox("Strategy Type", ['min', 'multi'])
        additional_params['strategy_type'] = strategy_type
        if strategy_type == 'multi':
            additional_params['energy_coef'] = st.number_input("Energy Coefficient", min_value=0.0, max_value=1.0, value=0.1)
            additional_params['packet_coef'] = st.number_input("Packet Coefficient", min_value=0.0, max_value=1.0, value=0.8)
            additional_params['latency_coef'] = st.number_input("Latency Coefficient", min_value=0.0, max_value=1.0, value=0.8)
            additional_params['setpoint_thresh'] = {
                'lower_bound': st.number_input("Setpoint Threshold Lower Bound", 12.8),
                'upper_bound': st.number_input("Setpoint Threshold Upper Bound", 13.0)
            }
        if strategy_type == 'min':
            goal = st.selectbox("Goal", ['energy', 'packet_loss', 'latency'])
            additional_params['goal'] = goal

    algo_name = st.selectbox("Algorithm Name", ['DQN', 'PPO', 'A2C'])
    policy = st.selectbox("Policy", ["MlpPolicy", 'BoltzmannPolicy', 'SoftmaxDQNPolicy', 'BoltzmannDQNPolicy', 'UCBDQNPolicy',
                                    'UCB1TUNEDDQNPolicy', 'BayesianUCBDQNPolicy', 'SoftmaxUCBDQNPolicy', 'SoftmaxUCBAdaptiveDQNPolicy'])
    lr = st.text_input("Learning Rate", "0.0001,0.001,0.01,0.1")
    exploration_fraction = st.text_input("Exploration Fraction (for DQN)", "0.1,0.2,0.4,0.6")
    gamma = st.number_input("Gamma", min_value=0.0, max_value=1.0,  value=0.99)
    batch_size = st.number_input("Batch Size", 64)
    total_timesteps = st.number_input("Total Timesteps", 15000)
    max_episode_steps = st.number_input("Max Episode Steps", additional_params['max_episode_steps'])
    chkpt_dir = st.text_input("Checkpoint Directory", "models")
    log_path = st.text_input("Log Path", "logs")
    execution_mode = st.selectbox("Execution Mode", ['Serial', 'Parallel'])

    # Number of processes input for parallel execution
    num_processes = None
    if execution_mode == 'Parallel':
        num_processes = st.number_input("Number of Processes", min_value=1, value=4)

    # Option to delete the contents of logs and models folders
    if st.button("Clear Logs and Models"):
        try:
            if os.path.exists(log_path):
                shutil.rmtree(log_path)
                st.success(f"Successfully deleted the contents of {log_path}")
            if os.path.exists(chkpt_dir):
                shutil.rmtree(chkpt_dir)
                st.success(f"Successfully deleted the contents of {chkpt_dir}")
        except Exception as e:
            st.error(f"An error occurred while deleting folders: {e}")

    warmup_count = 100
    eps_min = 0.001
    epsilon = 1.0
    num_pulls = np.zeros(100)
    network_layers = [150, 120, 100, 50, 25]

    # Set default observation and action spaces based on environment name
    n_obs_space, n_actions = get_env_parameters(env_name)

    if st.button("Train Model"):
        try:
            additional_params.update({
                'env_name': env_name,
                'total_timesteps': total_timesteps,
                'n_actions': n_actions,
                'n_obs_space': n_obs_space,
                'max_episode_steps': max_episode_steps
            })
            stable_dqn(env_name, algo_name, policy, lr, exploration_fraction, gamma, batch_size, total_timesteps, chkpt_dir, log_path, warmup_count, eps_min, epsilon, num_pulls, additional_params, execution_mode, num_processes)
            st.success("Model training completed")
        except PermissionError as e:
            st.error(f"PermissionError: {e}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

def get_env_parameters(env_name):
    if env_name in ['DeltaIoTv1', 'DeltaIoTv2']:
        n_obs_space = 3
        n_actions = 216 if env_name == 'DeltaIoTv1' else 4096
    elif env_name == 'BDBC_AllNumeric':
        n_obs_space = 1
        n_actions = 100
    return n_obs_space, n_actions

def setup_env(env_name, additional_params):
    total_timesteps = additional_params.get('total_timesteps', 512)
    n_actions = additional_params.get('n_actions', 216)
    n_obs_space = additional_params.get('n_obs_space', 3)
    max_episode_steps = additional_params.get('max_episode_steps', 216)
    
    if env_name in ['DeltaIoTv1', 'DeltaIoTv2']:
        env = Monitor(TimeLimit(DeltaIotEnv(
            data_dir=glob.glob(os.path.join(f'data/{env_name}/train', "*.json")),
            timesteps=total_timesteps, 
            n_actions=n_actions, 
            n_obs_space=n_obs_space,
            reward_type=RewardStrategy(additional_params.get('strategy_type')), 
            goal=additional_params.get('goal'),
            energy_coef=additional_params.get('energy_coef', 0.0),
            packet_coef=additional_params.get('packet_coef', 0.0), 
            latency_coef=additional_params.get('latency_coef', 0.0), 
            packet_thresh=10.0, 
            latency_thresh=5.0, 
            energy_thresh=12.9, 
            setpoint_thresh=additional_params.get('setpoint_thresh', 0.0)
        ), max_episode_steps=max_episode_steps))
    elif env_name == 'BDBC_AllNumeric':
        env = Monitor(TimeLimit(BDBC_AllNumeric(
            data_dir='data/BDBC_AllNumeric',
            timesteps=100,
            n_actions=n_actions, 
            n_obs_space=n_obs_space, 
            performance_thresh=39.0
        ), max_episode_steps=max_episode_steps))
    
    return DummyVecEnv([lambda: env])

def stable_dqn(env_name, algo_name, policy, lr, exploration_fraction, gamma, batch_size, total_timesteps, chkpt_dir, log_path, warmup_count, eps_min, epsilon, num_pulls, additional_params, execution_mode, num_processes=None):
    lrs = lr.split(',')
    exploration_fractions = exploration_fraction.split(',')
    tasks = []

    if execution_mode == 'Parallel':
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            for lr in lrs:
                for exploration_fraction in exploration_fractions:
                    tasks.append(executor.submit(train_model, env_name, algo_name, policy, float(lr), float(exploration_fraction), gamma, batch_size, total_timesteps, chkpt_dir, log_path, warmup_count, eps_min, epsilon, num_pulls, additional_params))
        
        for task in tasks:
            task.result()

    else:
        for lr in lrs:
            for exploration_fraction in exploration_fractions:
                train_model(env_name, algo_name, policy, float(lr), float(exploration_fraction), gamma, batch_size, total_timesteps, chkpt_dir, log_path, warmup_count, eps_min, epsilon, num_pulls, additional_params)

def train_model(env_name, algo_name, policy, lr, exploration_fraction, gamma, batch_size, total_timesteps, chkpt_dir, log_path, warmup_count, eps_min, epsilon, num_pulls, additional_params):
    env = setup_env(env_name, additional_params)
    log_path_base = os.path.join(log_path, f"{env_name}-policy={policy}-lr={lr}-batch_size={batch_size}-gamma={gamma}-total_timesteps={total_timesteps}-exploration_fraction={exploration_fraction}")
    checkpoint_callback = CheckpointCallback(save_freq=1, save_path=chkpt_dir, name_prefix=f"{env_name}-policy={policy}-lr={lr}-batch_size={batch_size}-gamma={gamma}-total_timesteps={total_timesteps}-exploration_fraction={exploration_fraction}")
    eval_callback = EvalCallback(env, callback_on_new_best=checkpoint_callback, eval_freq=int(total_timesteps / 20), verbose=1, n_eval_episodes=5)

    if algo_name == "DQN":
        train_dqn(env, policy, lr, exploration_fraction, gamma, batch_size, total_timesteps, warmup_count, eps_min, epsilon, num_pulls, additional_params.get('setpoint_thresh'), log_path_base, checkpoint_callback, eval_callback)
    elif algo_name == "PPO":
        train_ppo(env, policy, lr, gamma, batch_size, total_timesteps, log_path_base, checkpoint_callback, eval_callback)
    elif algo_name == "A2C":
        train_a2c(env, policy, lr, gamma, total_timesteps, log_path_base, checkpoint_callback, eval_callback)

def train_dqn(env, policy, lr, exploration_fraction, gamma, batch_size, total_timesteps, warmup_count, eps_min, epsilon, num_pulls, setpoint_thresh, log_path, checkpoint_callback, eval_callback):
    bayesian_ucb = BayesianUCB(env.action_space.n)
    model = CustomDQN(policy,
                      env,
                      learning_rate=lr,
                      learning_starts=warmup_count,
                      batch_size=batch_size,
                      gamma=gamma,
                      exploration_initial_eps=epsilon,
                      exploration_final_eps=eps_min,
                      exploration_fraction=exploration_fraction,
                      replay_buffer_class=None,
                      tensorboard_log=log_path,
                      device='cuda',
                      verbose=1,
                      deterministic=False,
                      num_pulls=num_pulls,
                      setpoint_thresh=setpoint_thresh,
                      bayesian_ucb=bayesian_ucb)
    model.learn(total_timesteps=total_timesteps, log_interval=1, callback=eval_callback)

def train_ppo(env, policy, lr, gamma, batch_size, total_timesteps, log_path, checkpoint_callback, eval_callback):
    model = PPO(policy,
                env,
                learning_rate=lr,
                batch_size=batch_size,
                gamma=gamma,
                tensorboard_log=log_path,
                device='cuda',
                verbose=1)
    model.learn(total_timesteps=total_timesteps, log_interval=1, callback=eval_callback)

def train_a2c(env, policy, lr, gamma, total_timesteps, log_path, checkpoint_callback, eval_callback):
    model = A2C(policy,
                env,
                learning_rate=lr,
                gamma=gamma,
                tensorboard_log=log_path,
                device='cuda',
                verbose=1)
    model.learn(total_timesteps=total_timesteps, log_interval=1, callback=eval_callback)

if __name__ == "__main__":
    main()
