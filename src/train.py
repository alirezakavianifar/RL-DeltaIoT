import os
import glob
from concurrent.futures import ProcessPoolExecutor

from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.wrappers.time_limit import TimeLimit
from stable_baselines3.common.monitor import Monitor

from src.rl_agents import train_a2c, train_dqn, train_her_dqn, train_ppo
from src.environments.deltaiot_env import DeltaIotEnv
from src.environments.env_helpers import RewardStrategy
from src.environments.bdbc_allNumeric_env import BDBC_AllNumeric

from src.utility.utils import display_error_message

def setup_env(env_name, algo_name, additional_params):
    total_timesteps = additional_params.get('total_timesteps', 512)
    n_actions = additional_params.get('n_actions', 216)
    n_obs_space = additional_params.get('n_obs_space', 3)
    max_episode_steps = additional_params.get('max_episode_steps', 216)
    use_dict_obs_space = additional_params.get('use_dict_obs_space', False)
    
    if env_name in ['DeltaIoTv1', 'DeltaIoTv2']:
        env = Monitor(TimeLimit(DeltaIotEnv(
            n_actions=n_actions, 
            n_obs_space=n_obs_space, 
            data_dir=glob.glob(os.path.join('data', env_name, 'train', "*.json")),
            reward_type=RewardStrategy(additional_params.get('strategy_type')), 
            goal=additional_params.get('goal'),
            energy_coef=additional_params.get('energy_coef', 0.0),
            packet_coef=additional_params.get('packet_coef', 0.0), 
            latency_coef=additional_params.get('latency_coef', 0.0), 
            energy_thresh=additional_params.get('energy_thresh', 12.95), 
            packet_thresh=additional_params.get('packet_thresh', 10.0), 
            latency_thresh=additional_params.get('latency_thresh', 5.0), 
            timesteps=total_timesteps, 
            setpoint_thresh=additional_params.get('setpoint_thresh', 0.0),
            use_dict_obs_space=use_dict_obs_space
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

def stable_dqn(env_name, algo_name, goal, policy, lr, exploration_fraction, gamma, batch_size, total_timesteps, chkpt_dir, log_path, warmup_count, eps_min, epsilon, num_pulls, additional_params, execution_mode, num_processes=None):
    lrs = lr.split(',')
    exploration_fractions = exploration_fraction.split(',')
    tasks = []

    if execution_mode == 'Parallel':
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            for lr in lrs:
                for exploration_fraction in exploration_fractions:
                    tasks.append(executor.submit(train_model, env_name, algo_name, goal, policy, float(lr), float(exploration_fraction), gamma, batch_size, total_timesteps, chkpt_dir, log_path, warmup_count, eps_min, epsilon, num_pulls, additional_params))
        
        for task in tasks:
            try:
                task.result()
            except Exception as e:
                display_error_message(e, "Parallel Execution")
                break

    else:
        for lr in lrs:
            try:
                for exploration_fraction in exploration_fractions:
                    try:
                        train_model(env_name, algo_name, goal, policy, float(lr), float(exploration_fraction), gamma, batch_size, total_timesteps, chkpt_dir, log_path, warmup_count, eps_min, epsilon, num_pulls, additional_params)
             
                    except Exception as e:
                        display_error_message(e, "Serial Execution")
                        raise Exception
            except:
                break

def train_model(env_name, algo_name, goal, policy, lr, exploration_fraction, gamma, batch_size, total_timesteps, chkpt_dir, log_path, warmup_count, eps_min, epsilon, num_pulls, additional_params):
    try:
        env = setup_env(env_name, algo_name, additional_params)
        log_path_base = os.path.join(log_path, f"algo={algo_name}-goal={goal}-env={env_name}-policy={policy}-lr={lr}-batch_size={batch_size}-gamma={gamma}-total_timesteps={total_timesteps}-exploration_fraction={exploration_fraction}")
        checkpoint_callback = CheckpointCallback(save_freq=1, save_path=chkpt_dir, name_prefix=f"algo={algo_name}-goal={goal}-env={env_name}-policy={policy}-lr={lr}-batch_size={batch_size}-gamma={gamma}-total_timesteps={total_timesteps}-exploration_fraction={exploration_fraction}")
        eval_callback = EvalCallback(env, callback_on_new_best=checkpoint_callback, eval_freq=int(total_timesteps / 20), verbose=1, n_eval_episodes=5)

        if algo_name == "DQN":
            train_dqn(env, policy, lr, exploration_fraction, gamma, batch_size, total_timesteps, warmup_count, eps_min, epsilon, num_pulls, additional_params.get('setpoint_thresh'), log_path_base, checkpoint_callback, eval_callback)
        elif algo_name == "PPO":
            train_ppo(env, policy, lr, gamma, batch_size, total_timesteps, log_path_base, checkpoint_callback, eval_callback)
        elif algo_name == "A2C":
            train_a2c(env, policy, lr, gamma, total_timesteps, log_path_base, checkpoint_callback, eval_callback)
        elif algo_name == "HER_DQN":
            train_her_dqn(env, policy, lr, exploration_fraction, gamma, batch_size, total_timesteps, warmup_count, eps_min, epsilon, num_pulls, additional_params.get('setpoint_thresh'), log_path_base, checkpoint_callback, eval_callback, additional_params)
    except Exception as e:
        display_error_message(e, "Model Training")
        raise Exception
