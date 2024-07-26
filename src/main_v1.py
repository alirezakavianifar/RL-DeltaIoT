import warnings
warnings.filterwarnings('ignore')
import os
import glob
import numpy as np
import streamlit as st
import shutil
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.wrappers.time_limit import TimeLimit
from stable_baselines3.common.monitor import Monitor
from src.rl_agents import train_a2c, train_dqn, train_her_dqn, train_ppo
from src.utility.constantsv1 import MAX_EPISODE_STEPS
from src.environments.deltaiot_env import DeltaIotEnv
from src.environments.env_helpers import RewardStrategy
from src.environments.bdbc_allNumeric_env import BDBC_AllNumeric
from concurrent.futures import ProcessPoolExecutor
import traceback
import subprocess
import webbrowser
from src.utility.utils import plot_adaptation_spaces, plot_latency_vs_packet_loss, load_and_prepare_data

def display_error_message(e, context=""):
    error_message = f"""
    <div style="border:1px solid red; padding: 10px; border-radius: 5px; background-color: #ffe6e6;">
        <h4 style="color: red;">An error occurred</h4>
        <p><strong>Context:</strong> {context}</p>
        <p><strong>Error:</strong> {str(e)}</p>
        <details>
            <summary>Traceback</summary>
            <pre>{traceback.format_exc()}</pre>
        </details>
    </div>
    """
    st.markdown(error_message, unsafe_allow_html=True)


def main():
    st.title("RL Environment Configuration")

    env_name = st.selectbox("Environment Name", ['DeltaIoTv1', 'DeltaIoTv2', 'BDBC_AllNumeric'])
    additional_params = {'max_episode_steps': MAX_EPISODE_STEPS[env_name]}
    if env_name in ['DeltaIoTv1', 'DeltaIoTv2']:
        strategy_type = st.selectbox("Strategy Type", ['min', 'multi'])
        additional_params['strategy_type'] = strategy_type
        additional_params['energy_coef'] = additional_params['packet_coef'] = additional_params['latency_coef'] = 0
        if strategy_type == 'multi':
            additional_params['energy_coef'] = st.number_input("Energy Coefficient", min_value=0.0, max_value=1.0, value=0.1)
            additional_params['packet_coef'] = st.number_input("Packet Coefficient", min_value=0.0, max_value=1.0, value=0.8)
            additional_params['latency_coef'] = st.number_input("Latency Coefficient", min_value=0.0, max_value=1.0, value=0.8)
            additional_params['setpoint_thresh'] = {
                'lower_bound': st.number_input("Setpoint Threshold Lower Bound", 12.8),
                'upper_bound': st.number_input("Setpoint Threshold Upper Bound", 13.0)
            }
        elif strategy_type == 'min':
            goal = st.selectbox("Goal", ['energy', 'packet', 'latency', 
                                         'energy_thresh', 'packet_thresh', 'latency_thresh'])
            additional_params['goal'] = goal
            additional_params[f'{goal.split('_')[0]}_coef'] = 1.0

    algo_name = st.selectbox("Algorithm Name", ['DQN', 'PPO', 'A2C', 'HER_DQN'])
    policy_options = {
        "DQN": ["MlpPolicy", 'SoftmaxDQNPolicy', 'BoltzmannDQNPolicy', 'UCBDQNPolicy', 'UCB1TUNEDDQNPolicy', 'BayesianUCBDQNPolicy', 'SoftmaxUCBDQNPolicy', 'SoftmaxUCBAdaptiveDQNPolicy'],
        "PPO": ["MlpPolicy", "CnnPolicy", "MultiInputPolicy"],
        "A2C": ["MlpPolicy", "CnnPolicy", "MultiInputPolicy"],
        "HER_DQN": ["MultiInputPolicy"]
    }
    policy = st.selectbox("Policy", policy_options[algo_name])
    
    lr = st.text_input("Learning Rate", "0.0001,0.001,0.01,0.1")
    exploration_fraction = st.text_input("Exploration Fraction (for DQN)", "0.1,0.2,0.4,0.6")
    warmup_count = st.number_input("Warmup Count", min_value=0, value=1024)
    gamma = st.number_input("Gamma", min_value=0.0, max_value=1.0,  value=0.99)
    batch_size = st.number_input("Batch Size", 64)
    total_timesteps = st.number_input("Total Timesteps", min_value=2000, value=15000)
    max_episode_steps = st.number_input("Max Episode Steps", additional_params['max_episode_steps'])
    chkpt_dir = st.text_input("Checkpoint Directory", "models")
    log_path = st.text_input("Log Path", "logs")
    execution_mode = st.selectbox("Execution Mode", ['Serial', 'Parallel'])

    num_processes = None
    if execution_mode == 'Parallel':
        num_processes = st.number_input("Number of Processes", min_value=1, value=4)

    if st.button("Clear Logs and Models"):
        try:
            if os.path.exists(log_path):
                shutil.rmtree(log_path)
                st.success(f"Successfully deleted the contents of {log_path}")
            if os.path.exists(chkpt_dir):
                shutil.rmtree(chkpt_dir)
                st.success(f"Successfully deleted the contents of {chkpt_dir}")
        except Exception as e:
            display_error_message(e, "Clearing Logs and Models")

    if st.button("Launch TensorBoard"):
        try:
            log_dir = log_path if log_path else "logs"
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            tb_process = subprocess.Popen(['tensorboard', '--logdir', log_dir])
            tensorboard_url = "http://localhost:6006"
            st.markdown(f'[Open TensorBoard in a new tab]({tensorboard_url})', unsafe_allow_html=True)
            webbrowser.open_new_tab(tensorboard_url)
        except Exception as e:
            display_error_message(e, "Launching TensorBoard")

    eps_min = 0.001
    epsilon = 1.0
    network_layers = [150, 120, 100, 50, 25]
    n_obs_space, n_actions, use_dict_obs_space = get_env_parameters(env_name, algo_name)
    num_pulls = np.zeros(n_actions)

    if env_name == "DeltaIoTv1":
        st.header("Visualization for DeltaIoTv1 Environment")
        data_dir = st.text_input("Data Directory", r'D:\projects\gheibi-material\generated_data_by_deltaiot_simulation\under_drift_scenario')
        from_cycles = st.number_input("From cycles", min_value=0,  value=0)
        to_cycles = st.number_input("To cycles", min_value=0,  value=1505)
        if st.button("Load and Plot Data"):
            LST_PACKET, LST_ENERGY, LST_LATENCY, df = load_and_prepare_data(data_dir, from_cycles=from_cycles, to_cycles=to_cycles)
            plot_adaptation_spaces(st, df, from_cycles=from_cycles, to_cycles=to_cycles)
            plot_latency_vs_packet_loss(st, LST_LATENCY, LST_PACKET)

    if st.button("Train Model"):
        try:
            additional_params.update({
                'env_name': env_name,
                'total_timesteps': total_timesteps,
                'n_actions': n_actions,
                'n_obs_space': n_obs_space,
                'max_episode_steps': max_episode_steps,
                'use_dict_obs_space': use_dict_obs_space
            })
            stable_dqn(env_name, algo_name, policy, lr, exploration_fraction, gamma, batch_size, total_timesteps, chkpt_dir, log_path, warmup_count, eps_min, epsilon, num_pulls, additional_params, execution_mode, num_processes)
            st.success("Model training completed")
        except PermissionError as e:
            display_error_message(e, "Training Model (PermissionError)")
        except Exception as e:
            display_error_message(e, "Training Model")

def get_env_parameters(env_name, algo_name):
    if env_name in ['DeltaIoTv1', 'DeltaIoTv2']:
        n_obs_space = 3
        n_actions = 216 if env_name == 'DeltaIoTv1' else 4096 
    else:
        n_obs_space = 1
        n_actions = 100 
        use_dict_obs_space = False

    use_dict_obs_space = True if algo_name == 'HER_DQN' else False

    return n_obs_space, n_actions, use_dict_obs_space

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
                        train_model(env_name, algo_name, policy, float(lr), float(exploration_fraction), gamma, batch_size, total_timesteps, chkpt_dir, log_path, warmup_count, eps_min, epsilon, num_pulls, additional_params)
                    except Exception as e:
                        display_error_message(e, "Serial Execution")
                        raise Exception
            except:
                break

def train_model(env_name, algo_name, policy, lr, exploration_fraction, gamma, batch_size, total_timesteps, chkpt_dir, log_path, warmup_count, eps_min, epsilon, num_pulls, additional_params):
    try:
        env = setup_env(env_name, algo_name, additional_params)
        log_path_base = os.path.join(log_path, f"{env_name}-policy={policy}-lr={lr}-batch_size={batch_size}-gamma={gamma}-total_timesteps={total_timesteps}-exploration_fraction={exploration_fraction}")
        checkpoint_callback = CheckpointCallback(save_freq=1, save_path=chkpt_dir, name_prefix=f"{env_name}-policy={policy}-lr={lr}-batch_size={batch_size}-gamma={gamma}-total_timesteps={total_timesteps}-exploration_fraction={exploration_fraction}")
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

if __name__ == "__main__":
    main()
