import os
import shutil
import traceback
import subprocess
import webbrowser
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import streamlit as st

from src.utility.utils import display_error_message, get_env_parameters
from src.utility.constantsv1 import MAX_EPISODE_STEPS
from train import stable_dqn
from visualization import load_and_plot_data

warnings.filterwarnings('ignore')

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
        "DQN": ["MlpPolicy", 'SoftmaxDQNPolicy', 'BoltzmannDQNPolicy', 'UCBDQNPolicy', 'UCB1TUNEDDQNPolicy', 'BayesianUCBDQNPolicy', 'SoftmaxUCBDQNPolicy'],
        "PPO": ["MlpPolicy", "CnnPolicy", "MultiInputPolicy"],
        "A2C": ["MlpPolicy", "CnnPolicy", "MultiInputPolicy"],
        "HER_DQN": ["MultiInputPolicy"]
    }
    policy = st.selectbox("Policy", policy_options[algo_name])
    
    lr = st.text_input("Learning Rate", "0.0001,0.001,0.01,0.1")
    exploration_fraction = st.text_input("Exploration Fraction (for DQN)", "0.1,0.2,0.4,0.6")
    warmup_count = st.number_input("Warmup Count", min_value=0, value=1024)
    gamma = st.number_input("Gamma", min_value=0.0, max_value=1.0, value=0.99)
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
        from_cycles = st.number_input("From cycles", min_value=0, value=0)
        to_cycles = st.number_input("To cycles", min_value=0, value=1505)
        if st.button("Load and Plot Data"):
            load_and_plot_data(st, data_dir, from_cycles, to_cycles) 

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

if __name__ == "__main__":
    main()
