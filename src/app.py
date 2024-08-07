import os
import shutil
import subprocess
import webbrowser
import warnings

import numpy as np
import streamlit as st

from src.utility.utils import display_error_message, get_env_parameters, read_from_tensorboardlog, parse_log_directory_name
from src.utility.constantsv1 import MAX_EPISODE_STEPS
from train import stable_dqn
from visualization import load_and_plot_data

warnings.filterwarnings('ignore')


def main():
    st.title("RL Environment Configuration")

    st.sidebar.title("Configuration")
    st.sidebar.header("Environment and Algorithm Settings")
    
    env_name = st.sidebar.selectbox("Environment Name", ['DeltaIoTv1', 'DeltaIoTv2', 'BDBC_AllNumeric'])
    additional_params = {'max_episode_steps': MAX_EPISODE_STEPS[env_name]}
    
    if env_name in ['DeltaIoTv1', 'DeltaIoTv2']:
        strategy_type = st.sidebar.selectbox("Strategy Type", ['min', 'multi'])
        additional_params['strategy_type'] = strategy_type
        goal = st.sidebar.selectbox("Goal", ['packet', 'energy', 'latency', "multi", 
                                        'energy_thresh', 'packet_thresh', 'latency_thresh'])
        additional_params['goal'] = goal
        additional_params[f'{goal.split('_')[0]}_coef'] = 1.0
        additional_params['energy_coef'] = additional_params['packet_coef'] = additional_params['latency_coef'] = 0
        # if strategy_type == 'multi':
        energy_coef = st.sidebar.number_input("Energy Coefficient", min_value=0.0, max_value=1.0, value=0.6)
        packet_coef = st.sidebar.number_input("Packet Coefficient", min_value=0.0, max_value=1.0, value=0.2)
        latency_coef = st.sidebar.number_input("Latency Coefficient", min_value=0.0, max_value=1.0, value=0.2)
        
        total_coef = energy_coef + packet_coef + latency_coef
        if total_coef > 0:
            additional_params['energy_coef'] = energy_coef / total_coef
            additional_params['packet_coef'] = packet_coef / total_coef
            additional_params['latency_coef'] = latency_coef / total_coef

        additional_params['setpoint_thresh'] = {
            'lower_bound': st.sidebar.number_input("Setpoint Threshold Lower Bound", 12.8),
            'upper_bound': st.sidebar.number_input("Setpoint Threshold Upper Bound", 13.0)
        }
        # elif strategy_type == 'min':
        

    algo_name = st.sidebar.selectbox("Algorithm Name", [ 'HER_DQN', 'DQN', 'PPO', 'A2C'])
    policy_options = {
        "DQN": ["MlpPolicy", 'SoftmaxDQNPolicy', 'BoltzmannDQNPolicy', 'UCBDQNPolicy', 'UCB1TUNEDDQNPolicy', 'BayesianUCBDQNPolicy', 'SoftmaxUCBDQNPolicy'],
        "PPO": ["MlpPolicy", "CnnPolicy", "MultiInputPolicy"],
        "A2C": ["MlpPolicy", "CnnPolicy", "MultiInputPolicy"],
        "HER_DQN": ["MultiInputPolicy"]
    }
    policy = st.sidebar.selectbox("Policy", policy_options[algo_name])
    
    lr = st.sidebar.text_input("Learning Rate", "0.0001,0.001,0.01,0.1")
    exploration_fraction = st.sidebar.text_input("Exploration Fraction (for DQN)", "0.1,0.2,0.4,0.6")
    warmup_count = st.sidebar.number_input("Warmup Count", min_value=0, value=1024)
    gamma = st.sidebar.number_input("Gamma", min_value=0.0, max_value=1.0, value=0.99)
    batch_size = st.sidebar.number_input("Batch Size", 64)
    total_timesteps = st.sidebar.number_input("Total Timesteps", min_value=2000, value=5000)
    max_episode_steps = st.sidebar.number_input("Max Episode Steps", additional_params['max_episode_steps'])
    chkpt_dir = st.sidebar.text_input("Checkpoint Directory", "models")
    log_path = st.sidebar.text_input("Log Path", "logs")
    execution_mode = st.sidebar.selectbox("Execution Mode", ['Parallel', 'Serial'])

    num_processes = None
    if execution_mode == 'Parallel':
        num_processes = st.sidebar.number_input("Number of Processes", min_value=1, value=4)

    if st.sidebar.button("Clear Logs and Models"):
        try:
            if os.path.exists(log_path):
                shutil.rmtree(log_path)
                st.sidebar.success(f"Successfully deleted the contents of {log_path}")
            if os.path.exists(chkpt_dir):
                shutil.rmtree(chkpt_dir)
                st.sidebar.success(f"Successfully deleted the contents of {chkpt_dir}")
        except Exception as e:
            display_error_message(e, "Clearing Logs and Models")

    if st.sidebar.button("Launch TensorBoard"):
        try:
            log_dir = log_path if log_path else "logs"
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            tb_process = subprocess.Popen(['tensorboard', '--logdir', log_dir])
            tensorboard_url = "http://localhost:6006"
            st.sidebar.markdown(f'[Open TensorBoard in a new tab]({tensorboard_url})', unsafe_allow_html=True)
            webbrowser.open_new_tab(tensorboard_url)
        except Exception as e:
            display_error_message(e, "Launching TensorBoard")

    st.header("Training Configuration")
    
    eps_min = 0.001
    epsilon = 1.0
    network_layers = [150, 120, 100, 50, 25]
    n_obs_space, n_actions, use_dict_obs_space = get_env_parameters(env_name, algo_name)
    num_pulls = np.zeros(n_actions)

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
            stable_dqn(env_name, algo_name, goal, policy, lr, exploration_fraction, gamma, batch_size, total_timesteps, chkpt_dir, log_path, warmup_count, eps_min, epsilon, num_pulls, additional_params, execution_mode, num_processes)
            st.success("Model training completed")
        except PermissionError as e:
            display_error_message(e, "Training Model (PermissionError)")
        except Exception as e:
            display_error_message(e, "Training Model")

    st.header("Visualization")
    
    if env_name == "DeltaIoTv1":
        st.subheader("Visualization for DeltaIoTv1 Environment")
        data_dir = st.text_input("Data Directory", r'D:\projects\gheibi-material\generated_data_by_deltaiot_simulation\under_drift_scenario')
        from_cycles = st.number_input("From cycles", min_value=0, value=0)
        to_cycles = st.number_input("To cycles", min_value=0, value=1505)
        if st.button("Load and Plot Data"):
            load_and_plot_data(st, data_dir, from_cycles, to_cycles) 

    st.header("TensorBoard Log Visualization")
    
    log_dir = r'D:\projects\RL-DeltaIoT\logs'
    available_dirs = [f.path for f in os.scandir(log_dir) if f.is_dir()]

    parsed_dirs = [parse_log_directory_name(d) for d in available_dirs]
    unique_algos = list(set(d['algo'] for d in parsed_dirs))
    unique_goals = list(set(d['goal'] for d in parsed_dirs))
    unique_envs = list(set(d['env'] for d in parsed_dirs))
    unique_policies = list(set(d['policy'] for d in parsed_dirs))
    unique_lrs = list(set(d['lr'] for d in parsed_dirs))
    unique_batch_sizes = list(set(d['batch_size'] for d in parsed_dirs))
    unique_gammas = list(set(d['gamma'] for d in parsed_dirs))
    unique_timesteps = list(set(d['total_timesteps'] for d in parsed_dirs))
    unique_explorations = list(set(d['exploration_fraction'] for d in parsed_dirs))

    selected_algos = st.sidebar.multiselect("Algorithm", unique_algos, default=unique_algos)
    selected_goals = st.sidebar.multiselect("Goal", unique_goals, default=unique_goals)
    selected_envs = st.sidebar.multiselect("Environment", unique_envs, default=unique_envs)
    selected_policies = st.sidebar.multiselect("Policy", unique_policies, default=unique_policies)
    selected_lrs = st.sidebar.multiselect("Learning Rate", unique_lrs, default=unique_lrs)
    selected_batch_sizes = st.sidebar.multiselect("Batch Size", unique_batch_sizes, default=unique_batch_sizes)
    selected_gammas = st.sidebar.multiselect("Gamma", unique_gammas, default=unique_gammas)
    selected_timesteps = st.sidebar.multiselect("Total Timesteps", unique_timesteps, default=unique_timesteps)
    selected_explorations = st.sidebar.multiselect("Exploration Fraction", unique_explorations, default=unique_explorations)

    filtered_dirs = [
        d for d in available_dirs 
        if parse_log_directory_name(d)['algo'] in selected_algos
        and parse_log_directory_name(d)['goal'] in selected_goals
        and parse_log_directory_name(d)['env'] in selected_envs
        and parse_log_directory_name(d)['policy'] in selected_policies
        and parse_log_directory_name(d)['lr'] in selected_lrs
        and parse_log_directory_name(d)['batch_size'] in selected_batch_sizes
        and parse_log_directory_name(d)['gamma'] in selected_gammas
        and parse_log_directory_name(d)['total_timesteps'] in selected_timesteps
        and parse_log_directory_name(d)['exploration_fraction'] in selected_explorations
    ]

    smoothed = st.checkbox("Apply Smoothing", value=True)
    smooth_factor = st.slider("Smoothness Factor", min_value=0.0, max_value=1.0, value=0.6)
    tags = st.text_input("Tags", "rollout/ep_rew_mean,eval/mean_reward").split(",")

    if st.button("Visualize Logs"):
        try:
            read_from_tensorboardlog(st=st, log_dirs=filtered_dirs, smoothed=smoothed, smooth_factor=smooth_factor, tags=tags)
        except Exception as e:
            display_error_message(e, "Visualizing Logs")

if __name__ == "__main__":
    main()
