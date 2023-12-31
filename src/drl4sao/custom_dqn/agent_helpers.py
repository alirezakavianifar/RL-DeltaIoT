import os
import re
from collections import defaultdict
import glob
from stable_baselines3 import DQN

from src.utility.utils import set_log_dir, timeit

GET_CWD = os.getcwd()


def get_models_v1(str_base=f'{os.path.join(GET_CWD, "models")}\\DQN_v1_multi-n_games=*'):
    lrs = ['lr=0.0001', 'lr=0.001', 'lr=0.01', 'lr=0.1']
    # lrs = [0.0001]
    epochs = [100, 200, 300, 400, 500]
    batch_sizes = [64]
    gammas = [1.0, 0.99, 0.98, 0.95, 0.90]
    gammas = ['gamma=1.0']
    items = [(lr, (epoch, bs, gamma)) for lr in lrs
             for epoch in epochs
             for bs in batch_sizes
             for gamma in gammas]
    model_dics = defaultdict(list)
    [model_dics[key].append(
        f'{str_base}{value[0]}-{key}-eps_dec=0.00167-batch_size={value[1]}-{value[2]}-q_eval') for key, value in items]

    return model_dics


def get_models_v2(str_base=f'{os.path.join(os.getcwd(), "models")}\\DQN_v1_multi-n_games=*',
                  algo_name=None, quality_type=None, model_dir=None, load_model=None):
    
    if model_dir == '1':
        str_bases = list(dict.fromkeys([re.sub(
            r'n_games=\d*[a-z]*', 'n_games=*', item) for item in str_base[quality_type]]))
    elif model_dir == '2':
        str_bases = list(dict.fromkeys([re.sub(
            r'\d*[a-z]*_steps', '*_steps', item) for item in str_base[quality_type]]))
    files = defaultdict(list)
    [files[lr].append(sorted(glob.glob(lr), key=os.path.getmtime)) for lr in str_bases]
    return files


def get_models_v3(str_base,algo_name=None, quality_type=None):
    files = defaultdict(list)


    # lrs = [
    #     f'{str_base}\\*.zip',
    # ]

    # [files[lr].append(str_base[quality_type])]
    return files


def get_models(get_models_v):
    return get_models_v()


def dqn(agent_params):
    model = DQN("MlpPolicy", agent_params['env'], verbose=1, learning_starts=216, learning_rate=agent_params['lr'],
                gamma=agent_params['gamma'], exploration_final_eps=agent_params['eps_min'], batch_size=agent_params['batch_size'])
    model.learn(total_timesteps=agent_params['n_games'] * 216, log_interval=1)
    model.save(os.path.join(agent_params['log_path'],
                            f"DQN_v1_multi_lr={agent_params['lr']}_eps_dec={agent_params['eps_min']}_batch_size={agent_params['batch_size']}_gamma={agent_params['gamma']}.zip"))
