import glob
import os
import click
import glob
from collections import defaultdict
from src.utility.utils import load_data, move_files
from src.environments.deltaiot_env import DeltaIotEnv
import numpy as np
from src.drl4sao.custom_dqn.eps_dec_types import EpsDecTypeOne, EpsDecTypeTwo
from src.environments.env_helpers import RewardMcOne, RewardMcTwo, RewardMcThree, RewardMcFour, RewardMcFive

GET_CWD = os.getcwd()

PROMPT = True
# FROM_SCRATCH is used if we want to split data into training and testing from scratch
FROM_SCRATCH = False
# If the agent needs training then TRAINING flag is used.
TRAINING = True
# DeltaioT versions are DeltaIoTv1 and DeltaIoTv2
VERSION = 'DeltaIoTv1'
ALGO_NAME = 'DQN_v1'
# Quality types
QUALITY_TYPES = {'energy': 'energy', 'packet': 'packet',
                 'latency': 'latency', 'multi': 'multi', 'multi_tto': 'multi_tto'}
# Reward types
REWARD_TYPES = {'energy': 'rm2', 'packet': 'rm2',
                'latency': 'rm2', 'multi': 'rm3', 'multi_tto': 'rm5', 'multi_tt': 'rm4'}
# Reward mechanism: rm1=threshold, rm2=minimum, rm3=multi, rm4=multi_tt, rm5=multi_tto
REWARD_TYPE = REWARD_TYPES['multi']
# QUALITY_TYPE could be energy, packet, latency, multi, multi_tt, multi_tto
QUALITY_TYPE = QUALITY_TYPES['multi']
# deep type could be either tensor or torch
DEEP_TYPE = 'tensor'
# deep types could be a collection of dqn, ddpg, ppo or etc...
RL_TYPES = ['dqn']
DEEP_TYPES = ['dqn']

MODEL_NAMES = ['%s_energy' % ALGO_NAME,
               '%s_packet' % ALGO_NAME,
               '%s_latency' % ALGO_NAME,
               '%s_multi' % ALGO_NAME,
               '%s_multi_tt' % ALGO_NAME]


# In a multi-setting environment, quality objective coefficients are set.
ENERGY_COEF = 0.0
PACKET_COEF = 1.0
LATENCY_COEF = 0.0

# model hyperparameters
ALPHA = 0.1
GAMMA = 0.99
EPS = 1.0

LOG_PATH = os.path.join(GET_CWD, 'logs')
EPS_DEC_TYPE = EpsDecTypeTwo()

if VERSION == 'DeltaIoTv1':
    INPUT_DIMS = 3
    TIME_STEPS = 216
    ENERGY_THRESH = 13.20
    PACKET_THRESH = 15.0
    LATENCY_THRESH = 10.0
    N_ACTIONS = 216
    NETWORK_LAYERS = [50, 25, 15]
    DATA_DIR = os.path.join(GET_CWD, 'data', 'DeltaIoTv1')
    # DATA_DIR = r'D:\projects\papers\Deep Learning for Effective and Efficient  Reduction of Large Adaptation Spaces in Self-Adaptive Systems\DLASeR_plus_online_material\dlaser_plus\raw\DeltaIoTv1'
    N_STATES = 216
    N_OBS_SPACE = 3
else:
    INPUT_DIMS = 42
    TIME_STEPS = 4096
    ENERGY_THRESH = 67.30
    PACKET_THRESH = 15.0
    LATENCY_THRESH = 10.0
    NETWORK_LAYERS = [150, 120, 100, 50, 25]
    N_STATES = 4096
    N_ACTIONS = 4096
    N_OBS_SPACE = 42
    DATA_DIR = os.path.join(GET_CWD, 'data', 'DeltaIoTv2')
# Create Training and testing Data Directory

TRAIN_DIR = os.path.join(DATA_DIR, 'train')
os.makedirs(TRAIN_DIR, exist_ok=True)
TEST_DIR = os.path.join(DATA_DIR, 'test')
os.makedirs(TEST_DIR, exist_ok=True)

if FROM_SCRATCH:
    TRAIN_LST, TEST_LST = load_data(
        path=DATA_DIR, load_all=False, version='', shuffle=False, fraction=1.0, test_size=0.2, return_train_test=True)

    move_files(TRAIN_LST, TRAIN_DIR)
    move_files(TEST_LST, TEST_DIR)
else:
    TRAIN_LST = glob.glob(os.path.join(TRAIN_DIR, "*.json"))
    TEST_LST = glob.glob(os.path.join(TEST_DIR, "*.json"))


NUMGAMES = len(TRAIN_LST)
print(TRAIN_LST)
EPSILON = 1
EPS_MIN = 0.001
EPS_STEP_SIZE = 1
EPS_DEC = EPS_STEP_SIZE/NUMGAMES
GAMMA = 1
LR = 0.0001
MEM_SIZE = 1024
BATCH_SIZE = 64
REPLACE = 100
ENV_NAME = 'DeltaIoT'
ALGO = 'DeltaIOTAgent'
WARMUP_COUNT = 100

ALGORITHMS = ['SARSA', 'Q_LEARNING', 'DOUBLE_Q_LEARNING']
TOTAT_TIMES = []
CHKPT_DIR = os.path.join(
    GET_CWD, 'models', ALGO_NAME + '_' + QUALITY_TYPE)
FNAME = ALGO + '_' + ENV_NAME + '_lr' + str(LR) + '_' \
    + str(NUMGAMES) + 'games'

MODEL_DIR = os.path.join(GET_CWD, 'models')


def get_models(model_names, model_load_type=None):

    models = defaultdict(lambda: defaultdict(list))
    for key, values in model_names.items():
        for items in values:
            for item in items:
                # cycle = item.split('=')[1].split('-')[0]
                models[key][item].append(model_load_type(item))
    return models


def wrapper_get_params_for_training(is_training, *args, **kwargs):
    if is_training:
        kwargs = get_params_for_training.main(standalone_mode=False)
    else:
        kwargs = get_params_for_testing.main(standalone_mode=False)
    kwargs['training'] = is_training
    return kwargs


@click.command()
@click.option('--environment', prompt=PROMPT, default="DeltaIoT")
@click.option('--warmup_count', prompt=PROMPT, default=WARMUP_COUNT)
@click.option('--lr', prompt=PROMPT, default=LR)
@click.option('--n_games', prompt=PROMPT, default=NUMGAMES)
@click.option('--gamma', prompt=PROMPT, default=GAMMA, type=float)
@click.option('--mem_size', prompt=PROMPT, default=MEM_SIZE)
@click.option('--batch_size', prompt=PROMPT, default=BATCH_SIZE)
@click.option('--epsilon', prompt=PROMPT, default=EPSILON)
# @click.option('--eps_dec_type', prompt=PROMPT, default=EPS_DEC_TYPE)
@click.option('--eps_step_size', prompt=PROMPT, default=EPS_STEP_SIZE, type=float)
@click.option('--eps_min', prompt=PROMPT, default=EPS_MIN)
@click.option('--env_name', prompt=PROMPT, default=ENV_NAME)
@click.option('--algo', prompt=PROMPT, default=ALGO)
@click.option('--replace', prompt=PROMPT, default=REPLACE)
@click.option('--fname', prompt=PROMPT, default=FNAME)
# @click.option('--Network_layers', prompt=PROMPT, default=NETWORK_LAYERS)
@click.option('--quality_type', prompt=PROMPT, default=QUALITY_TYPE)
@click.option('--algo_name', prompt=PROMPT, default=ALGO_NAME)
@click.option('--log_path', prompt=PROMPT, default=LOG_PATH)
@click.option('--chkpt_dir', prompt=PROMPT, default=CHKPT_DIR)
def get_params_for_training(*args, **kwargs):

    if kwargs['environment'] == 'DeltaIoT':
        ENV = DeltaIotEnv(data_dir=TRAIN_LST, timesteps=TIME_STEPS, n_actions=N_ACTIONS, n_obs_space=N_OBS_SPACE,
                          reward_type=RewardMcThree, energy_coef=ENERGY_COEF, packet_coef=PACKET_COEF, latency_coef=LATENCY_COEF,
                          packet_thresh=PACKET_THRESH, latency_thresh=LATENCY_THRESH, energy_thresh=ENERGY_THRESH)

    # agent params
    DEEP_AGENT_PARAMS = {
        'env': ENV,
        'eps_min': kwargs['eps_min'],
        'n_games': kwargs['n_games'],
        'n_actions': N_ACTIONS,
        'gamma': kwargs['gamma'],
        'epsilon': kwargs['epsilon'],
        'eps_dec_type': EPS_DEC_TYPE,
        'eps_dec': kwargs['eps_step_size']/kwargs['n_games'],
        'env_name': kwargs['env_name'],
        'algo': kwargs['algo'],
        'mem_size': kwargs['mem_size'],
        'input_dims': INPUT_DIMS,
        'batch_size': kwargs['batch_size'],
        'replace': kwargs['replace'],
        'lr': kwargs['lr'],
        'fname': kwargs['fname'],
        'network_layers': NETWORK_LAYERS,
        'quality_type': kwargs['quality_type'],
        'algo_name': kwargs['algo_name'],
        'log_path': kwargs['log_path'],
        'chkpt_dir': kwargs['chkpt_dir'],
        'warmup_count': kwargs['warmup_count'],
    }
    return DEEP_AGENT_PARAMS


def is_test(ctx, param, value):
    if not value or ctx.resilient_parsing:
        return False
    return True


@click.command()
@click.option('--training', prompt=True, is_flag=True, callback=is_test,
              expose_value=True, is_eager=True)
def get_params(*args, **kwargs):
    return wrapper_get_params_for_training(is_training=kwargs['training'])


@click.command()
@click.option('--model_dir', prompt=True, default=MODEL_DIR)
@click.option('--model_names', prompt=True, default='multi')
def get_params_for_testing(*args, **kwargs):
    MODEL_DICS = {}
    model_names = kwargs['model_names'].split(',')

    for item in model_names:
        dir_name = r'%s\DQN_v1_%s' % (kwargs['model_dir'], item)
        files = glob.glob(dir_name + '*q_eval')
        if len(files) > 0:
            MODEL_DICS[item] = files
    DEEP_AGENT_PARAMS = {'model_dics': MODEL_DICS}
    return DEEP_AGENT_PARAMS
