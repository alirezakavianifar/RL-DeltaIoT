import pretty_errors
from collections import defaultdict
import glob
from tensorflow.keras.models import load_model
# from src.experiments.dqn.dqn import dqn
from src.utility.agent_helpers import dqn
from stable_baselines3 import DQN, PPO
from src.utility.agent_helpers import get_models, get_models_v1, get_models_v2, get_models_v3
from src.utility.test_utils import test_phase, evaluate_models, load_models
from src.utility import config


if __name__ == '__main__':

    DEEP_AGENT_PARAMS = config.get_params.main(standalone_mode=False)

    if DEEP_AGENT_PARAMS['training']:
        dqn(agent_params=DEEP_AGENT_PARAMS)

    else:

        model_dics = get_models(lambda: get_models_v2(r'D:\repo\models\DQN_v1_multi-n_games=*'))

        models = config.get_models(
            model_names=model_dics, model_load_type=load_model)

        test_phase(
            config.TEST_LST, models,
            energy_coef=config.ENERGY_COEF,
            packet_coef=config.PACKET_COEF,
            latency_coef=config.LATENCY_COEF,
            energy_thresh=config.ENERGY_THRESH,
            packet_thresh=config.PACKET_THRESH,
            latency_thresh=config.LATENCY_THRESH,
            num_features=config.INPUT_DIMS)
