import pretty_errors
from collections import defaultdict
import glob
import os
from tensorflow.keras.models import load_model
# from src.experiments.dqn.dqn import dqn
from src.drl4sao.custom_dqn.dqn import dqn
from stable_baselines3 import DQN, PPO
from src.drl4sao.custom_dqn.agent_helpers import get_models, get_models_v1, get_models_v2, get_models_v3
from src.utility.test_utils import test_phase, evaluate_models, load_models
from src.utility import config


if __name__ == '__main__':

    DEEP_AGENT_PARAMS = config.get_params.main(standalone_mode=False)

    if DEEP_AGENT_PARAMS['training']:
        dqn(agent_params=DEEP_AGENT_PARAMS)

    else:

        def get_chosen_model(model_dics, params):
            model_dics_ = defaultdict(list)
            for key, item in model_dics.items():
                if key == os.path.join(os.getcwd(), 'models', f"DQN_v1_multi-n_games=*-lr={params['lr']}-eps_dec={params['eps_dec']}-batch_size={params['batch_size']}-gamma={params['gamma']}-q_eval"):
                    model_dics_[key] = model_dics[key]
                    return model_dics_
                             
        model_dics = get_models(lambda: get_models_v2(
            DEEP_AGENT_PARAMS['model_dics']))
        
        chosen_model_params = {'lr':'0.0001', 'eps_dec':'0.00167', 'batch_size': '64', 'gamma': '0.95'}
        
        model_dics = get_chosen_model(model_dics, chosen_model_params)

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
