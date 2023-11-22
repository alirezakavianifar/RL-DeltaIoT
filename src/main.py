import pretty_errors
from collections import defaultdict
import glob
import os
# from src.experiments.dqn.dqn import dqn
from src.drl4sao.custom_dqn.dqn import dqn
from src.drl4sao.stable_algos.stable_dqn import stable_dqn
from src.drl4sao.custom_dqn.agent_helpers import get_models, get_models_v1, get_models_v2, get_models_v3
from src.utility.test_utils import test_phase, evaluate_models, load_models
from src.utility import config
from src.utility.utils import get_chosen_model


if __name__ == '__main__':

    DEEP_AGENT_PARAMS = config.get_params.main(standalone_mode=False)


    if DEEP_AGENT_PARAMS['training']:
        if DEEP_AGENT_PARAMS['algo_type'] == 1:
            dqn(agent_params=DEEP_AGENT_PARAMS)
        elif DEEP_AGENT_PARAMS['algo_type'] == 2:
            stable_dqn(agent_params=DEEP_AGENT_PARAMS)
    else:

        model_dics = get_models(lambda: get_models_v2(
            DEEP_AGENT_PARAMS['model_dics'], 
            DEEP_AGENT_PARAMS['algo_name'],
            DEEP_AGENT_PARAMS['quality_type'],
            DEEP_AGENT_PARAMS['model_dir'],
            DEEP_AGENT_PARAMS['load_model']))

        if DEEP_AGENT_PARAMS['cmp']:
            params_dict = defaultdict(list)

            for key, values in DEEP_AGENT_PARAMS['model_dics'].items():
                for item in values:
                    [it.split('=') for it in item.split('-')[2:-1]]
                    items = item.split('-')[2:]
                

            chosen_model_params = {
                'lr': '0.0001', 'eps_dec': '0.01961', 'batch_size': '64', 'gamma': '1.0'}

            model_dics = get_chosen_model(model_dics, chosen_model_params)

        models = config.get_models(
            model_names=model_dics, model_load_type=DEEP_AGENT_PARAMS['load_model'])

        test_phase(
            config.TEST_LST, models,
            energy_coef=config.ENERGY_COEF,
            packet_coef=config.PACKET_COEF,
            latency_coef=config.LATENCY_COEF,
            energy_thresh=config.ENERGY_THRESH,
            packet_thresh=config.PACKET_THRESH,
            latency_thresh=config.LATENCY_THRESH,
            num_features=config.INPUT_DIMS,
            cmp=DEEP_AGENT_PARAMS['cmp'],
            algo_name=DEEP_AGENT_PARAMS['algo_name'],
            quality_type=DEEP_AGENT_PARAMS['quality_type'],
            model_type=DEEP_AGENT_PARAMS['model_dir'])
