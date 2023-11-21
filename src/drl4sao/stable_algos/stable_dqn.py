import os
from stable_baselines3 import DQN, PPO


def stable_dqn(agent_params):
    model = DQN("MlpPolicy", 
                env=agent_params['env'],
                learning_rate=agent_params['lr'],
                learning_starts=agent_params['warmup_count'],
                batch_size=agent_params['batch_size'],
                gamma=agent_params['gamma'],
                exploration_initial_eps=agent_params['epsilon'],
                exploration_final_eps=agent_params['eps_min'],
                tensorboard_log=agent_params['log_path'],
                verbose=1)
    
    model.learn(total_timesteps=7500, log_interval=4)
    saved_dir = f"{agent_params['chkpt_dir']}-n_games="
    model.save(os.path.join(agent_params['chkpt_dir'],'stable_models', 'models.zip'))