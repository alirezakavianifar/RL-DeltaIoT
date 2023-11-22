import os
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import CheckpointCallback




def stable_dqn(agent_params):
    name = agent_params['chkpt_dir'].rsplit('\\',1)[1]
    save_path = agent_params['chkpt_dir'].rsplit('\\',1)[0]
    checkpoint_callback = CheckpointCallback(save_freq=4320, save_path=save_path,
                                         name_prefix=f"{name}-lr={agent_params['lr']}-batch_size={agent_params['batch_size']}-gamma={agent_params['gamma']}")
    model = DQN("MlpPolicy", 
                agent_params['env'],
                learning_rate=agent_params['lr'],
                learning_starts=agent_params['warmup_count'],
                batch_size=agent_params['batch_size'],
                target_update_interval=agent_params['n_actions'] * 5,
                gamma=agent_params['gamma'],
                exploration_initial_eps=agent_params['epsilon'],
                exploration_final_eps=agent_params['eps_min'],
                tensorboard_log=agent_params['log_path'],
                verbose=1)
    
    model.learn(total_timesteps=43200, log_interval=4, callback=checkpoint_callback)
