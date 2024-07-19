import os
# from stable_baselines3 import DQN, PPO, HerReplayBuffer
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import EvalCallback
from gymnasium.wrappers.time_limit import TimeLimit
from stable_baselines3.common.monitor import Monitor

from src.drl4sao.stable_algos.custom_stable import CustomDQN
from src.drl4sao.stable_algos.rl_algos.ppo import PPO
from src.drl4sao.stable_algos.rl_algos.a2c import A2C
from src.drl4sao.stable_algos.custom_policies.bayesian_ucb import BayesianUCB

from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy

def stable_dqn(agent_params):
    lrs = agent_params['lr'].split(',')
    exploration_fractions = agent_params['exploration_fraction'].split(',')
    algo_name = agent_params['chkpt_dir'].rsplit('\\', 1)[1].split('_')[0]
    save_path = agent_params['chkpt_dir'].rsplit('\\', 1)[0]

    for lr in lrs:
        if algo_name == "DQN":
            for exploration_fraction in exploration_fractions:
                env = Monitor(env=TimeLimit(agent_params['env'],
                                            max_episode_steps=agent_params['max_episode_steps']), allow_early_resets=True)
                name = agent_params['chkpt_dir'].rsplit('\\', 1)[1]
                
                log_path = os.path.join(agent_params['log_path'], f"{agent_params['env_name']}_{name}-policy={agent_params['policy']}-lr={lr}-eps_min={agent_params['eps_min']}-batch_size={agent_params['batch_size']}-gamma={agent_params['gamma']}-exploration_fraction={exploration_fraction}-total_timesteps={agent_params['total_timesteps']}")
                checkpoint_callback = CheckpointCallback(save_freq=1, save_path=save_path,
                                                        name_prefix=f"{agent_params['env_name']}_{name}-policy={agent_params['policy']}-lr={lr}-eps_min={agent_params['eps_min']}-batch_size={agent_params['batch_size']}-gamma={agent_params['gamma']}-exploration_fraction={exploration_fraction}-total_timesteps={agent_params['total_timesteps']}")
                eval_callback = EvalCallback(
                    env, callback_on_new_best=checkpoint_callback, eval_freq=int(agent_params['total_timesteps']/20), verbose=1, n_eval_episodes=5)
                
                bayesian_ucb = BayesianUCB(agent_params['n_actions'])

                model = CustomDQN(agent_params['policy'],
                                  env,
                                  learning_rate=float(lr),
                                  learning_starts=agent_params['warmup_count'],
                                  batch_size=agent_params['batch_size'],
                                  gamma=agent_params['gamma'],
                                  exploration_initial_eps=agent_params['epsilon'],
                                  exploration_final_eps=agent_params['eps_min'],
                                  exploration_fraction=float(exploration_fraction),
                                  replay_buffer_class=None,
                                  tensorboard_log=log_path,
                                  device='cuda',
                                  verbose=1,
                                  deterministic=False,
                                  total_timesteps=agent_params['total_timesteps'],
                                  num_pulls=agent_params['num_pulls'],
                                  setpoint_thresh=agent_params['setpoint_thresh'],
                                  bayesian_ucb=bayesian_ucb
                                  )

                model.learn(total_timesteps=agent_params['total_timesteps'], log_interval=1,
                            callback=eval_callback)

        elif algo_name == "PPO":
            env = Monitor(env=TimeLimit(agent_params['env'],
                                        max_episode_steps=agent_params['max_episode_steps']), allow_early_resets=True)
            name = agent_params['chkpt_dir'].rsplit('\\', 1)[1]
            
            log_path = os.path.join(agent_params['log_path'], f"{agent_params['env_name']}_{name}-policy={agent_params['policy']}-lr={lr}-batch_size={agent_params['batch_size']}-gamma={agent_params['gamma']}-total_timesteps={agent_params['total_timesteps']}")
            checkpoint_callback = CheckpointCallback(save_freq=1, save_path=save_path,
                                                    name_prefix=f"{agent_params['env_name']}_{name}-policy={agent_params['policy']}-lr={lr}-batch_size={agent_params['batch_size']}-gamma={agent_params['gamma']}-total_timesteps={agent_params['total_timesteps']}")
            eval_callback = EvalCallback(
                env, callback_on_new_best=checkpoint_callback, eval_freq=int(agent_params['total_timesteps']/20), verbose=1, n_eval_episodes=5)

            model = PPO("MlpPolicy",
                        env,
                        learning_rate=float(lr),
                        batch_size=agent_params['batch_size'],
                        gamma=agent_params['gamma'],
                        tensorboard_log=log_path,
                        n_steps=100,
                        device='cuda',
                        verbose=1)
            
            model.learn(total_timesteps=agent_params['total_timesteps'], log_interval=1,
                        callback=eval_callback)

        elif algo_name == "A2C":
            env = Monitor(env=TimeLimit(agent_params['env'],
                                        max_episode_steps=agent_params['max_episode_steps']), allow_early_resets=True)
            name = agent_params['chkpt_dir'].rsplit('\\', 1)[1]
            
            log_path = os.path.join(agent_params['log_path'], f"{agent_params['env_name']}_{name}-policy={agent_params['policy']}-lr={lr}-batch_size={agent_params['batch_size']}-gamma={agent_params['gamma']}-total_timesteps={agent_params['total_timesteps']}")
            checkpoint_callback = CheckpointCallback(save_freq=1, save_path=save_path,
                                                    name_prefix=f"{agent_params['env_name']}_{name}-policy={agent_params['policy']}-lr={lr}-batch_size={agent_params['batch_size']}-gamma={agent_params['gamma']}-total_timesteps={agent_params['total_timesteps']}")
            eval_callback = EvalCallback(
                env, callback_on_new_best=checkpoint_callback, eval_freq=int(agent_params['total_timesteps']/20), verbose=1, n_eval_episodes=5)

            model = A2C("MlpPolicy",
                        env,
                        learning_rate=float(lr),
                        gamma=agent_params['gamma'],
                        tensorboard_log=log_path,
                        n_steps=100,
                        device='cuda',
                        verbose=1)
            
            model.learn(total_timesteps=agent_params['total_timesteps'], log_interval=1,
                        callback=eval_callback)
