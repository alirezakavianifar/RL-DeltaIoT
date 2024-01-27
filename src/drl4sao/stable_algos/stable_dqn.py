import os
# from stable_baselines3 import DQN, PPO, HerReplayBuffer
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import EvalCallback
from gymnasium.wrappers.time_limit import TimeLimit
from stable_baselines3.common.monitor import Monitor

from src.drl4sao.stable_algos.custom_stable import CustomDQN

from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy


def stable_dqn(agent_params):
    env = Monitor(env=TimeLimit(agent_params['env'],
                                max_episode_steps=agent_params['n_actions']), allow_early_resets=True)
    name = agent_params['chkpt_dir'].rsplit('\\', 1)[1]
    algo_name = name.split('_')[0]
    save_path = agent_params['chkpt_dir'].rsplit('\\', 1)[0]
    log_path = os.path.join(agent_params['log_path'], f"{name}-policy={agent_params['policy']}-lr={agent_params['lr']}-eps_min={agent_params['eps_min']}-batch_size={agent_params['batch_size']}-gamma={agent_params['gamma']}-exploration_fraction={agent_params['exploration_fraction']}")
    checkpoint_callback = CheckpointCallback(save_freq=1, save_path=save_path,
                                             name_prefix=f"{name}-policy={agent_params['policy']}-lr={agent_params['lr']}-eps_min={agent_params['eps_min']}-batch_size={agent_params['batch_size']}-gamma={agent_params['gamma']}-exploration_fraction={agent_params['exploration_fraction']}")
    eval_callback = EvalCallback(
        env, callback_on_new_best=checkpoint_callback, eval_freq=10_000, verbose=1, n_eval_episodes=5)

    if algo_name == 'DQN':
        model = CustomDQN(agent_params['policy'],
                          env,
                          learning_rate=agent_params['lr'],
                          learning_starts=agent_params['warmup_count'],
                          batch_size=agent_params['batch_size'],
                          # target_update_interval=agent_params['n_actions'] * 8,
                          gamma=agent_params['gamma'],
                          exploration_initial_eps=agent_params['epsilon'],
                          exploration_final_eps=agent_params['eps_min'],
                          exploration_fraction=agent_params['exploration_fraction'],
                          tensorboard_log=log_path,
                          device='cuda',
                          verbose=1,
                          deterministic=False,
                          total_timesteps=agent_params['total_timesteps'],
                          num_pulls = agent_params['num_pulls']
                          )

    if algo_name == "PPO":
        model = PPO("MlpPolicy",
                    env,
                    learning_rate=agent_params['lr'],
                    batch_size=agent_params['batch_size'],
                    # target_update_interval=agent_params['n_actions'] * 8,
                    gamma=agent_params['gamma'],
                    tensorboard_log=agent_params['log_path'],
                    device='cuda',
                    verbose=1)
    # model.learn(total_timesteps=((agent_params['n_games'] - 30) * agent_params['n_actions']), log_interval=1,
    #             callback=eval_callback)
    model.learn(total_timesteps=agent_params['total_timesteps'], log_interval=1,
                callback=eval_callback)
