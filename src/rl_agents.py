import warnings
warnings.filterwarnings('ignore')
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.her import GoalSelectionStrategy, HerReplayBuffer
from src.drl4sao.stable_algos.custom_stable import CustomDQN
from src.drl4sao.stable_algos.custom_policies.bayesian_ucb import BayesianUCB
from src.drl4sao.stable_algos.rl_algos.a2c import A2C


def train_dqn(env, policy, lr, exploration_fraction, gamma, batch_size, total_timesteps, warmup_count, eps_min, epsilon, num_pulls, setpoint_thresh, log_path, checkpoint_callback, eval_callback):
    bayesian_ucb = BayesianUCB(env.action_space.n)
    model = CustomDQN(policy,
                      env,
                      learning_rate=lr,
                      learning_starts=warmup_count,
                      batch_size=batch_size,
                      gamma=gamma,
                      exploration_initial_eps=epsilon,
                      exploration_final_eps=eps_min,
                      exploration_fraction=exploration_fraction,
                      replay_buffer_class=None,
                      tensorboard_log=log_path,
                      device='cuda',
                      verbose=1,
                      deterministic=False,
                      num_pulls=num_pulls,
                      setpoint_thresh=setpoint_thresh,
                      bayesian_ucb=bayesian_ucb)
    model.learn(total_timesteps=total_timesteps, log_interval=1, callback=eval_callback)

def train_ppo(env, policy, lr, gamma, batch_size, total_timesteps, log_path, checkpoint_callback, eval_callback):
    model = PPO(policy,
                env,
                learning_rate=lr,
                batch_size=batch_size,
                gamma=gamma,
                tensorboard_log=log_path,
                device='cuda',
                verbose=1)
    model.learn(total_timesteps=total_timesteps, log_interval=1, callback=eval_callback)

def train_a2c(env, policy, lr, gamma, total_timesteps, log_path, checkpoint_callback, eval_callback):
    model = A2C(policy,
                env,
                learning_rate=lr,
                gamma=gamma,
                tensorboard_log=log_path,
                device='cuda',
                verbose=1)
    model.learn(total_timesteps=total_timesteps, log_interval=1, callback=eval_callback)

def train_her_dqn(env, policy, lr, exploration_fraction, gamma, batch_size, total_timesteps, warmup_count, eps_min, epsilon, num_pulls, setpoint_thresh, log_path, checkpoint_callback, eval_callback, additional_params):
    goal_selection_strategy = GoalSelectionStrategy.FUTURE
    replay_buffer_kwargs = {
        'n_sampled_goal': 4,
        'goal_selection_strategy': goal_selection_strategy,
    }
    bayesian_ucb = BayesianUCB(env.action_space.n)
    model = CustomDQN(policy,
                      env,
                      replay_buffer_class=HerReplayBuffer,
                      replay_buffer_kwargs=replay_buffer_kwargs,
                      learning_rate=lr,
                      gamma=gamma,
                      batch_size=batch_size,
                      learning_starts=warmup_count,
                      exploration_initial_eps=epsilon,
                      exploration_final_eps=eps_min,
                      exploration_fraction=exploration_fraction,
                      tensorboard_log=log_path,
                      device='cuda',
                      verbose=1,
                      bayesian_ucb=bayesian_ucb)
    
    model.learn(total_timesteps=total_timesteps, log_interval=1, callback=eval_callback)
