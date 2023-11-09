import os
import numpy as np
import tensorflow as tf
from src.drl4sao.custom_dqn.agent import Agent

from src.utility.utils import set_log_dir, timeit


@timeit
def dqn(agent_params=None):
    train_summary_writer = set_log_dir(
        name=agent_params['algo_name'] + '_' + agent_params['quality_type'] +
        '_' + 'lr=' + str(agent_params['lr']) + '_' +
        'eps_dec=' + str("{:.5f}".format(agent_params['eps_dec'])) + '_' + 'batch_size=' + str(agent_params['batch_size']) +
        '_' + 'gamma=' + str(agent_params['gamma']),
        path=agent_params['log_path'])
    chkpt_dir = os.path.join(
        agent_params['log_path'], 'models', agent_params['algo_name'] + '_' + agent_params['quality_type'])
    fname = chkpt_dir + '_'
    env = agent_params['env']
    best_score = -np.inf
    best_avg_score = -np.inf
    load_checkpoint = False
    record_agent = False
    agent = Agent(agent_params=agent_params)
    if load_checkpoint:
        agent.load_models()
        agent.epsilon = agent.eps_min
    n_steps = 0
    scores, eps_history, steps_array = [], [], []

    for i in range(agent_params['n_games']):
        done = False
        observation = env.reset()[0]
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, truncated, info = env.step(action)
            score += reward
            if not load_checkpoint:
                agent.store_transition(observation, action,
                                       reward, observation_, done)

                train_loss = agent.learn()

            observation = observation_
            n_steps += 1

        agent.reset_states()
        score = score/env.init_time_steps
        scores.append(score)
        steps_array.append(n_steps)
        agent.decrement_epsilon()
        avg_score = np.mean(scores[-100:])

        if len(scores) >= agent_params['warmup_count']:

            if train_loss is not None:
                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', train_loss.result(), step=i)
                    tf.summary.scalar('score', score, step=i)
                    tf.summary.scalar('avg_score', avg_score, step=i)

            if score > best_score:
                best_score = score

            if (avg_score > best_avg_score):
                best_agent = agent
                best_avg_score = avg_score
            if (i % agent_params['warmup_count'] == 0):
                if not load_checkpoint:
                    best_agent.save_models(i)

        print('episode {} score {:.1f} avg score {:.1f} '
              'best score {:.1f} best average score {:.1f} epsilon {:.2f} steps {}'.
              format(i, score, avg_score, best_score, best_avg_score, agent.epsilon,
                     n_steps))

        eps_history.append(agent.epsilon)

    x = [i+1 for i in range(len(scores))]
