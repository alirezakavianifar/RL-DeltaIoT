import os
import numpy as np
import tensorflow as tf
from src.drl4sao.custom_dqn.network import DeepQNetwork
from src.drl4sao.custom_dqn.replay_memory import ReplayBuffer
train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
checkpoint_filepath = os.path.join(os.getcwd(), 'logs')


import os
import numpy as np
import tensorflow as tf


class Agent:
    def __init__(self, agent_params):
        """
        Initializes the Agent class.

        Parameters:
            - agent_params: Dictionary containing parameters for the agent.
        """
        self.gamma = agent_params['gamma']
        self.n_games = agent_params['n_games']
        self.epsilon = agent_params['epsilon']
        self.eps_dec_type = agent_params['eps_dec_type']
        self.lr = agent_params['lr']
        self.n_actions = agent_params['n_actions']
        self.input_dims = agent_params['input_dims']
        self.batch_size = agent_params['batch_size']
        self.eps_min = agent_params['eps_min']
        self.eps_dec = agent_params['eps_dec']
        self.replace_target_cnt = agent_params['replace']
        self.algo = agent_params['algo']
        self.env_name = agent_params['env_name']
        self.chkpt_dir = agent_params['chkpt_dir']
        self.action_space = [i for i in range(int(agent_params['n_actions']))]
        self.learn_step_counter = 0

        # Initialize ReplayBuffer and DeepQNetwork instances
        self.memory = ReplayBuffer(
            agent_params['mem_size'], agent_params['input_dims'], agent_params['n_actions'])

        self.q_eval = DeepQNetwork(
            agent_params['input_dims'], agent_params['n_actions'], network_layers=agent_params['network_layers'])
        self.q_eval.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=agent_params['lr']))

        self.q_next = DeepQNetwork(
            agent_params['input_dims'], agent_params['n_actions'], network_layers=agent_params['network_layers'])
        self.q_next.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=agent_params['lr']))

    def save_models(self, iteration):
        """
        Saves the Q-network models.

        Parameters:
            - iteration: Iteration or episode number.
        """
        saving_dir = (f'{self.chkpt_dir}-n_games={iteration}-lr={self.lr}'
                      f'-eps_dec={str("{:.5f}".format(self.eps_dec))}'
                      f'-batch_size={str(self.batch_size)}-gamma={str(self.gamma)}')
        self.q_eval.save(f'{saving_dir}-q_eval')
        self.q_next.save(f'{saving_dir}-q_next')
        print('... models saved successfully ...')

    def load_models(self):
        """
        Loads the Q-network models.
        """
        self.q_eval = tf.keras.models.load_model(self.chkpt_dir + '_' + 'lr=' + str(self.lr) +
                                                 '_' + 'eps_dec=' + str(self.eps_dec) + '_' +
                                                 str(self.batch_size) + '_' + str(self.gamma) + '_' + 'q_eval')
        self.q_next = tf.keras.models.load_model(self.chkpt_dir + '_' + 'lr=' + str(self.lr) +
                                                 '_' + 'eps_dec=' + str(self.eps_dec) + '_' +
                                                 str(self.batch_size) + '_' + str(self.gamma) + '_' + 'q_next')
        print('... models loaded successfully ...')

    def store_transition(self, state, action, reward, state_, done, her=False):
        """
        Stores a transition in the replay buffer.

        Parameters:
            - state: Current state.
            - action: Chosen action.
            - reward: Obtained reward.
            - state_: Next state.
            - done: Whether the episode is done.
            - her: Flag for Hindsight Experience Replay (HER).
        """
        self.memory.store_transition(
            (state, action, reward, state_, done))

    def sample_memory(self):
        """
        Samples a batch of experiences from the replay buffer.

        Returns:
            Tuple of tensors containing states, actions, rewards, next states, and done flags.
        """
        states = []
        rewards = []
        dones = []
        actions = []
        states_ = []
        experiences = \
            self.memory.sample_buffer(self.batch_size)

        for index, item in enumerate(experiences):
            states.append(item[0])
            actions.append(item[1])
            rewards.append(item[2])
            states_.append(item[3])
            dones.append(item[4])

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        states_ = tf.convert_to_tensor(states_, dtype=tf.float32)
        return states, actions, rewards, states_, dones

    def choose_action(self, observation):
        """
        Chooses an action based on the current observation.

        Parameters:
            - observation: Current observation.

        Returns:
            Chosen action.
        """
        if np.random.random() > self.epsilon:
            state = tf.convert_to_tensor([observation], dtype=tf.float32)
            actions = self.q_eval(state)
            action = tf.math.argmax(actions, axis=1).numpy()[0]
        else:
            action = np.random.choice(self.action_space)
        return action

    def replace_target_network(self):
        """
        Replaces the target Q-network with the evaluation Q-network.
        """
        if self.learn_step_counter > 0:
            if self.learn_step_counter % self.replace_target_cnt == 0:
                self.q_next.set_weights(self.q_eval.get_weights())

    def decrement_epsilon(self):
        """
        Decrements the exploration parameter epsilon.
        """
        self.epsilon = self.eps_dec_type.eps_dec_type(
            self.epsilon, self.eps_min, self.n_games, self.eps_dec)

    def learn(self):
        """
        Performs the Q-learning update.

        Returns:
            The training loss.
        """
        if self.memory.mem_cntr < self.batch_size * 2:
            return None

        self.replace_target_network()

        states, actions, rewards, states_, dones = self.sample_memory()

        indices = tf.range(self.batch_size, dtype=tf.int32)
        action_indices = tf.stack(
            [indices, tf.cast(actions, tf.int32)], axis=1)

        with tf.GradientTape() as tape:
            q_pred = tf.gather_nd(self.q_eval(states), indices=action_indices)
            q_next = self.q_next(states_)

            max_actions = tf.math.argmax(q_next, axis=1, output_type=tf.int32)
            max_action_idx = tf.stack([indices, max_actions], axis=1)

            q_target = rewards + \
                self.gamma*tf.gather_nd(q_next, indices=max_action_idx) *\
                (1 - dones.numpy())
            loss = tf.keras.losses.MSE(q_pred, q_target)
            train_loss(loss)

        params = self.q_eval.trainable_variables
        grads = tape.gradient(loss, params)

        self.q_eval.optimizer.apply_gradients(zip(grads, params))

        self.learn_step_counter += 1

        return train_loss

    def reset_states(self):
        """
        Resets the states of the agent.
        """
        train_loss.reset_states()

