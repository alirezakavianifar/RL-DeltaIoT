import numpy as np
from collections import deque
import random


class ReplayBuffer(object):
    """
    A replay buffer for storing and sampling experiences for training a reinforcement learning agent.

    Parameters:
        - max_size (int): The maximum capacity of the replay buffer.
        - input_shape (tuple): The shape of the input state in the experiences.
        - n_actions (int): The number of possible actions in the environment.
        - mem_cntr (int): Initial memory counter, representing the number of stored experiences.

    Attributes:
        - mem_size (int): The maximum capacity of the replay buffer.
        - buffer (deque): A double-ended queue to store experiences.
        - mem_cntr (int): The current count of stored experiences in the buffer.
    """
    def __init__(self, max_size, input_shape, n_actions, mem_cntr=0):
        """
        Initializes the ReplayBuffer instance.

        Parameters:
            - max_size (int): The maximum capacity of the replay buffer.
            - input_shape (tuple): The shape of the input state in the experiences.
            - n_actions (int): The number of possible actions in the environment.
            - mem_cntr (int): Initial memory counter, representing the number of stored experiences.
        """
        self.mem_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.mem_cntr = mem_cntr

    def store_transition(self, experience):
        """
        Stores a new experience in the replay buffer.

        Parameters:
            - experience (tuple): The experience to be stored, typically containing (state, action, reward, next_state, done).
        """
        self.buffer.append(experience)
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        """
        Randomly samples a batch of experiences from the replay buffer.

        Parameters:
            - batch_size (int): The number of experiences to be sampled in the batch.

        Returns:
            List: A list containing randomly sampled experiences.
        """
        batch = random.sample(self.buffer, batch_size)
        return batch

