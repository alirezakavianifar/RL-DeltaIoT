import numpy as np
from collections import deque
import random


class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions, mem_cntr=0):
        self.mem_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.mem_cntr = mem_cntr

    def store_transition(self, experience):
        self.buffer.append(experience)
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return batch
