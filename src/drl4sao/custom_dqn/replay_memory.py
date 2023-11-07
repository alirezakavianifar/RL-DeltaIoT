import numpy as np
from collections import deque
import random

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions, her=False, her_probability=0.8, mem_cntr=0):
        self.mem_size = max_size
        if her:
            self.her_probability = her_probability
            self.buffer = deque(maxlen=max_size//2)
            self.her_buffer = deque(maxlen=max_size//2)
        else:
            self.buffer = deque(maxlen=max_size)
        self.mem_cntr = mem_cntr
        

    def store_transition(self, experience, her=False):
        
        if her:
            self.her_buffer.append(experience)
        else:
            self.buffer.append(experience)
        
        self.mem_cntr+= 1

    def sample_buffer(self, batch_size, her=False):
        if her:
            her_batch_size = int(batch_size * self.her_probability)
            regular_batch_size = batch_size - her_batch_size
            
            batch = random.sample(self.buffer, regular_batch_size)
            her_batch = random.sample(self.her_buffer, her_batch_size)
            fully_batch = list(batch + her_batch)
            # fully_batch = np.array(fully_batch)

            return fully_batch
        else:
            batch = random.sample(self.buffer, batch_size)
            # batch = np.array(batch)
            return batch
        
        
        
