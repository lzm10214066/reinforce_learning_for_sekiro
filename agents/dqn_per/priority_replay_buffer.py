'''
modified by zhenmaoli
2020.9.4
'''

from collections import deque
import random
import numpy as np


class PriorityReplayBuffer(object):

    def __init__(self, buffer_size, alpah):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        self.priority = deque()
        self.alpha = alpah

    def add_T(self, T):
        if self.count < self.buffer_size:
            self.buffer.append(T)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(T)

    def add_P(self, td_error):
        p = np.exp(abs(td_error) * self.alpha)
        if len(self.priority) < self.buffer_size:
            self.priority.append(p)
        else:
            self.priority.popleft()
            self.priority.append(p)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        '''
        batch_size specifies the number of experiences to add
        to the batch. If the replay buffer has less than batch_size
        elements, simply return all of the elements within the buffer.
        Generally, you'll want to wait until the buffer has at least
        batch_size elements before beginning to sample from it.
        '''
        probs = np.array(self.priority)
        probs /= np.sum(probs)

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        batch = [self.buffer[i] for i in indices]

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        s2_batch = np.array([_[3] for _ in batch])
        t_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, s2_batch, t_batch, indices

    def clear(self):
        self.buffer.clear()
        self.priority.clear()
        self.count = 0

    def update_p(self, td_error, sample_indices):
        ps = np.exp(td_error * self.alpha)
        for i, index in enumerate(sample_indices):
            self.priority[index] = ps[i]
