from collections import deque
from random import choices 
import numpy as np 

class ReplayBuffer:
    
    def __init__(self):
        self.buffer = []
        
        self.indexes = []
        self.current_size = 0
        self.max_size = 1000000
        self.head = 0

    def add_samples(self, inputs, targets):
        assert( len(inputs)==len(targets))
        n = len(inputs)
        for (inp, target) in zip( inputs,targets):
            self.add(inp,target)
        
    def add(self, inp, target):
        if self.current_size < self.max_size:
            self.buffer.append({'input': inp, 'target':target})
            self.indexes.append(self.current_size)
            self.current_size += 1 
        else:
            self.buffer[self.head]={'input': inp, 'target':target}
            self.indexes.remove(self.head)
            self.indexes.append(self.head)
            self.head = (self.head +1) % self.max_size
    
    def sample(self):
        n_sample = self.current_size 
        samples_indexes = choices( self.indexes, k=n_sample)
        x, y = [], []
        for idx in samples_indexes:
            sample = self.buffer[idx]
            x.append(sample['input'])
            y.append(sample['target'])
        return np.array(x), np.array(y)
