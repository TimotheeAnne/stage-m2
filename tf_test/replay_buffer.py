from collections import deque

import numpy as np 

class ReplayBuffer:
    
    def __init__(self):
        self.buffer = []
        
        self.indexes = [[],[],[],[],[]]
        self.current_size = 0
        self.max_size = 2000000
        self.head = 0

    def add_samples(self, inputs, targets):
        assert( len(inputs)==len(targets))
        n = len(inputs)
        for (inp, target) in zip( inputs,targets):
            self.add(inp,target)


    def add_in_indexes(self, target, idx):
        pertinent = False
        if target[18] == 1:
            self.indexes[1].append(idx)
            pertinent = True
        if target[19] == 1:
            self.indexes[2].append(idx)
            pertinent = True
        if target[20] == 1:
            self.indexes[3].append(idx)
            pertinent = True
        if target[21] == 1:
            self.indexes[4].append(idx)
            pertinent = True
        if not pertinent:
            self.indexes[0].append(idx)

    def remove_from_indexes(self, idx):
        for i in range(5):
            if idx in self.indexes[i]:
                self.indexes[i].remove(idx)

    def add(self, inp, target):
        if self.current_size < self.max_size:
            self.buffer.append({'input': inp, 'target':target})
            self.add_in_indexes( target, self.current_size)
            self.current_size += 1 
        else:
            self.buffer[self.head]={'input': inp, 'target':target}
            self.remove_from_indexes(self.head)
            self.add_in_indexes(target, self.head)
            self.head = (self.head +1) % self.max_size
    
    def sample(self, sampling_function, objects):
        n_sample = min([len(self.indexes[i]) for i in objects])
        samples_indexes = []
        for i in objects:
            samples_indexes += list(sampling_function( self.indexes[i], n_sample))
            
        x, y = [], []
        for idx in samples_indexes:
            sample = self.buffer[idx]
            x.append(sample['input'])
            y.append(sample['target'])
        return np.array(x), np.array(y)

    def pretty_print(self):
        sizes = [len(self.indexes[i]) for i in range(5)]
        print("Current size: ", self.current_size)
        print("Head: ", self.head)
        print("Indexes sizes: ", sizes)
        if self.current_size > 0:
            print( " Proportion: ", np.array(sizes)/self.current_size)
