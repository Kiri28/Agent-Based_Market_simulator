import random
from collections import deque

class ExpirienceReplay:
    def __init__(self, size=10000):
        self.data = deque(maxlen = size)
    
    def add(self, transition):
        self.data.append(transition)
        
    def sample(self, size):
        batch = random.sample(self.data, size)
        return list(zip(*batch))