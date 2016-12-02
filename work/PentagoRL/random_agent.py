# -*- coding: utf-8 -*-
import numpy as np
from gym.utils import seeding

class RandomAgent(object):    
    def __init__(self, tag):
        self.tag = tag
        self.np_random = None

        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        

    def get_action(self, obs):
        (_, legal_actions_mask, _) = obs
        buf = np.argwhere(legal_actions_mask == 1)
        buf_idx = self.np_random.randint(0,len(buf))
        action = buf[buf_idx][0]
        return action