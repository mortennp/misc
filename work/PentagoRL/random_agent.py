# -*- coding: utf-8 -*-
import numpy as np

class RandomAgent(object):    
    def __init__(self, tag):
        self.tag = tag
        

    def get_action(self, obs):
        (_, legal_actions_mask, _) = obs
        buf = np.argwhere(legal_actions_mask == 1)
        buf_idx = np.random.randint(0,len(buf))
        action = buf[buf_idx][0]
        return action