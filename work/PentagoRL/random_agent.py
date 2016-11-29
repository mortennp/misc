# -*- coding: utf-8 -*-
import numpy as np

class RandomAgent(object):    
    def __init__(self, tag):
        self.tag = tag
        
                    
    def reset(self):
        return
            

    def get_action(self, obs, verbose=False):
        (_, legal_actions_mask, _) = obs
        buf = np.argwhere(legal_actions_mask == 1)
        buf_idx = np.random.randint(0,len(buf))
        action = buf[buf_idx][0]
        return action
        
    
    def learn(self, obs, action, obs_next, reward, done, info, verbose=False):
        return
        
        
    def trace(self, verbose=False, save=False):
        return
    