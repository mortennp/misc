# -*- coding: utf-8 -*-
from collections import defaultdict
import hashlib
import numpy as np
from gym.spaces import prng

class RandomAgent(object):    
    def __init__(self, env):        
        self.action_n = env.action_space.n
        
        self.legal_actions_masks = defaultdict(lambda: np.ones(self.action_n))
        self.obs_idx_num = 0
        self.obs_idx_map = dict()        
                    
    def reset(self):
        return
            

    def act(self, obs, verbose=False, eps=None):
        obs_idx = self.get_obs_idx(obs)
        mask = self.legal_actions_masks[obs_idx]
        buf = np.argwhere(mask == 1)
        buf_idx = prng.np_random.randint(len(buf))
        action = buf[buf_idx][0]
        return action
        
    
    def learn(self, obs, action, obs_next, reward, done, info, verbose=False):
        if done and info.find("illegal") != -1:
            obs_idx = self.get_obs_idx(obs)
            self.legal_actions_masks[obs_idx][action] = 0
            
            
    def get_obs_idx(self, obs):
        h = self.get_hash_key(obs)
        if not h in self.obs_idx_map:
            self.obs_idx_map[h] = self.obs_idx_num            
            self.obs_idx_num += 1            
        return self.obs_idx_map[h]
        
                        
    def get_hash_key(self, obs):
        return hashlib.sha1(obs.view(np.uint8)).hexdigest() #return np.array_str(obs)            
        