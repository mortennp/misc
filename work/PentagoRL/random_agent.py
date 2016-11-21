# -*- coding: utf-8 -*-
import hashlib
import numpy as np
from os import path
import pickle
from gym.spaces import prng

class RandomAgent(object):    
    def __init__(self, env, tag, load_model=True):
        self.action_n = env.action_space.n
        self.tag = tag
        
        self.legal_actions_file_name = "random-legal-actions-{}.p".format(tag)
        self.obs_idx_map_file_name = "random-obs_idx_map-{}.p".format(tag)
        if load_model and path.isfile(self.legal_actions_file_name) and path.isfile(self.obs_idx_map_file_name):
            print "'{}' Loading from {} and {}".format(self.tag, self.legal_actions_file_name, self.obs_idx_map_file_name)
            with open(self.legal_actions_file_name, "rb") as f:
                self.legal_actions_masks = pickle.load(f)
            with open(self.obs_idx_map_file_name, "rb") as f:
                self.obs_idx_map = pickle.load(f)
                self.obs_idx_num = len(self.obs_idx_map)
        else:
            self.legal_actions_masks = dict() 
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
            self.legal_actions_masks[self.obs_idx_num] = np.ones(self.action_n)
            self.obs_idx_num += 1            
        return self.obs_idx_map[h]
        
                        
    def get_hash_key(self, obs):
        return hashlib.sha1(obs.view(np.uint8)).hexdigest() #return np.array_str(obs)            
        
        
    def trace(self, verbose=False, save=False):
        if verbose:
            print("'{}' States seen: {}".format(self.tag, len(self.legal_actions_masks)))
        if save:
            with open(self.legal_actions_file_name, "wb") as f:
                pickle.dump(self.legal_actions_masks, f)
            with open(self.obs_idx_map_file_name, "wb") as f:
                pickle.dump(self.obs_idx_map, f)
    