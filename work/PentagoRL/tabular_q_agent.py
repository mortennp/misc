# -*- coding: utf-8 -*-
from os import path
import numpy as np
import hashlib
import pickle
from gym.spaces import prng

#https://github.com/openai/gym/blob/master/examples/agents/tabular_q_agent.py

class TabularQAgent(object):    
    def __init__(self, env, tag, load_model=True, **userconfig):
        self.env = env
        self.tag = tag
        self.action_n = self.env.action_space.n
        self.config = {            
            "init_mean" : 0.0,      # Initialize Q values with this mean
            "init_std" : 0.1,       # Initialize Q values with this standard deviation
            "learning_rate" : 0.8,
            "eps": 0.3,            # Epsilon in epsilon greedy policies
            "discount": 0.98
            }        
        self.config.update(userconfig)
        
        self.q_file_name = "tabular-q-{}.p".format(tag)
        self.obs_idx_map_file_name = "tabular-obs_idx_map-{}.p".format(tag)
        if load_model and path.isfile(self.q_file_name) and path.isfile(self.obs_idx_map_file_name):
            print "'{}' Loading from {} and {}".format(self.tag, self.q_file_name, self.obs_idx_map_file_name)
            with open(self.q_file_name, "rb") as f:
                self.q = pickle.load(f)
            with open(self.obs_idx_map_file_name, "rb") as f:
                self.obs_idx_map = pickle.load(f)
                self.obs_idx_num = len(self.obs_idx_map)
        else:
            self.q = dict()
            self.obs_idx_num = 0
            self.obs_idx_map = dict()
            
        
    def reset(self):
        self.actions = []
        self.total_reward = 0
            

    def act(self, obs, verbose=False, eps=None):
        if eps is None: eps = self.config["eps"] # epsilon greedy.
        actions = self.q[self.get_obs_idx(obs)]
        if np.random.random() > eps:
            buf = np.argwhere(actions == np.amax(actions))
            buf_idx = prng.np_random.randint(len(buf))
            action = buf[buf_idx][0]
        else:
            action = self.env.action_space.sample()
        return action
        

    def learn(self, obs, action, obs_next, reward, done, info, verbose=False):
        self.actions.append(action)
        self.total_reward += reward
        future = 0.0 if done else np.max(self.q[self.get_obs_idx(obs_next)])
        obs_idx = self.get_obs_idx(obs)
        self.q[obs_idx][action] -= \
            self.config["learning_rate"] * (self.q[obs_idx][action] - reward - self.config["discount"] * future)
            
            
    def trace(self, verbose=False, save=False):
        if verbose:
            print("'{}' Total reward: {}".format(self.tag, self.total_reward))
            print("'{}' Actions: {}".format(self.tag, self.actions))
        if save:
            print("'{}' States seen: {}".format(self.tag,len(self.q)))
            with open(self.q_file_name, "wb") as f:
                pickle.dump(self.q, f)
            with open(self.obs_idx_map_file_name, "wb") as f:
                pickle.dump(self.obs_idx_map, f)
            

    def get_obs_idx(self, obs):
        h = self.get_hash_key(obs)
        if not h in self.obs_idx_map:
            self.obs_idx_map[h] = self.obs_idx_num
            self.q[self.obs_idx_num] = self.config["init_std"] * np.random.randn(self.action_n) + self.config["init_mean"]
            self.obs_idx_num += 1            
        return self.obs_idx_map[h]
        
                        
    def get_hash_key(self, obs):
        return hashlib.sha1(obs.view(np.uint8)).hexdigest() #return np.array_str(obs)