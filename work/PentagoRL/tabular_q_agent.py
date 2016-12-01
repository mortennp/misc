# -*- coding: utf-8 -*-
from os import path
import numpy as np
import pandas as pd
import pickle

#https://github.com/openai/gym/blob/master/examples/agents/tabular_q_agent.py

class TabularQAgent(object):
    TOTAL_REWARD_KEY = "TotalReward"
    DELTA_KEY = "Delta"

    def __init__(self, env, tag, exploration_policy, load_model=True, **userconfig):
        self.env = env
        self.action_n = self.env.action_space.n
        self.tag = tag
        self.exploration_policy = exploration_policy

        self.config = {            
            "init_mean" : 0.0,      # Initialize Q values with this mean
            "init_std" : 0.1,       # Initialize Q values with this standard deviation
            "learning_rate" : 0.5,
            "eps": 0.5,            # Epsilon in epsilon greedy policies
            "discount": 0.98
            }        
        self.config.update(userconfig)

        self.history = None
        
        self.q_file_name = "tabular-q_{}.p".format(tag)
        if load_model and path.isfile(self.q_file_name):
            print "'{}' Loading from {}".format(self.tag, self.q_file_name)
            with open(self.q_file_name, "rb") as f:
                self.q = pickle.load(f)
        else:
            self.q = dict()
            
        
    def reset(self):
        self.actions = []
        self.total_reward = 0
            

    def get_action(self, obs, eps=None):
        if eps is None: eps = self.config["eps"]         
        if np.random.random() > eps:
            # Exploit
            (_, legal_actions_mask, state_key) = obs
            self.default_q(state_key)
            legal_actions = np.ma.array(self.q[state_key], mask=np.logical_not(legal_actions_mask))
            buf = np.argwhere(legal_actions == legal_actions.max())
            buf_idx = np.random.randint(0,len(buf))
            action = buf[buf_idx][0]
            return action
        else:
            # Explore
            return self.exploration_policy.get_action(obs)
        
        

    def learn(self, obs, action, obs_next, reward, done, info):
        (_, next_state_legal_actions_mask, next_state_key) = obs_next
        self.default_q(next_state_key)
        next_legal_actions = np.ma.array(self.q[next_state_key], mask=np.logical_not(next_state_legal_actions_mask))
        future = 0.0 if done else next_legal_actions.max()
        (_, _, state_key) = obs
        self.default_q(state_key)
        delta = self.q[state_key][action] - (reward + self.config["discount"] * future)
        self.q[state_key][action] -= self.config["learning_rate"] * delta

        self.actions.append(action)
        self.total_reward += reward
        if done:
            if self.history is None:
                self.history = pd.DataFrame(columns=[self.TOTAL_REWARD_KEY, self.DELTA_KEY,] + info.keys())
            info[self.TOTAL_REWARD_KEY] = self.total_reward
            info[self.DELTA_KEY] = abs(delta)
            self.history.loc[len(self.history)] = info
            
            
    def render(self):        
        print("'{}' Actions: {}".format(self.tag, self.actions))
        print("'{}' Mean Total Reward: {}".format(self.tag, self.history[-1000:][self.TOTAL_REWARD_KEY].mean()))
        print("'{}' Mean Absolute Delta: {}".format(self.tag, self.history[-1000:][self.DELTA_KEY].mean()))
        print(self.history[-1000:]["Result"].value_counts())        

    
    def save(self):
        print("'{}' States seen: {}".format(self.tag,len(self.q)))
        #with h5py.File(self.q_file_name, "w") as f:
        #    for state_key, action_values in self.q.items():
        #            f["/" + state_key] = action_values
        with open(self.q_file_name + "2", "wb") as f:
            pickle.dump(self.q, f, protocol=2)


    def default_q(self, key):
        if not key in self.q:
            self.q[key] = self.config["init_std"] * np.random.randn(self.action_n) + self.config["init_mean"]