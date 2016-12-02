# -*- coding: utf-8 -*-
from os import path
import numpy as np
import pandas as pd
import pickle
from gym.utils import seeding

#https://github.com/openai/gym/blob/master/examples/agents/tabular_q_agent.py

class TabularQAgent(object):
    TOTAL_REWARD_KEY = "TotalReward"
    DELTA_KEY = "MeanDeltaPrStep"
    Q_TABLE_INCREMENT = 1000000

    def __init__(self, env, tag, exploration_policy, load_model=True, **userconfig):
        self.env = env
        self.action_n = self.env.action_space.n
        self.tag = tag
        self.exploration_policy = exploration_policy
        self.np_random = None

        self.config = {            
            "init_mean" : 0.0,      # Initialize Q values with this mean
            "init_std" : 0.1,       # Initialize Q values with this standard deviation
            "learning_rate" : 0.3,
            "eps": 1.0,            # Epsilon in epsilon greedy policies
            "discount": 0.98
            }        
        self.config.update(userconfig)

        self.history = None
        
        self.q_file_name = "tabular-q-action-values_{}.npy".format(tag)
        self.state_key_2_q_idx_map_file_name = "tabular-q-key-2-idx-map_{}.p2".format(tag)
        if load_model and path.isfile(self.q_file_name) and path.isfile(self.state_key_2_q_idx_map_file_name):
            print "'{}' Loading from {} and {}".format(self.tag, self.q_file_name, self.state_key_2_q_idx_map_file_name)
            self.q = np.load(self.q_file_name)
            with open(self.state_key_2_q_idx_map_file_name, "rb") as f:
                self.state_key_2_q_idx_map = pickle.load(f)            
        else:
            self.q = np.empty((self.Q_TABLE_INCREMENT, self.action_n))            
            self.state_key_2_q_idx_map = dict()
        self.append_key_idx = len(self.state_key_2_q_idx_map)


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]    
            
        
    def reset(self):
        self.actions = []
        self.total_reward = 0
        self.total_delta = 0
            

    def get_action(self, obs):        
        if self.np_random.rand() > self.config["eps"]:
            # Exploit
            (_, legal_actions_mask, state_key) = obs
            q_idx = self.default_q(state_key)
            legal_actions = np.ma.array(self.q[q_idx], mask=np.logical_not(legal_actions_mask))
            buf = np.argwhere(legal_actions == legal_actions.max())            
            action = buf[self.np_random.randint(0,len(buf))][0]
        else:
            # Explore
            action = self.exploration_policy.get_action(obs)
        return action
        
        
    def learn(self, obs, action, obs_next, reward, done, info):
        (_, next_state_legal_actions_mask, next_state_key) = obs_next
        next_q_idx = self.default_q(next_state_key)
        next_legal_actions = np.ma.array(self.q[next_q_idx], mask=np.logical_not(next_state_legal_actions_mask))
        future = 0.0 if done else next_legal_actions.max()
        (_, _, state_key) = obs
        q_idx = self.default_q(state_key)
        delta = self.q[q_idx][action] - (reward + self.config["discount"] * future)
        self.q[q_idx][action] -= self.config["learning_rate"] * delta

        self.actions.append(action)        
        self.total_reward += reward
        self.total_delta += abs(delta)

        if done:
            if self.history is None:
                self.history = pd.DataFrame(columns=[self.TOTAL_REWARD_KEY, self.DELTA_KEY,] + info.keys())
            info[self.TOTAL_REWARD_KEY] = self.total_reward
            info[self.DELTA_KEY] = self.total_delta / len(self.actions)
            self.history.loc[len(self.history)] = info
            
            
    def render(self):        
        print("'{}' Episodes Mean Total Reward: {}".format(self.tag, self.history[:][self.TOTAL_REWARD_KEY].mean()))
        print("'{}' Episodes Mean Step Abs Delta: {}".format(self.tag, self.history[:][self.DELTA_KEY].mean()))
        print(self.history[:]["Result"].value_counts())
        self.history = None

    
    def save(self):
        print("'{}' States seen: {}".format(self.tag,len(self.state_key_2_q_idx_map)))
        np.save(self.q_file_name, self.q)
        with open(self.state_key_2_q_idx_map_file_name, "wb") as f:
            pickle.dump(self.state_key_2_q_idx_map, f, protocol=2)


    def default_q(self, key):
        if not key in self.state_key_2_q_idx_map:
            idx = self.append_key_idx
            self.append_key_idx += 1
            if idx == len(self.q):
                self.q = np.append(self.q, np.zeros((self.Q_TABLE_INCREMENT, self.action_n)))
            self.state_key_2_q_idx_map[key] = idx
            random_action_values = self.config["init_std"] * self.np_random.randn(self.action_n) + self.config["init_mean"]            
            self.q[idx] = random_action_values
            return idx
        else:
            return self.state_key_2_q_idx_map[key]