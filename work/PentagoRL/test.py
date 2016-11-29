# -*- coding: utf-8 -*-
import numpy as np
from pentago_env import PentagoEnv
from tabular_q_agent import TabularQAgent
from random_agent import RandomAgent

SEED = 123
SIZE = 4
AGENT_STARTS = True

EPISODES = 1000000
EPISODES_VERBOSE_INTERVAL = 10000
EPISODES_SAVE_MODEL_INTERVAL = 100000

def main():    
    np.random.seed(SEED)
    opponent_policy = RandomAgent("Player 2 Random")    
    env = PentagoEnv(SIZE, opponent_policy, agent_starts = AGENT_STARTS)
    env.seed(SEED)
    nb_actions = env.action_space.n

    agent = TabularQAgent(env, "Player 1 Tabular Q", opponent_policy, load_model=False)
    
    for e in range(EPISODES):
        verbose = e % EPISODES_VERBOSE_INTERVAL == 0
        save = e % EPISODES_SAVE_MODEL_INTERVAL == 0
        agent.reset()
        obs = env.reset()
        done = False
        info = {}
        if verbose: print("\n Episode {}".format(e))
        while not done:
            action = agent.get_action(obs, verbose)
            obs_next, reward, done, info = env.step(action)
            agent.learn(obs, action, obs_next, reward, done, info, verbose)
            obs = obs_next            
        if verbose:
            print(obs) 
            print(info)
        agent.trace(verbose, save)


if __name__ == '__main__':
    main()