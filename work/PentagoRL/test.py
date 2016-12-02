# -*- coding: utf-8 -*-
import numpy as np
from pentago_env import PentagoEnv
from tabular_q_agent import TabularQAgent
from random_agent import RandomAgent

SEED = 123
SIZE = 4
AGENT_STARTS = True
AGENT_TAG = "Player 1_4x4_4 to win"

EPISODES = 1000000
EPISODES_VERBOSE_INTERVAL = 1000
EPISODES_SAVE_MODEL_INTERVAL = 100000

#@profile
def main():    
    np.random.seed(SEED)
    opponent_policy = RandomAgent("Player 2 Random")    
    env = PentagoEnv(SIZE, opponent_policy, agent_starts = AGENT_STARTS, to_win=SIZE)
    #env.monitor.start(AGENT_TAG)
    env.seed(SEED)
    nb_actions = env.action_space.n

    agent = TabularQAgent(env, AGENT_TAG, opponent_policy, load_model=True)
    
    for e in range(EPISODES):
        verbose = e % EPISODES_VERBOSE_INTERVAL == 0
        save = e % EPISODES_SAVE_MODEL_INTERVAL == 0
        if verbose: print("\n Episode {}".format(e))
        obs = env.reset()
        done = False
        info = {}
        agent.reset()
        while not done:
            action = agent.get_action(obs)
            obs_next, reward, done, info = env.step(action)
            agent.learn(obs, action, obs_next, reward, done, info)
            obs = obs_next            
        if verbose:
            print(info)
            env.render(mode="human")
            agent.render() 
        if save:
            agent.save()
    
    #env.monitor.close()
    env.close()


if __name__ == "__main__":
    main()