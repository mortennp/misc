# -*- coding: utf-8 -*-
import numpy as np
from pentago_env import PentagoEnv
from tabular_q_agent import TabularQAgent
from random_agent import RandomAgent

SIZE = 4
AGENT_STARTS = True
AGENT_TAG = "Player 1_4x4_4 to win"

EPISODES = 1000000
EPISODES_VERBOSE_INTERVAL = 1000
EPISODES_SAVE_MODEL_INTERVAL = 100000

#@profile
def main():    
    opponent_policy = RandomAgent("Player 2 Random")
    opponent_policy.seed(12345)

    env = PentagoEnv(SIZE, opponent_policy, agent_starts = AGENT_STARTS, to_win=SIZE)
    env.seed(67890)

    agent = TabularQAgent(env, AGENT_TAG, opponent_policy, load_model=True)
    agent.seed(13579)
    
    #env.monitor.start(AGENT_TAG)
    for e in range(EPISODES):
        verbose = e >= EPISODES_VERBOSE_INTERVAL and e % EPISODES_VERBOSE_INTERVAL == 0
        save = e >= EPISODES_SAVE_MODEL_INTERVAL and e % EPISODES_SAVE_MODEL_INTERVAL == 0
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