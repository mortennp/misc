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


def simulate_episode(env, agent):
    feedbacks = []
    reset_obs = env.reset()
    done = False
    obs = reset_obs
    agent.reset()    
    while not done:
        action = agent.get_action(obs)
        obs_next, reward, done, info = env.step(action)
        feedbacks += [(action, obs_next, reward, done, info)]
        obs = obs_next
    return reset_obs, feedbacks


def train_agent(agent, reset_obs, feedbacks):
    agent.reset()
    obs = reset_obs
    for (action, obs_next, reward, done, info) in feedbacks:
        agent.learn(obs, action, obs_next, reward, done, info)
        obs = obs_next


#@profile
def main():    
    opponent_policy = RandomAgent("Player 2 Random")
    opponent_policy.seed(12345)
    env = PentagoEnv(SIZE, opponent_policy, agent_starts = AGENT_STARTS, to_win=SIZE)
    env.seed(67890)

    exploration_policy = RandomAgent("Player 1 Random")
    exploration_policy.seed(24680)
    agent = TabularQAgent(env, AGENT_TAG, exploration_policy, load_model=True, userconfig={ "eps" : 0.3 })
    agent.seed(13579)
    
    #env.monitor.start(AGENT_TAG)
    for e in range(EPISODES):
        reset_obs, feedbacks = simulate_episode(env, agent)
        train_agent(agent, reset_obs, feedbacks)

        verbose = e >= EPISODES_VERBOSE_INTERVAL and e % EPISODES_VERBOSE_INTERVAL == 0
        if verbose:
            print("\n Episode {}".format(e))
            final_obs = None
            for (action, obs_next, reward, done, info) in feedbacks:
                print("Action: {}, Reward: {}".format(action, reward))
                final_obs = obs_next
            print(final_obs[0])
            agent.render() 

        save = e >= EPISODES_SAVE_MODEL_INTERVAL and e % EPISODES_SAVE_MODEL_INTERVAL == 0
        if save:
            agent.save()
    
    #env.monitor.close()
    env.close()


if __name__ == "__main__":
    main()