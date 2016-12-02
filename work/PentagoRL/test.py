# -*- coding: utf-8 -*-
import numpy as np
from pentago_env import PentagoEnv
from tabular_q_agent import TabularQAgent
from random_agent import RandomAgent
from ipyparallel import Client

SIZE = 4
AGENT_STARTS = True
AGENT_TAG = "Player 1_4x4_4 to win"

EPOCHS = 1000
EPISODES_PR_EPOCH = 10000
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
    return (reset_obs, feedbacks)


def train_agent(agent, reset_obs, feedbacks):
    agent.reset()
    obs = reset_obs
    for (action, obs_next, reward, done, info) in feedbacks:
        agent.learn(obs, action, obs_next, reward, done, info)
        obs = obs_next


def harvest_experience(episodes):
    opponent_policy = RandomAgent("Player 2 Random")
    opponent_policy.seed(12345)
    env = PentagoEnv(SIZE, opponent_policy, agent_starts = AGENT_STARTS, to_win=SIZE)
    env.seed(67890)

    exploration_policy = RandomAgent("Player 1 Random")
    exploration_policy.seed(24680)
    exploring_agent = TabularQAgent(env, AGENT_TAG, exploration_policy, load_model=True, userconfig={ "eps" : 0.3 })
    exploring_agent.seed(13579)

    experience = []
    for e in range(episodes):
        experience += simulate_episode(env, exploring_agent)
    return experience


#@profile
def main():
    env = PentagoEnv(SIZE, None, agent_starts = AGENT_STARTS, to_win=SIZE)
    #env.seed(67890)
    exploiting_agent = TabularQAgent(env, AGENT_TAG, None, load_model=True, userconfig={ "eps" : 1.0 })
    #exploiting_agent.seed(13579)

    rc = Client()
    dview = rc[:]
    nb_nodes = len(rc.ids)
    episodes_pr_node = EPISODES_PR_EPOCH / nb_nodes
    with dview.sync_imports():
        import pentago_env
   
    #env.monitor.start(AGENT_TAG)
    for e in range(EPOCHS):
        experience = dview.map_sync(lambda x: harvest_experience(episodes_pr_node), range(nb_nodes))
        
        for (reset_obs, feedback) in experience:
            train_agent(exploiting_agent, reset_obs, feedbacks)

        verbose = True # e >= EPISODES_VERBOSE_INTERVAL and e % EPISODES_VERBOSE_INTERVAL == 0
        if verbose:
            print("\n Episode {}".format(e))
            final_obs = None
            for (action, obs_next, reward, done, info) in feedbacks:
                print("Action: {}, Reward: {}".format(action, reward))
                final_obs = obs_next
            print(final_obs[0])
            agent.render() 

        save = True # e >= EPISODES_SAVE_MODEL_INTERVAL and e % EPISODES_SAVE_MODEL_INTERVAL == 0
        if save:
            agent.save()
    
    #env.monitor.close()
    env.close()


if __name__ == "__main__":
    main()