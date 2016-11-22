# -*- coding: utf-8 -*-
from pentago_env import PentagoEnv
from tabular_q_agent import TabularQAgent
from random_agent import RandomAgent
from deep_q_agent_keras import DeepQAgentKeras
from tatsuyaokubo_dqn_agent import TatsuyaokuboDqnAgent

def main():
    episodes = 1000000
    episodes_verbose_interval = 1000
    episodes_save_model_interval = 10000
    
    env = PentagoEnv()    
    agent1 = TatsuyaokuboDqnAgent(env.action_space.n, "Tatsuyaokubo-1", load_model=False) # DeepQAgentKeras(env, "Deep-1", load_model=True) #TabularQAgent(env, "1", unpickle=True)
    agent2 = RandomAgent(env, "Random-2", load_model=False)
    
    for e in range(episodes):
        agent1.reset()
        agent2.reset()
        obs = env.reset()
        done = False
        info = None
        verbose = e % episodes_verbose_interval == 0
        save = e % episodes_save_model_interval == 0
        if verbose: print("\n Episode {}".format(e))
        while not done:
            action1 = agent1.act(obs, verbose)
            obs_next, reward, done, info = env.step(action1)
            agent1.learn(obs, action1, obs_next, reward, done, info, verbose)
            obs = obs_next
            if not done:
                action2 = agent2.act(obs, False)
                obs_next, reward, done, info = env.step(action2)
                agent2.learn(obs, action1, obs_next, reward, done, info, verbose)
                obs = obs_next
            if verbose: print(obs)
        if verbose: print(info)
        agent1.trace(verbose, save)
        agent2.trace(verbose, False)


if __name__ == '__main__':
    main()