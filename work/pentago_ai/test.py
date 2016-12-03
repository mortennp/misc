# -*- coding: utf-8 -*-
from sys import maxint
import numpy as np
import  ipyparallel as ipp
from gym.utils import seeding
from pentago_ai import PentagoEnv, TabularQAgent, Episode


SIZE = 4
AGENT_STARTS = True
AGENT_TAG = "Player 1_4x4_4 to win"

BASE_SEED = 12345
EPOCHS = 1000
EPOCHS_VERBOSE_INTERVAL = 1
EPOCHS_SAVE_MODEL_INTERVAL = 100
EPISODES_PR_EPOCH = 100000


def simulate_episode(env, agent):
    from pentago_ai import Episode

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
    return Episode(reset_obs, feedbacks)


def simulate_episodes(nb_episodes):
    from sys import maxint
    from gym.utils import seeding
    from pentago_ai import RandomAgent, PentagoEnv, TabularQAgent, Episode

    CHUNK_SIZE = 1000

    np_random, _ = seeding.np_random(base_seeds[0])

    opponent_policy = RandomAgent("Player 2 Random")
    opponent_policy.seed(np_random.randint(maxint))
    env = PentagoEnv(SIZE, opponent_policy, agent_starts = AGENT_STARTS, to_win=SIZE)
    env.seed(np_random.randint(maxint))

    exploration_policy = RandomAgent("Player 1 Random")
    exploration_policy.seed(np_random.randint(maxint))
    exploring_agent = TabularQAgent(env, AGENT_TAG, exploration_policy, load_model=True, userconfig={ "eps" : 0.3 })
    exploring_agent.seed(np_random.randint(maxint))

    episodes = []
    for epi in range(1, nb_episodes):
        episodes.append(simulate_episode(env, exploring_agent))
        if epi % CHUNK_SIZE == 0:
            yield episodes
            episodes = []
    if len(episodes) > 0:
        yield episodes 


def train_agent(agent, reset_obs, feedbacks):
    agent.reset()
    obs = reset_obs
    for (action, obs_next, reward, done, info) in feedbacks:
        agent.learn(obs, action, obs_next, reward, done, info)
        obs = obs_next    


def remote_generator(view, f, *args, **kwargs):
    """Run a generator function remotely, returning a generator that yields in the same way."""
    # First, create the iterator by calling our function remotely
    def _create_iterator(f, *a, **kw):
        """Creates an iterator, storing it in the interactive namespace as _iterator"""
        g = globals()
        generator = f(*a, **kw)
        g['_iterator'] = iter(generator)
    
    # Create the remote iterator
    view.apply_sync(_create_iterator, f, *args, **kwargs)
    
    # then iterate over it remotely by applying the `next` builtin
    r_iterator = ipp.Reference('_iterator')
    while True:
        try:
            yield view.apply_sync(next, r_iterator)
        except ipp.RemoteError as e:
            if e.ename == 'StopIteration':
                raise StopIteration()
            else:
                raise


#@profile
def main():
    np_random, seed = seeding.np_random(BASE_SEED)
    print("Base seed: {}, derived seed: {}".format(BASE_SEED, seed))

    rc = ipp.Client()
    dview = rc[:]
    nb_nodes = len(rc.ids)
    dview.push({
        "simulate_episode": simulate_episode,
        "SIZE" : SIZE,
        "AGENT_STARTS": AGENT_STARTS,
        "AGENT_TAG": AGENT_TAG})

    env = PentagoEnv(SIZE, None, agent_starts = AGENT_STARTS, to_win=SIZE)
    env.seed(np_random.randint(maxint))
    exploiting_agent = TabularQAgent(env, AGENT_TAG, None, load_model=True, userconfig={ "eps" : 1.0 })
    exploiting_agent.seed(np_random.randint(maxint))   
    #env.monitor.start(AGENT_TAG)
    episodes_pr_node = EPISODES_PR_EPOCH / nb_nodes    
    for epoch in range(1, EPOCHS):
        verbose = epoch >= EPOCHS_VERBOSE_INTERVAL and epoch % EPOCHS_VERBOSE_INTERVAL == 0
        if verbose:
            print("\n Epoch {}".format(epoch))

        dview.scatter('base_seeds', np_random.randint(maxint, size=nb_nodes))                
        epi = 1
        for result in remote_generator(dview, simulate_episodes, episodes_pr_node):
            for episodes in result:
                for episode in episodes:
                    if verbose: print("    episode: {}\r".format(epi))
                    train_agent(exploiting_agent, episode.reset_obs, episode.feedbacks)
                    epi += 1

        if verbose:
            # final_obs = None
            # for (action, obs_next, reward, done, info) in feedbacks:
            #     print("Action: {}, Reward: {}".format(action, reward))
            #     final_obs = obs_next
            # print(final_obs[0])
            exploiting_agent.render() 

        save = epoch >= EPOCHS_SAVE_MODEL_INTERVAL and epoch % EPOCHS_SAVE_MODEL_INTERVAL == 0
        if save:
            print("\n Saving model... {}")
            exploiting_agent.save()
    
    #env.monitor.close()
    env.close()


if __name__ == "__main__":
    main()