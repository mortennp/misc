import datetime
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import RMSprop
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from pentago_env import PentagoEnv

SEED = 123
SIZE = 6
AGENT_STARTS = True
TAG = "Pentago-size{}-agent_starts{}".format(SIZE, int(AGENT_STARTS))

def main():
    # Create env
    np.random.seed(SEED)    
    env = PentagoEnv(SIZE, agent_starts = AGENT_STARTS)
    env.seed(SEED)
    nb_actions = env.action_space.n

    # Define model
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='sigmoid'))
    model.add(Dense(nb_actions))
    print(model.summary())

    # Configure and compile  agent
    memory = SequentialMemory(limit=5000, window_length=1)
    policy = BoltzmannQPolicy()
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=1000,
                    target_model_update=1000, policy=policy)
    optimizer=RMSprop(lr=0.00025, epsilon=0.01)
    dqn.compile(optimizer)

    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    dqn.fit(env, nb_steps=50000, visualize=True, verbose=1)

    # After training is done, we save the final weights.
    dqn.save_weights('weights/dqn-{}-weights-{}.h5f'.format(TAG, datetime.datetime.now()))    

if __name__ == '__main__':
    main()