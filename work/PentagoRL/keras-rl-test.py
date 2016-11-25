import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten
from keras.optimizers import RMSprop
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from pentago_env import PentagoEnv

SIZE = 6
ENV_NAME = "Pentago-6"

def main():
    np.random.seed(123)    
    env = PentagoEnv(SIZE)
    env.seed(123)
    nb_actions = env.action_space.n

    model = Sequential()
    #model.add(Reshape((SIZE ** 2,), input_shape=(SIZE, SIZE)))
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='sigmoid'))
    model.add(Dense(nb_actions))
    print(model.summary())

    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    memory = SequentialMemory(limit=5000, window_length=1)
    policy = BoltzmannQPolicy()
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=1000,
                    target_model_update=1e-2, policy=policy)
    optimizer=RMSprop(lr=0.00025, epsilon=0.01)
    dqn.compile(optimizer)

    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    dqn.fit(env, nb_steps=50000, visualize=True, verbose=1)

    # After training is done, we save the final weights.
    dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)    

if __name__ == '__main__':
    main()