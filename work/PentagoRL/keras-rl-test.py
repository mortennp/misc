import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from pentago_env import PentagoEnv

def main():
    np.random.seed(123)    
    env = PentagoEnv(6)
    env.seed(123)

    model = Sequential()
    model.add(Reshape((FRAME_WIDTH * FRAME_HEIGHT,), input_shape=(FRAME_WIDTH, FRAME_HEIGHT)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(self.num_actions))
    print(model.summary())

    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    memory = SequentialMemory(limit=50000, window_length=1)
    policy = BoltzmannQPolicy()
    dqn = DQNAgent(model=model, nb_actions=env.action_space.n, memory=memory, nb_steps_warmup=10,
                    target_model_update=1e-2, policy=policy)
    optimizer=RMSprop(lr=LEARNING_RATE, epsilon=0.01)
    dqn.compile(RMSprop)

    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    dqn.fit(env, nb_steps=50000, visualize=True, verbose=2)

    # After training is done, we save the final weights.
    dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)    

if __name__ == '__main__':
    main()