# https://github.com/tatsuyaokubo/dqn/blob/master/dqn.py

import os
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Reshape
from keras.optimizers import RMSprop
from keras import backend as kb

FRAME_WIDTH = 6  # Resized frame width
FRAME_HEIGHT = 6  # Resized frame height
GAMMA = 0.99  # Discount factor
EXPLORATION_STEPS = 1000000  # Number of steps over which the initial value of epsilon is linearly annealed to its final value
INITIAL_EPSILON = 1.0  # Initial value of epsilon in epsilon-greedy
FINAL_EPSILON = 0.1  # Final value of epsilon in epsilon-greedy
INITIAL_REPLAY_SIZE = 1000  # Number of steps to populate the replay memory before training starts
NUM_REPLAY_MEMORY = 400000  # Number of replay memory the agent uses for training
BATCH_SIZE = 32  # Mini batch size
TARGET_UPDATE_INTERVAL = 10000  # The frequency with which the target network is updated
TRAIN_INTERVAL = 4  # The agent selects 4 actions between successive updates
LEARNING_RATE = 0.00025  # Learning rate used by RMSProp
MOMENTUM = 0.95  # Momentum used by RMSProp
MIN_GRAD = 0.01  # Constant added to the squared gradient in the denominator of the RMSProp update


class TatsuyaokuboDqnAgent():
    def __init__(self, num_actions, tag, load_model=False):
        self.num_actions = num_actions
        self.tag = tag

        # Setup annealing
        self.epsilon = INITIAL_EPSILON
        self.epsilon_step = (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORATION_STEPS
        self.t = 0

        # Create q network
        self.q_network = self.build_network()
        # Create target network
        self.target_network = self.build_network()

        # Load network weights
        self.network_file_name = "{}.h5".format(self.tag)
        if load_model and os.path.isfile(self.network_file_name):
            self.q_network = keras.models.load_model(self.network_file_name)

        # Initialize target network
        self.update_target_network()

        # Create replay memory
        self.states = np.empty((NUM_REPLAY_MEMORY, FRAME_WIDTH, FRAME_HEIGHT))
        self.actions = np.empty((NUM_REPLAY_MEMORY,))
        self.rewards = np.empty((NUM_REPLAY_MEMORY,))
        self.next_states = np.empty((NUM_REPLAY_MEMORY, FRAME_WIDTH, FRAME_HEIGHT))
        self.dones = np.empty((NUM_REPLAY_MEMORY,))
        self.ys = np.empty((NUM_REPLAY_MEMORY,))
        self.replay_idx = 0


    def build_network(self):
        model = Sequential()
        model.add(Reshape((FRAME_WIDTH * FRAME_HEIGHT,), input_shape=(FRAME_WIDTH, FRAME_HEIGHT)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.num_actions))
        model.compile(optimizer=RMSprop(lr=LEARNING_RATE, epsilon=MIN_GRAD), loss='mse')
        return model


    def update_target_network(self):
        self.target_network.set_weights(self.q_network.get_weights())


    def reset(self):
        return


    def act(self, state, verbose=False):
        if self.epsilon >= random.random() or self.t < INITIAL_REPLAY_SIZE:
            action = random.randrange(self.num_actions)
        else:        
            action = np.argmax(self.q_network.predict_on_batch([state]))

        # Anneal epsilon linearly over time
        if self.epsilon > FINAL_EPSILON and self.t >= INITIAL_REPLAY_SIZE:
            self.epsilon -= self.epsilon_step

        return action


    def learn(self, state, action, next_state, reward, done, info, verbose=False):
        # Store transition in replay memory
        self.replay_memory_upsert(state, action, reward, next_state, done)

        # Train
        if self.t >= INITIAL_REPLAY_SIZE:
            # Train network
            if self.t % TRAIN_INTERVAL == 0:
                self.train_network()

            # Update target network
            if self.t % TARGET_UPDATE_INTERVAL == 0:
                self.update_target_network()
                self.ys = self.calculate_target_ys(self.states, self.actions, self.rewards, self.next_states, self.dones)

        self.t += 1


    def replay_memory_upsert(self, state, action, reward, next_state, done):                
        self.states[self.replay_idx] = state
        self.actions[self.replay_idx] = action
        self.rewards[self.replay_idx] = reward
        self.next_states[self.replay_idx] = next_state
        self.dones[self.replay_idx] = done
        self.ys[self.replay_idx] = self.calculate_target_ys(np.array([state]), np.array([action]), np.array([reward]), np.array([next_state]), np.array([done]))
        self.replay_idx = (self.replay_idx + 1) % NUM_REPLAY_MEMORY


    def calculate_target_ys(self, states, actions, rewards, next_states, dones):
        terminals = dones + 0 # Convert True to 1, False to 0        
        target_q_values = self.target_network.predict_on_batch(next_states)
        ys = rewards + (1 - terminals) * GAMMA * np.max(target_q_values, axis=1)
        return ys


    def train_network(self): 
        # Sample random minibatch of transition from replay memory
        mask = np.random.choice(2, min((self.t, NUM_REPLAY_MEMORY)))
        hist = self.q_network.train_on_batch(self.states[mask], self.ys[mask])


    def trace(self, verbose=False, save=False):
        if verbose:
            if self.t < INITIAL_REPLAY_SIZE:
                mode = 'random'
            elif INITIAL_REPLAY_SIZE <= self.t < INITIAL_REPLAY_SIZE + EXPLORATION_STEPS:
                mode = 'explore'
            else:
                mode = 'exploit'
            print('{}'.format(mode))
            #print('TIMESTEP: {1:8d} / EPSILON: {3:.5f} / TOTAL_REWARD: {4:3.0f} / AVG_MAX_Q: {5:2.4f} / AVG_LOSS: {6:.5f} / MODE: {7}'.format(
            #    self.t, self.epsilon,
            #    0.0, 0.0, 0.0,
            #    mode))       

        if save:
            self.q_network.save(self.network_file_name)