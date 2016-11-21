# -*- coding: utf-8 -*-
from os import path
import random
import numpy as np
from keras.models import Model
from keras.layers import Convolution2D, Dense, Flatten, Input
from keras.optimizers import RMSprop
from keras import backend as kb
#from theano.gradient import disconnected_grad

#https://github.com/sherjilozair/dqn/blob/master/dqn.py

class DeepQAgentKeras(object):
    def __init__(self, env, tag, load_model=True,
                 epsilon=0.1, mbsz=32, discount=0.9, memory=50):
        obs = env.reset()        
        self.state_size = np.multiply.reduce(obs.shape)
        #self.state_shape = obs.shape
        self.number_of_actions = env.action_space.n
        
        self.tag = tag
        self.load_model = load_model
        
        self.epsilon = epsilon
        self.mbsz = mbsz
        self.discount = discount
        self.memory = memory
        
        self.save_name = "deep-{}.h5".format(self.tag)
        
        self.build_functions()
        
        self.states = []
        self.next_states = []
        self.actions = []
        self.rewards = []
        self.terminals = []


    def build_model(self):
        S = Input(shape=(self.state_size,))
#        H = Convolution2D(16, 8, 8, subsample=(4, 4),
#            border_mode='same', activation='relu')(S)
#        H = Convolution2D(32, 4, 4, subsample=(2, 2),
#            border_mode='same', activation='relu')(H)
#        H = Flatten()(H)
        H = Dense(256, activation='relu')(S)
        V = Dense(self.number_of_actions)(H)
        self.model = Model(S, V)
        
        if self.load_model and path.isfile(self.save_name):
            print "'{}' Loading from {}".format(self.tag, self.save_name)
            self.model.load_weights(self.save_name)


    def build_functions(self):
        self.build_model()
                
        S = Input(shape=(self.state_size,))
        NS = Input(shape=(self.state_size,))
        A = Input(shape=(1,), dtype='int32')
        R = Input(shape=(1,), dtype='float32')
        T = Input(shape=(1,), dtype='int32')
        
        self.value_fn = kb.function([S], [self.model(S)])

        values = self.model(S)
        next_values = self.model(NS) #disconnected_grad(self.model(NS))
        future_value = kb.cast((1-T), dtype='float32') * kb.max(next_values, axis=1, keepdims=True)
        discounted_future_value = self.discount * future_value
        target = R + discounted_future_value
        cost = kb.mean(kb.pow(values - target, 2))
        opt = RMSprop(0.0001)
        params = self.model.trainable_weights
        updates = opt.get_updates(params, [], cost)
        self.train_fn = kb.function([S, NS, A, R, T], [cost], updates=updates)

        
    def reset(self):
        self.states = self.states[-self.memory:]
        self.next_states = self.next_states[-self.memory:]
        self.actions = self.actions[-self.memory:]
        self.rewards = self.rewards[-self.memory:]
        self.terminals = self.rewards[-self.memory:]


    def act(self, state, verbose=False):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.number_of_actions)
        else:
            values = self.value_fn([state.flatten()[None, :]])
            action = np.argmax(values)
        return action

        
    def learn(self, state, action, state_next, reward, done, info, verbose=False):
        self.states.append(state.flatten())
        self.next_states.append(state_next.flatten())
        self.actions.append(action)
        self.rewards.append(reward)
        self.terminals.append(done)
                
        self.error_cost = self.train_replay_minibatch()

        
    def train_replay_minibatch(self):
        n = len(self.states)
        
        S = np.zeros((self.mbsz, self.state_size))
        NS = np.zeros((self.mbsz, self.state_size))
        A = np.zeros((self.mbsz, 1), dtype=np.int32)
        R = np.zeros((self.mbsz, 1), dtype=np.float32)
        T = np.zeros((self.mbsz, 1), dtype=np.int32)
        
        for i in xrange(self.mbsz):
            episode = random.randint(max(0, n-self.memory), n-1)
            S[i] = self.states[episode]
            NS[i] = self.next_states[episode]
            T[i] = self.terminals[episode]
            A[i] = self.actions[episode]
            R[i] = self.rewards[episode]

        return self.train_fn([S, NS, A, R, T])
        
        
    def trace(self, verbose=False, save=False):
        if verbose:
            print("'{}' Error cost: {}".format(self.tag, self.error_cost))
        if save:
            self.model.save_weights(self.save_name, True)
        