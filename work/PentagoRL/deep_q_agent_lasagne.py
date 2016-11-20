# -*- coding: utf-8 -*-
import numpy as np
import lasagne
import theano
import theano.tensor as T

class DeepQAgentLasagne(object):    
    def __init__(self, player, env, 
                 discount, learning_rate, rho, rms_epsilon, 
                 batch_size, rng, input_scale=2.0):
        self.env = env
        obs = self.env.reset()
        self.input_width = obs.shape[0]
        self.input_height = obs.shape[1]
        self.input_shape = obs.shape                
        self.num_actions = self.env.action_space.n
        
        self.discount = discount
        self.lr = learning_rate
        self.rho = rho
        self.rms_epsilon = rms_epsilon
        
        self.batch_size = batch_size
        self.rng = rng        
        self.input_scale = input_scale
        
        self.setup()
        

    def act(self, obs, verbose=False, eps=None):
        if eps is None:
            eps = self.config["eps"]
        # epsilon greedy.
        return
        

    def learn(self):
        return
        
        
    def setup(self):
        lasagne.random.set_rng(self.rng)
        
        self.update_counter = 0

        self.l_out = self.build_q_network()              
               
        states = T.tensor3('states')
        next_states = T.tensor3('next_states')
        rewards = T.col('rewards')
        actions = T.icol('actions')
        terminals = T.icol('terminals')
        
        # Shared variables for training from a minibatch of replayed
        # state transitions, each consisting of an observation,
        # along with the chosen action and resulting
        # reward and terminal status.
        self.states_shared = theano.shared(
            np.zeros((self.batch_size, self.input_height, self.input_width), dtype=theano.config.floatX))
        self.rewards_shared = theano.shared(
            np.zeros((self.batch_size, 1), dtype=theano.config.floatX),
            broadcastable=(False, True))
        self.actions_shared = theano.shared(
            np.zeros((self.batch_size, 1), dtype='int32'),
            broadcastable=(False, True))
        self.terminals_shared = theano.shared(
            np.zeros((self.batch_size, 1), dtype='int32'),
            broadcastable=(False, True))
        
        # Shared variable for a single state, to calculate q_vals.
        self.state_shared = theano.shared(
            np.zeros((self.input_height, self.input_width), dtype=theano.config.floatX))

        # Formulas
        q_vals = lasagne.layers.get_output(self.l_out, states / self.input_scale)
        
        next_q_vals = lasagne.layers.get_output(self.l_out, next_states / self.input_scale)
        next_q_vals = theano.gradient.disconnected_grad(next_q_vals)
        
        terminalsX = terminals.astype(theano.config.floatX)
        action_mask = T.eq(T.arange(self.num_actions).reshape((1, -1)),
                          actions.reshape((-1, 1))).astype(theano.config.floatX)

        target = (rewards +
                  (T.ones_like(terminalsX) - terminalsX) *
                  self.discount * T.max(next_q_vals, axis=1, keepdims=True))
        output = (q_vals * action_mask).sum(axis=1).reshape((-1, 1))
        diff = target - output

        loss = 0.5 * diff ** 2
        loss = T.sum(loss)
        #loss = T.mean(loss)

        # Params and givens            
        params = lasagne.layers.helper.get_all_params(self.l_out)  
        updates = lasagne.updates.rmsprop(loss, params, self.lr, self.rho, self.rms_epsilon)
        train_givens = {
            states: self.states_shared[:, :-1],
            next_states: self.imgs_shared[:, 1:],
            rewards: self.rewards_shared,
            actions: self.actions_shared,
            terminals: self.terminals_shared
        }
        self._train = theano.function([], [loss], updates=updates,
                                      givens=train_givens)
        q_givens = {
            states: self.state_shared.reshape((1,
                                               self.input_height,
                                               self.input_width))
        }
        self._q_vals = theano.function([], q_vals[0], givens=q_givens)
        
        
    def train(self, states, actions, rewards, terminals):
        self.states_shared.set_value(states)
        self.actions_shared.set_value(actions)
        self.rewards_shared.set_value(rewards)
        self.terminals_shared.set_value(terminals)
        loss = self._train()
        self.update_counter += 1
        return np.sqrt(loss)        
        
        
    def q_vals(self, state):
        self.state_shared.set_value(state)
        return self._q_vals()

        
    def choose_action(self, state, epsilon):
        if self.rng.rand() < epsilon:
            return self.rng.randint(0, self.num_actions)
        q_vals = self.q_vals(state)
        return np.argmax(q_vals)        
        
    
    def build_q_network(self):        
        l_in = lasagne.layers.InputLayer(
            shape=self.input_shape
        )
        
        l_hidden1 = lasagne.layers.DenseLayer(
            l_in,
            num_units=self.action_n,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Normal(.01),
            b=lasagne.init.Constant(.1)
        )

        l_out = lasagne.layers.DenseLayer(
            l_hidden1,
            num_units=self.action_n,
            nonlinearity=lasagne.nonlinearities.softmax,
            W=lasagne.init.Normal(.01),
            b=lasagne.init.Constant(.1)
        )

        return l_out        