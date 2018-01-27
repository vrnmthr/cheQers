#!/usr/bin/env python

import tensorflow as tf
import numpy as np
from abc import ABCMeta, abstractmethod

class QLearner(ABCMeta):
    """
    Abstract base class defining generalized Q-learning algorithm
    """

    def __init__(self, Lambda, alpha = 0.1, dim, acts):
        """
        Lambda = discount factor
        alpha = learning rate
        dim = dimension of vector describing state
        acts = number of actions
        """
        self.Lambda = Lambda
        self.alpha = alpha
        self.dim = dim
        self.acts = acts

        #sets up the network
        tf.reset_default_graph()
        # first layer of inputs
        self.inputs1 = tf.placeholder(tf.float32,shape=[1,dim])
        # weights for network
        self.W = tf.get_variable("weights",[dim,acts],
            dtype=ft.float32, initializer=tf.random_uniform_initializer)
        # result of one layer of computation
        self.Qout = tf.matmult(self.inputs1, self.W)
        self.result = tf.argmax(self.Qout)

        # result of 
        self.nextQ = tf.placeholder(tf.float32,shape=[1,4])
        loss = tf.reduce_sum(tf.square(nextQ - Qout))
        trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
        updateModel = trainer.minimize(loss)

    @abstractmethod
    def reward(self, parameter_list):
        raise NotImplementedError

    @abstractmethod
    @staticmethod
    def transition(state, action):
        raise NotImplementedError

    @staticmethod
    def loss(actual, expected):
        pass

    def Q(self, state):
        """
        Returns utilities as an np array of (action, utility)
        couples. Calculated using backing neural net.
        """
        #tensorflow things here
        pass

    def step(self, state, actions, transition):
        """
        Takes as input a state of the current problem
        and a set of actions. Also contains a transition function that
        returns a state given an action to be executed.
        Returns the optimal action for the given state
        and updates the backing neural net.
        """
        acts = Q(state)
        # action w/ max utility
        max_util_a = max(acts, key = lambda x: x[1])[0]
        next_state = transition(state, max_util_a)
        next_acts = Q(next_state)
