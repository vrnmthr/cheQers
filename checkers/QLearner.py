#!/usr/bin/env python

import tensorflow as tf
import numpy as np
from abc import ABCMeta, abstractmethod

class QLearner(ABCMeta):
    """
    Abstract base class defining generalized Q-learning algorithm
    """

    def __init__(self, Lambda, alpha):
        """
        Lambda = discount factor
        alpha = learning rate
        """
        self.Lambda = Lambda
        self.alpha = alpha

    @abstractmethod
    def reward(self, parameter_list):
        raise NotImplementedError

    @staticmethod
    def loss(actual, expected):
        pass

    def Q(self, state):
        """
        Returns utilities as an np array of (action, utility)
        couples. Calculated using backing neural net.
        """
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
        opt = max(acts, key = lambda x: x[1])[0]
        next_state = transition(state, opt)
        next_acts = Q(next_state)
        next_util
