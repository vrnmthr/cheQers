#!/usr/bin/env python

import tensorflow as tf
import numpy as np
from abc import ABCMeta, abstractmethod

class QLearner(ABCMeta):
    """
    Abstract base class defining generalized Q-learning algorithm
    """

    def __init__(self, Lambda, alpha = 0.1, epsilon, dim, acts):
        """
        Lambda = discount factor
        alpha = learning rate
        dim = dimension of vector describing state
        epsilon = chance of not following greedy action
        acts = number of actions
        """
        self.Lambda = Lambda
        self.alpha = alpha
        self.epsilon = epsilon
        self.dim = dim
        self.acts = acts

        #sets up the network
        tf.reset_default_graph()
        self.init = tf.initialize_all_variables()

        # first layer of inputs
        self.inputs1 = tf.placeholder(tf.float32,shape=[1,dim])
        # weights for network
        self.weights = tf.get_variable("weights",[dim,acts],
            dtype=ft.float32, initializer=tf.random_uniform_initializer)
        # result of one layer of computation
        self.Qout = tf.matmult(self.inputs1, self.weights)
        # index of greatest value => index of optimal action
        self.result = tf.argmax(self.Qout)

        # result of next Q values used in Bellman update equation
        self.nextQ = tf.placeholder(tf.float32,shape=[1,acts])
        self.loss = tf.reduce_sum(tf.square(nextQ - Qout))
        self.trainer = tf.train.GradientDescentOptimizer(alpha)
        self.updateModel = self.trainer.minimize(self.loss)


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

        #updates backing net
        with tf.Session() as sess:

            sess.run(self.init)

            # calculates the values of result and Qout in TF, using the
            # inputs defined in feed_dict and filling them in a, allQ
            a_opt, allQ = sess.run([self.result,self.Qout],
                feed_dict={inputs1:state})

            # unpacks tensor a_opt
            a_opt = a_opt[0]

            # generates random action with probability epsilon
            if np.random.rand(1) < self.epsilon:
                i = np.random.randint(0,len(actions),size=1)[0]
                a_opt = actions[i]

            # TO-DO: THE WAY THIS CURRENTLY RUNS IS THAT A IS A ARRAY
            # OF SIZE 1 WHERE THE SINGLE ELEMENT IS THE INDEX OF THE ELEMENT
            # IN ALLQ THAT HAS THE HIGHEST UTILITY => MAKES THE ASSUMPTION
            # THAT ACTIONS ARE REPRESENTED AS INTEGERS. THIS WILL
            # CURRENTLY ALL BREAK

            # get new state and reward by executing preferred action
            new_state, reward = transition(state, a_opt)

            # Obtain Q1 values through network
            Q1 = sess.run(self.Qout,feed_dict={inputs1:new_state})
            # find maximum utility for new_state
            maxQ1 = np.max(Q1)

            # implements temporal difference equation by updating the
            # score of the action we picked in targetQ. Everything else
            # stays the same so is unaffected
            targetQ = allQ
            targetQ[a_opt][0] = reward + self.Lambda*maxQ1

            # Train our network using target and predicted Q values
            _,_ = sess.run([self.updateModel,self.W],
                feed_dict={inputs1:state,nextQ:targetQ})

            return a_opt
