#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import copy


class CheQer:
    """
    Abstract base class defining generalized Q-learning algorithm
    """

    def __init__(self, Lambda, epsilon, hidden_dims, alpha = 0.1, self_training=True):
        """
        Lambda = discount factor
        alpha = learning rate
        dim = dimension of vector describing state
        epsilon = chance of not following greedy action
        """
        self.SAVE_STEP_NUM = 100000  # modulus of steps to save (100,000 currently)
        self.SAVE_DIREC = "./models/"
        self.SAVE_FILE = self.SAVE_DIREC + "checkers-model"
        self.Lambda = Lambda
        self.alpha = alpha
        self.epsilon = epsilon
        self.hidden_dims = hidden_dims
        self.self_training = self_training
        self.old_state = [None] * (2 if self_training else 1)

        # sets up the network
        tf.reset_default_graph()

        # create model
        self.inputs1, self.Qout, self.weights, self.biases = self.build_mlp(hidden_dims)
        self.init = tf.global_variables_initializer()

        # placeholder for old Q value for training
        self.placeholder_q = tf.placeholder(tf.float32, shape=[1, 1])

        self.train_step = 0
        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        # restore saved graph + variables - https://www.tensorflow.org/programmers_guide/saved_model
        file = tf.train.latest_checkpoint(self.SAVE_DIREC)
        if file is not None:
            print("Loading model from %s" % file)
            self.saver = tf.train.import_meta_graph(self.SAVE_FILE + ".meta")
            self.saver.restore(self.sess, file)

        self.sess.run(self.init)

        # result of next Q values used in Bellman update equation
        self.loss = tf.reduce_sum(tf.square(self.placeholder_q - self.Qout))
        self.trainer = tf.train.GradientDescentOptimizer(alpha)
        self.updateModel = self.trainer.minimize(self.loss)

        # finalize structure
        tf.get_default_graph().finalize()

    def __del__(self):
        print("Saving model to %s" % self.SAVE_FILE)
        self.saver.save(self.sess, self.SAVE_FILE)
        self.sess.close()

    @staticmethod
    def build_mlp(hidden_dimensions):
        """
        Must provide at least one hidden layer dimension

        :param hidden_dimensions: A list of ints with at least one element. All elements must be greater than zero.
        :return:
        """
        hidden_layer_count = len(hidden_dimensions)
        assert hidden_layer_count > 0

        inputs = tf.placeholder(tf.float32, shape=[1, 64])

        weights = [None] * (hidden_layer_count + 1)
        biases = [None] * (hidden_layer_count + 1)

        weights[0] = tf.get_variable("input_weights", [64, hidden_dimensions[0]], dtype=tf.float32, initializer=tf.random_uniform_initializer)
        biases[0] = tf.get_variable("input_bias", [hidden_dimensions[0]], dtype=tf.float32, initializer=tf.random_uniform_initializer)

        output = tf.add(tf.matmul(inputs, weights[0]), biases[0])

        for i in range(hidden_layer_count):
            dim = 1
            if not i+1 == hidden_layer_count:
                dim = hidden_dimensions[i + 1]
            weights[i + 1] = tf.get_variable("weights_" + str(i), [hidden_dimensions[i], dim], dtype=tf.float32, initializer=tf.random_uniform_initializer)
            biases[i + 1] = tf.get_variable("bias_" + str(i), [1, dim], dtype=tf.float32, initializer=tf.random_uniform_initializer)

            output = tf.add(tf.matmul(output, weights[i+1]), biases[i+1])

        return inputs, output, weights, biases

    @staticmethod
    def simulate(board, move):
        """
        Calculates reward, applies chosen move to board, then returns
        board.

        :param board: the board
        :param move: chosen move
        :return: the board, now with move applied. Also the reward.
        """
        reward = 0

        board.apply_white_move(move)
        if board.cur_player_won() == 1:
            reward = 1
        elif board.cur_player_won() == 2:
            reward = -1
        elif board.cur_player_won() == 0:
            reward = -.25

        return board, reward

    def train(self, reward, cur_q):
        # implements temporal difference equation by updating the
        # score of the action we picked in target_q. Everything else
        # stays the same so is unaffected
        old_state = self.old_state[1 if self.self_training else 0]
        if old_state is not None:
            target_q = np.array(reward + self.Lambda*cur_q)
            target_q.shape = (1, 1)
            old_state.shape = (1, 64)

            # Train our network using target and predicted Q values
            _, _ = self.sess.run([self.updateModel, self.loss],
                feed_dict={self.inputs1: old_state, self.placeholder_q: target_q})

    def find_optimal_move(self, board):
        actions = board.available_white_moves()

        # initialize array of scores of all moves
        all_q = np.zeros([len(actions)])

        # find scores of all moves
        for i in range(len(actions)):
            # calculates board state after move
            future_state = copy.deepcopy(board)
            future_state.apply_white_move(actions[i])
            future_state = future_state.board_arr

            # calculates the value of Qout in TF (using the
            # inputs defined in feed_dict) and places it in all_q
            future_state.shape = (1, 64)
            all_q[i] = self.sess.run(self.Qout, feed_dict={self.inputs1: future_state})

        # get index of best-scored move
        a_opt = self.argmax(all_q)

        return a_opt, all_q

    @staticmethod
    def argmax(num_list):
        cur_max = num_list[0]
        max_index = 0
        for i in range(len(num_list)):
            cur_val = num_list[i]
            if cur_val > cur_max:
                cur_max = cur_val
                max_index = i
        return max_index

    def step(self, board, possible_moves):
        """
        Takes as input a state of the current problem
        and a set of actions. Also contains a transition function that
        returns a state given an action to be executed.
        Returns the optimal action for the given state
        and updates the backing neural net.
        """
        # Store the "base" state before board gets modified
        state = board.board_arr
        actions = board.available_white_moves()

        a_opt, allQ = self.find_optimal_move(board)

        # generates random action with probability epsilon
        if np.random.rand(1) < self.epsilon:
            a_opt = np.random.randint(0, len(actions), size=1)[0]

        # get new state and reward by executing preferred action
        board, reward = self.simulate(board, actions[a_opt])

        self.train(reward, allQ[a_opt])

        # save some subset of networks
        if self.train_step % self.SAVE_STEP_NUM == 0:
            self.saver.save(self.sess, self.SAVE_FILE)
        self.train_step += 1

        if self.self_training:
            self.old_state[1] = self.old_state[0]
        self.old_state[0] = state

        return a_opt

    def print_info(self):
        print("Biases:\n%s" % self.biases[len(self.hidden_dims)].eval(session=self.sess))
        print("Weights:\n%s" % self.weights[len(self.hidden_dims)].eval(session=self.sess))
