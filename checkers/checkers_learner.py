#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import copy


class CheQer:
    """
    Abstract base class defining generalized Q-learning algorithm
    """

    def __init__(self, Lambda, epsilon, dim, alpha = 0.1):
        """
        Lambda = discount factor
        alpha = learning rate
        dim = dimension of vector describing state
        epsilon = chance of not following greedy action
        """
        self.SAVE_STEP_NUM = 100  # modulus of steps to save
        self.SAVE_DIREC = "./models/"
        self.SAVE_FILE = self.SAVE_DIREC + "checkers-model.ckpt"
        self.Lambda = Lambda
        self.alpha = alpha
        self.epsilon = epsilon
        self.dim = dim

        # sets up the network
        tf.reset_default_graph()

        # first layer of inputs
        self.inputs1 = tf.placeholder(tf.float32, shape=[1, dim])
        # weights for network
        self.weights = tf.get_variable("weights", [dim, 1], dtype=tf.float32, initializer=tf.random_uniform_initializer)
        self.init = tf.global_variables_initializer()
        # result of one layer of computation
        self.Qout = tf.matmul(self.inputs1, self.weights)

        # result of next Q values used in Bellman update equation
        self.nextQ = tf.placeholder(tf.float32,shape=[1,1])
        self.loss = tf.reduce_sum(tf.square(self.nextQ - self.Qout))
        self.trainer = tf.train.GradientDescentOptimizer(alpha)
        self.updateModel = self.trainer.minimize(self.loss)

        self.train_step = 0
        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        # restore saved graph + variables
        file = tf.train.latest_checkpoint(self.SAVE_DIREC)
        print("Loading model from %s" % file)
        self.saver.restore(self.sess, file)

    def __del__(self):
        print("Saving model to %s" % self.SAVE_FILE)
        self.saver.save(self.sess, self.SAVE_FILE)
        self.sess.close()

    @staticmethod
    def simulate(board, move):
        """
        Calculates reward, applies chosen move to board, then returns
        board.

        :param board: the board
        :param a_opt: index of chosen move
        :return: the board, now with a_opt's move applied. Also the reward.
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

    def step(self, board, possible_moves):
        """
        Takes as input a state of the current problem
        and a set of actions. Also contains a transition function that
        returns a state given an action to be executed.
        Returns the optimal action for the given state
        and updates the backing neural net.
        """

        self.sess.run(self.init)

        # Store the "base" state before board gets modified
        state = board.board_arr
        actions = board.available_white_moves()

        # initialize array of scores of all moves
        allQ = np.zeros([len(actions)])

        # find scores of all moves
        for i in range(len(actions)):
            # calculates board state after move
            future_state = copy.deepcopy(board)
            future_state.apply_white_move(actions[i])
            future_state = future_state.board_arr

            # calculates the value of Qout in TF (using the
            # inputs defined in feed_dict) and places it in allQ
            future_state.shape = (1, 64)
            allQ[i] = self.sess.run(self.Qout, feed_dict={self.inputs1: future_state})

        # get index of best-scored move
        a_opt = tf.reshape(tf.argmax(allQ), [-1]).eval(session=self.sess)[0]

        # generates random action with probability epsilon
        if np.random.rand(1) < self.epsilon:
            a_opt = np.random.randint(0,len(actions),size=1)[0]

        # get new state and reward by executing preferred action
        board, reward = self.simulate(board, actions[a_opt])

        if not reward:
            # switch the perspective to simulate the opponent's move
            board.set_white_player((board.cur_white_player+1) % 2)
            op_actions = board.available_white_moves()

            # initialize array of scores of all opponent moves
            op_q = np.zeros([len(op_actions)])

            # find scores of all opponent moves
            for i in range(len(op_actions)):
                # calculates board state after opponent move
                op_state = copy.deepcopy(board)
                op_state.apply_white_move(op_actions[i])
                op_state = op_state.board_arr

                # calculates the value of op_q in TF (using the
                # inputs defined in feed_dict) and places it in op_q
                op_state.shape = (1, 64)
                op_q[i] = self.sess.run(self.Qout, feed_dict={self.inputs1: op_state})

            # get index of best-scored opponent move
            op_a_opt = tf.reshape(tf.argmax(op_q), [-1]).eval(session=self.sess)[0]
            # apply opponent's best move for use
            board, rew = self.simulate(board, op_actions[op_a_opt])

            if not rew:
                # switch the perspective back to our's
                board.set_white_player((board.cur_white_player+1)%2)
                predic_actions = board.available_white_moves()

                # initialize array of scores of all moves
                predic_q = np.zeros([len(predic_actions)])

                # find scores of all moves
                for i in range(len(predic_actions)):
                    # calculates board state after move
                    predic_state = copy.deepcopy(board)
                    predic_state.apply_white_move(predic_actions[i])
                    predic_state = predic_state.board_arr

                    # calculates the value of Qout in TF (using the
                    # inputs defined in feed_dict) and places it in allQ
                    predic_state.shape = (1, 64)
                    predic_q[i] = self.sess.run(self.Qout, feed_dict={self.inputs1: predic_state})

                predic_a_opt = tf.reshape(tf.argmax(predic_q), [-1]).eval(session=self.sess)[0]
                _, rew2 = self.simulate(board, predic_actions[predic_a_opt])
                # find maximum utility for new_state
                if rew2:
                    maxQ1 = rew2
                else:
                    maxQ1 = np.max(predic_q)
            elif rew == -.25:
                maxQ1 = rew
            else:
                maxQ1 = -1 * rew
        else:
            maxQ1 = 0

        # implements temporal difference equation by updating the
        # score of the action we picked in targetQ. Everything else
        # stays the same so is unaffected
        targetQ = np.array(reward + self.Lambda*maxQ1)
        targetQ.shape = (1, 1)
        state.shape = (1, 64)

        # Train our network using target and predicted Q values
        self.sess.run([self.updateModel, self.weights],
            feed_dict={self.inputs1: state, self.nextQ: targetQ})

        # save some subset of networks
        if self.train_step % self.SAVE_STEP_NUM == 0:
            self.saver.save(self.sess, self.SAVE_FILE)
        self.train_step += 1

        return a_opt

    def print_info(self):
        print("Weights:\n%s" % self.weights.eval(session=self.sess))