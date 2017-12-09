
#
#   Send raw picture to server.py
#   Get gary image(84x84) from server (use worker)  
#   Save the gray image(84x84)
#   
#   Author:  Shengru, Xiao  <stemsgrpy(at)gmail(dot)com>
#      

import tensorflow as tf
import numpy as np
from config import cfg
from Base import DRL

class DuelingDQN(DRL):
    def __init__(self, sess,  scope, globalAC=None):
        self.sess = sess

        self.lr = cfg['DuelingDQN']['LR']
        self.gamma = cfg['DuelingDQN']['GAMMA']
        self.e_greedy = cfg['DuelingDQN']['E_GREEDY']
        self.update_nn_step = cfg['DuelingDQN']['UPDATE_NN_STEP']
        self.memory_size = cfg['DuelingDQN']['MENORY_SIZE']
        self.batch_size = cfg['DuelingDQN']['BATCH_SIZE']

        if scope != cfg['DuelingDQN']['main_net_scope']:
            self.learn_step_counter = 0
            self.memory = np.zeros((self.memory_size, cfg['RL']['state_shape'][0] * 2 + 2))
            self._build_net()
            self.sess.run(tf.global_variables_initializer())
            self.cost_his = []

    def _build_net(self):
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, cfg['RL']['state_shape'][0]], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, cfg['RL']['action_num']], name='q_target')  # for calculating loss
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 20, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [cfg['RL']['state_shape'][0], n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            # second layer. 
            with tf.variable_scope('Value'):
                w2 = tf.get_variable('w2', [n_l1, 1], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, 1], initializer=b_initializer, collections=c_names)
                self.V = tf.matmul(l1, w2) + b2

            with tf.variable_scope('Advantage'):
                w2 = tf.get_variable('w2', [n_l1, cfg['RL']['action_num']], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, cfg['RL']['action_num']], initializer=b_initializer, collections=c_names)
                self.A = tf.matmul(l1, w2) + b2

            # third layer. 
            with tf.variable_scope('Q'):
                self.q_eval = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keep_dims=True))     # Q = V(s) + A(s,a)                

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, cfg['RL']['state_shape'][0]], name='s_')  # input
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [cfg['RL']['state_shape'][0], n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            # second layer. 
            with tf.variable_scope('Value'):
                w2 = tf.get_variable('w2', [n_l1, 1], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, 1], initializer=b_initializer, collections=c_names)
                self.V = tf.matmul(l1, w2) + b2

            with tf.variable_scope('Advantage'):
                w2 = tf.get_variable('w2', [n_l1, cfg['RL']['action_num']], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, cfg['RL']['action_num']], initializer=b_initializer, collections=c_names)
                self.A = tf.matmul(l1, w2) + b2

            # third layer. 
            with tf.variable_scope('Q'):
                self.q_next = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keep_dims=True))     # Q = V(s) + A(s,a)                

    def choose_action(self, s):  # run by a local
        # to have batch dimension when feed into tf placeholder
        s = s[np.newaxis, :]

        if np.random.uniform() < self.e_greedy:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: s})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, cfg['RL']['action_num'])
        return action

    def _replace_target_params(self):
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.sess.run([tf.assign(t, e) for t, e in zip(t_params, e_params)])

    def train(self,  states, actions, rewards, next_state, done):
        #if done:
        #    break
        #else:
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((states, [actions, rewards], next_state))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

        if (not done) and (self.memory_counter > 1000):
            if self.learn_step_counter % self.update_nn_step == 0:
                self._replace_target_params()

            if self.memory_counter > self.memory_size:
                sample_index = np.random.choice(self.memory_size, size=self.batch_size)
            else:
                sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
            batch_memory = self.memory[sample_index, :]

            q_next, q_eval = self.sess.run(
                [self.q_next, self.q_eval],
                feed_dict={
                    self.s_: batch_memory[:, -cfg['RL']['state_shape'][0]:],
                    self.s: batch_memory[:, :cfg['RL']['state_shape'][0]]})

            q_target = q_eval.copy()

            batch_index = np.arange(self.batch_size, dtype=np.int32)
            eval_act_index = batch_memory[:, cfg['RL']['state_shape'][0]].astype(int)
            reward = batch_memory[:, cfg['RL']['state_shape'][0] + 1]

            q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

            _, self.cost = self.sess.run([self._train_op, self.loss],
                                        feed_dict={self.s: batch_memory[:, :cfg['RL']['state_shape'][0]],
                                                    self.q_target: q_target})

            self.cost_his.append(self.cost)

            #self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
            self.learn_step_counter += 1
