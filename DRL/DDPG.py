
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

class DDPG(DRL):
    def __init__(self, sess,  scope, globalAC=None):
        self.sess = sess

        self.lr_a = cfg['DDPG']['LR_A']
        self.lr_c = cfg['DDPG']['LR_C']        
        self.gamma = cfg['DDPG']['GAMMA']
        self.update_nn_step = cfg['DDPG']['UPDATE_NN_STEP']  
        self.memory_size = cfg['DDPG']['MENORY_SIZE']
        self.batch_size = cfg['DDPG']['BATCH_SIZE']

        if scope != cfg['DDPG']['main_net_scope']:
            self.learn_step_counter = 0
            self.memory = np.zeros((self.memory_size, 2 * cfg['RL']['state_shape'][0] + cfg['RL']['action_num'] + 1))
            self.S = tf.placeholder(tf.float32, [None, cfg['RL']['state_shape'][0]], 's')
            self.S_ = tf.placeholder(tf.float32, [None, cfg['RL']['state_shape'][0]], 's_')
            self.R = tf.placeholder(tf.float32, [None, 1], 'r')            
            self._build_net()
            self.sess.run(tf.global_variables_initializer())
            self.cost_his = []

    def _build_net(self):
        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            self.a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            self.q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            self.q_ = self._build_c(self.S_, self.a_, scope='target', trainable=False)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        with tf.variable_scope('target_q'):
            self.target_q = self.R + self.gamma * self.q_

        with tf.variable_scope('TD_error'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.target_q, self.q))

        with tf.variable_scope('C_train'):
            self.train_op_c = tf.train.RMSPropOptimizer(self.lr_c).minimize(self.loss)

        with tf.variable_scope('a_grad'):
            self.a_grads = tf.gradients(self.q, self.a)[0]   # tensor of gradients of each sample (None, a_dim)

        with tf.variable_scope('policy_grads'):
            self.policy_grads = tf.gradients(ys=self.a, xs=self.ae_params, grad_ys=self.a_grads) # a_grads

        with tf.variable_scope('A_train'):
            opt = tf.train.RMSPropOptimizer(-self.lr_a)  # (- learning rate) for ascent policy
            self.train_op_a = opt.apply_gradients(zip(self.policy_grads, self.ae_params))

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.contrib.layers.xavier_initializer()
            init_b = tf.constant_initializer(0.001)
            net = tf.layers.dense(s, 200, activation=tf.nn.relu6,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l1',
                                  trainable=trainable)
            net = tf.layers.dense(net, 200, activation=tf.nn.relu6,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l2',
                                  trainable=trainable)
            net = tf.layers.dense(net, 10, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l3',
                                  trainable=trainable)
            with tf.variable_scope('a'):
                actions = tf.layers.dense(net, cfg['RL']['action_num'], activation=tf.nn.tanh, kernel_initializer=init_w,
                                          name='a', trainable=trainable)
                scaled_a = tf.multiply(actions, cfg['RL']['action_bound'][1], name='scaled_a')  # Scale output to -action_bound to action_bound
        return scaled_a

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.contrib.layers.xavier_initializer()
            init_b = tf.constant_initializer(0.01)

            with tf.variable_scope('l1'):
                n_l1 = 200
                w1_s = tf.get_variable('w1_s', [cfg['RL']['state_shape'][0], n_l1], initializer=init_w, trainable=trainable)
                w1_a = tf.get_variable('w1_a', [cfg['RL']['action_num'], n_l1], initializer=init_w, trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
                net = tf.nn.relu6(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net = tf.layers.dense(net, 200, activation=tf.nn.relu6,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l2',
                                  trainable=trainable)
            net = tf.layers.dense(net, 10, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l3',
                                  trainable=trainable)
            with tf.variable_scope('q'):
                q = tf.layers.dense(net, 1, kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)   # Q(s,a)
        return q

    def choose_action(self, s):
        s = s[np.newaxis, :]
        return self.sess.run(self.a, feed_dict={self.S: s})[0]

    #def add_grad_to_graph(self): # a_grads
    #    with tf.variable_scope('policy_grads'):
    #        self.policy_grads = tf.gradients(ys=self.a, xs=self.ae_params, grad_ys=self.a_grads) # a_grads

    #    with tf.variable_scope('A_train'):
    #        opt = tf.train.RMSPropOptimizer(-self.lr_a)  # (- learning rate) for ascent policy
    #        self.train_op_a = opt.apply_gradients(zip(self.policy_grads, self.ae_params))

    def _replace_target_params(self):
        self.sess.run([tf.assign(t, e) for t, e in zip(self.at_params, self.ae_params)])
        self.sess.run([tf.assign(t, e) for t, e in zip(self.ct_params, self.ce_params)])

    def train(self, states, actions, rewards, next_state, done):
        #if done:
        #    break
        #else:
        var = 2.
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        #print(states.shape)
        #print(actions.shape)
        #print([rewards])
        #print(next_state.shape)
        transition = np.hstack((states, actions, [rewards], next_state))
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

            var = max([var*.9999, 0.1]) # VAR_MIN
            b_s = batch_memory[:, :cfg['RL']['state_shape'][0]]
            b_a = batch_memory[:, cfg['RL']['state_shape'][0]: cfg['RL']['state_shape'][0] + cfg['RL']['action_num']]
            b_r = batch_memory[:, -cfg['RL']['state_shape'][0] - 1: -cfg['RL']['state_shape'][0]]
            b_s_ = batch_memory[:, -cfg['RL']['state_shape'][0]:]

            self.sess.run(self.train_op_c, feed_dict={self.S: b_s, self.a: b_a, self.R: b_r, self.S_: b_s_})
            self.sess.run(self.train_op_a, feed_dict={self.S: b_s})

            self.learn_step_counter += 1


