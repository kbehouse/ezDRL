# 
#   First Version: https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/10_A3C/A3C_continuous_action.py
#             

import tensorflow as tf
import numpy as np
from config import cfg
from Base import DRL


class A3C(DRL):
    def __init__(self, sess, scope, globalAC=None):
        super(A3C, self).__init__()

        self.sess = sess
        self.OPT_A = tf.train.RMSPropOptimizer(cfg['A3C']['LR_A'], name='RMSPropA')
        self.OPT_C = tf.train.RMSPropOptimizer(cfg['A3C']['LR_C'], name='RMSPropC')

        self.discrete = cfg['RL']['action_discrete']
        # print("I: A3C Use {} Graph".format( 'Discrete' if self.discrete else 'Continuous'))

        if scope == cfg['A3C']['main_net_scope']:   # get global network
            self.build_main_net(scope)
        else:   # local net, calculate losses
            self.build_worker_net(scope, globalAC)
    
    def build_main_net(self,scope):
        with tf.variable_scope(scope):
            self.s = tf.placeholder(tf.float32, [None, cfg['RL']['state_shape'][0] ], 'S')
            self._build_net()
            self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
            self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')

    def build_worker_net(self, scope, globalAC):      # worker(local) net
        with tf.variable_scope(scope):
            a_type = tf.int32 if self.discrete else tf.float32
            self.a_his = tf.placeholder(a_type, [None, ], 'A')
            self.s = tf.placeholder(tf.float32, [None, cfg['RL']['state_shape'][0] ], 'S')
            self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

            if self.discrete:  #discrete
                self.a_prob, self.v = self._build_net()
            else: # continuous
                mu, sigma, self.v = self._build_net()

            # ======= Critic Loss ====== #
            td = tf.subtract(self.v_target, self.v, name='TD_error')
            with tf.name_scope('c_loss'):
                self.c_loss = tf.reduce_mean(tf.square(td))

            # ======= Actor Loss ====== #
            if self.discrete:   #discrete
                with tf.name_scope('a_loss'):
                    tf_log = tf.log(self.a_prob)
                    # print("cfg['RL']['action_num']={},type={}".format(cfg['RL']['action_num'], type(cfg['RL']['action_num'])) )
                    tf_one_hot = tf.one_hot(self.a_his, cfg['RL']['action_num'], dtype=tf.float32)
                    log_prob = tf.reduce_sum(tf_log * tf_one_hot, axis=1, keep_dims=True)
                    # log_prob = tf.reduce_sum(tf.log(self.a_prob) * tf.one_hot(self.a_his, cfg['RL']['action_num'], dtype=tf.float32), axis=1, keep_dims=True)
                    exp_v = log_prob * td
                    entropy = -tf.reduce_sum(self.a_prob * tf.log(self.a_prob + 1e-5),
                                             axis=1, keep_dims=True)  # encourage exploration
                    self.exp_v = cfg['A3C']['ENTROPY_BETA'] * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)
            else:   # continuous
                with tf.name_scope('wrap_a_out'):
                    # self.test = sigma[0]
                    mu, sigma = mu * cfg['RL']['action_bound'][1], sigma + 1e-5

                normal_dist = tf.contrib.distributions.Normal(mu, sigma)

                with tf.name_scope('a_loss'):
                    log_prob = normal_dist.log_prob(self.a_his)
                    
                    exp_v = log_prob * td
                    
                    entropy = normal_dist.entropy()  # encourage exploration
                    self.exp_v = cfg['A3C']['ENTROPY_BETA'] * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope('choose_a'):  # use local params to choose action
                    self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=0), *cfg['RL']['action_bound'])
                
            
            with tf.name_scope('local_grad'):
                self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
                self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
                self.a_grads = tf.gradients(self.a_loss, self.a_params)
                self.c_grads = tf.gradients(self.c_loss, self.c_params)

        with tf.name_scope('sync'):
            with tf.name_scope('pull'):
                self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
            with tf.name_scope('push'):
                self.update_a_op = self.OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                self.update_c_op = self.OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))


    def _build_net(self):
        w_init = tf.contrib.layers.xavier_initializer()

        if self.discrete:   
            # with tf.variable_scope('actor'):
            #     l_a = tf.layers.dense(self.s, 200, tf.nn.relu6, kernel_initializer=w_init, name='la')
            #     a_prob = tf.layers.dense(l_a, cfg['RL']['action_num'], tf.nn.softmax, kernel_initializer=w_init, name='ap')
            # with tf.variable_scope('critic'):
            #     l_c = tf.layers.dense(self.s, 100, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            #     v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # state value

            with tf.variable_scope('actor'):
                l_a = tf.layers.dense(self.s, 400, tf.nn.relu6, kernel_initializer=w_init, name='la')
                l_a = tf.layers.dense(l_a, 300, tf.nn.relu6, kernel_initializer=w_init, name='la2')
                a_prob = tf.layers.dense(l_a, cfg['RL']['action_num'], tf.nn.softmax, kernel_initializer=w_init, name='ap')
            with tf.variable_scope('critic'):
                l_c = tf.layers.dense(self.s, 400, tf.nn.relu6, kernel_initializer=w_init, name='lc')
                l_c = tf.layers.dense(l_c, 200, tf.nn.relu6, kernel_initializer=w_init, name='lc2')
                v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # state value
                
            return a_prob, v
        else:
            # print("I: A3C Use Continuous Graph")
            with tf.variable_scope('actor'):
                l_a = tf.layers.dense(self.s, 400, tf.nn.relu6, kernel_initializer=w_init, name='la')
                l_a = tf.layers.dense(l_a, 300, tf.nn.relu6, kernel_initializer=w_init, name='la2')
                mu = tf.layers.dense(l_a, cfg['RL']['action_num'], tf.nn.tanh, kernel_initializer=w_init, name='mu')
                sigma = tf.layers.dense(l_a, cfg['RL']['action_num'], tf.nn.softplus, kernel_initializer=w_init, name='sigma')
            with tf.variable_scope('critic'):
                l_c = tf.layers.dense(self.s, 400, tf.nn.relu6, kernel_initializer=w_init, name='lc')
                l_c = tf.layers.dense(l_c, 200, tf.nn.relu6, kernel_initializer=w_init, name='lc2')
                v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # state value

            return mu, sigma, v
            
        

    def update_global(self, feed_dict):  # run by a local
        self.sess.run([self.update_a_op, self.update_c_op], feed_dict)  # local grads applies to global net

    def pull_global(self):  # run by a local
        self.sess.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s):  # run by a local
        s = np.array(s)
        if self.discrete:
            prob_weights = self.sess.run(self.a_prob, feed_dict={self.s: s[np.newaxis, :]})
            action = np.random.choice(range(prob_weights.shape[1]),
                                    p=prob_weights.ravel())  # select action w.r.t the actions prob
            return action
        else:
            s = s[np.newaxis, :]
            return self.sess.run(self.A, {self.s: s})[0]

    def train(self,  states, actions, rewards, next_state, done):
        '''
            NOTE about action:
            ------Discrete--------#
              action_discrete: yes
              action_num: 3
              train_run_steps: 5
                  train action shape: (1~5,)   (1~5 depend on done, not done = 5)
                  all number (0, 1, or 2)
            ------Continuous--------#
              action_discrete: yes
              action_num: 3
              action_bound: [-1,1]
              train_run_steps: 5 
                  train action shape: (5,3)  (1~5 depend on done, not done = 5)
                  all number between [-1,1]                           
        '''
        if np.shape(states)[0] <=0:
            print("W: Get 0 states.shape={},done={}".format(np.shape(states,done) )   )
            return

        if done:
            v_s_ = 0   # terminal
        else:
            next_state = np.array(next_state)
            v_s_ = self.sess.run(self.v, {self.s: next_state[np.newaxis, :]})[0, 0]
        buffer_v_target = []
        for r in rewards[::-1]:    # reverse buffer r
            v_s_ = r + cfg['A3C']['gamma'] * v_s_
            buffer_v_target.append(v_s_)
        buffer_v_target.reverse()
        
        buffer_a = np.array(actions) if self.discrete else np.vstack(actions)
        buffer_s, buffer_v_target = np.vstack(states), np.vstack(buffer_v_target)

        feed_dict = {
            self.s: buffer_s,
            self.a_his: buffer_a,
            self.v_target: buffer_v_target,
        }

        self.update_global(feed_dict)
        self.pull_global()