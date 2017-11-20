#
#   Get raw picture and modify to 84*84 gray picture 
#   Modify from ZMQ example (http://zguide.zeromq.org/py:spworker)
#   
#   Author:  Kartik, Chen  <kbehouse(at)gmail(dot)com>,
#
          
import os
from random import randint
import time
import zmq
from threading import Thread
from utility import *
import cv2
import numpy as np
from config import cfg
from network.ACNet import ACNet

BACKEND_ADR  = "tcp://%s:%d" % (cfg['conn']['server_ip'],cfg['conn']['server_backend_port'])


class Worker(Thread):
    """ Init Client """
    def __init__(self, sess, worker_id, main_net = None ):
        Thread.__init__(self)
        
        self.identity = worker_id # u"Worker-{}".format(worker_id).encode("ascii")
        # Connect Init
        self.context = zmq.Context(1)
        self.worker = self.context.socket(zmq.REQ)
        self.worker.setsockopt(zmq.IDENTITY, self.identity)
        self.worker.connect(BACKEND_ADR)

        # RL Init
        self.nA = cfg['RL']['action_num']
        # DL Init
        self.sess = sess
        self.net = ACNet(self.sess, worker_id, main_net)

        self.random = np.random.RandomState(1)

        print "I: (%s) worker ready" % self.identity

    def close_connect(self):
        # print('in {} close connect'.format(self.identity) )
        try:
            self.worker.setsockopt(zmq.LINGER, 0)
            self.worker.close()
            self.context.term()
        except Exception as e :
            print('E: Get Worker::close_connect() exception -> {}, e.errno={}'.format(e,str(e.errno)))


    # def __del__(self):
    #     print('in del____')

    def policy(self, s):
        ones = np.ones(self.nA)
        p = ones / self.nA
        return p



    def predict(self, state, greedy = True, epsilon = 0.01):
        """ 
            greedy = True for Train; False for only Predict Run
            type(state)=<type 'numpy.ndarray'>
        """
        
        actions = self.net.choose_action(state)
        
        #Process Image
        # pi = self.policy(state)
        # if greedy :
        #     actions = self.random.randint(0,self.nA)
        #     # if( self.random.rand() < epsilon ) :
        #     #     actions = self.random.randint(0,self.nA)
        #     # else :
        #     #     actions = np.argmax(pi)
        # else :
        #     actions = [ self.random.choice(self.nA, 1, p=p)[0] for p in pi ]
        

        # print('pi={}, actions={}'.format(pi,actions))

        return actions


    def train(self,tag_id, states, actions, rewards, next_state, done ):
        # print('train states.shape={}, type(states)={}'.format(np.shape(states), type(states)))
        # print('train actions.shape={}, type(actions)={}'.format(np.shape(actions), type(actions)))
        # print('train rewards.shape={}, type(rewards)={}'.format(np.shape(rewards), type(rewards)))
        # print('done = {}'.format(done)) 
       

        if cfg['RL']['method'] == "A3C":
            # print('train buffer_s.shape={}, type(buffer_s)={}'.format(np.shape(buffer_s), type(buffer_s)))
            # print('train next_state.shape={}, type(next_state)={}'.format(np.shape(next_state), type(next_state)))
            # next_state (7,)  ,  next_state[np.newaxis, :] (1, 7)
            if done:
                v_s_ = 0   # terminal
            else:
                v_s_ = self.sess.run(self.net.v, {self.net.s: next_state[np.newaxis, :]})[0, 0]
            buffer_v_target = []
            for r in rewards[::-1]:    # reverse buffer r
                v_s_ = r + cfg['A3C']['GAMMA'] * v_s_
                buffer_v_target.append(v_s_)
            buffer_v_target.reverse()
            

            buffer_s, buffer_a, buffer_v_target = np.vstack(states), np.vstack(actions), np.vstack(buffer_v_target)

            # print('train: vstack buffer_s.shape={}, type(buffer_s)={}'.format(np.shape(buffer_s), type(buffer_s)))
            # print('train: vstack buffer_a.shape={}, type(buffer_a)={}'.format(np.shape(buffer_a), type(buffer_a)))
            # print('train: vstack buffer_v_target.shape={}, type(buffer_v_target)={}'.format(np.shape(buffer_v_target), type(buffer_v_target)))
        

            feed_dict = {
                self.net.s: buffer_s,
                self.net.a_his: buffer_a,
                self.net.v_target: buffer_v_target,
            }
            test = self.net.update_global(feed_dict)
            # buffer_s, buffer_a, buffer_r = [], [], []
            self.net.pull_global()

    def run(self):
        self.worker.send(LRU_READY)

        cycles = 0
        while True:
            # reda data and blocking
            try:
                client_id, _ , msg   = self.worker.recv_multipart()
            except Exception as e :
                print('E: Get self.worker.recv_multipart() exception ->  {}'.format(e))
                self.close_connect()
                break
                
            if not msg:
                print('E: Worker say  msg is None')
                break
            # unpack msg
            load = loads(msg)

            # print('len(load) = {}'.format(len(load)))
            seq = load[0]
            cmd = load[1]

            # print("I: [{}]: Get [{}]'s cmd:({}) , seq:({}) ".format(self.identity, client_id, cmd, seq) )

            if PREDICT_CMD == cmd:
                state = load[2]
                actions = self.predict(state)
                msg = dumps( (seq, actions) )
                self.worker.send_multipart([client_id, _ , msg ])

                # print("I: [{}]: Get [{}]'s cmd:({}) , seq:({}), state.shape: {}, ".\
                #     format(self.identity, client_id, cmd, seq, state.shape) )

            elif TRAIN_CMD == cmd:
                state  = load[2]
                action = load[3]
                reward = load[4]
                next_state = load[5]
                done   = load[6]

                tag_id = "{}+{}".format(client_id, seq)
                self.train(tag_id, state, action ,reward, next_state, done )
                msg = dumps( seq )
                self.worker.send_multipart([client_id, _ , msg ])

                # print("I: [{}]: Get [{}]'s cmd:({}) , seq:({}), state.shape: {}, reward: {}, action: {} ".\
                #     format(self.identity, client_id, cmd, seq, np.shape(state), reward, action) )

            