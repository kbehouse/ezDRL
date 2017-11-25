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
from DRL.Base import RL, DRL
from DRL.A3C import A3C
from DRL.TD import SARSA, QLearning

BACKEND_ADR  = "tcp://%s:%d" % (cfg['conn']['server_ip'],cfg['conn']['server_backend_port'])

class Worker(Thread):
    """ Init Client """
    def __init__(self, worker_id, sess = None, main_net = None ):
        Thread.__init__(self)
        
        self.identity = worker_id # u"Worker-{}".format(worker_id).encode("ascii")
        # Connect Init
        self.context = zmq.Context(1)
        self.worker = self.context.socket(zmq.REQ)
        self.worker.setsockopt(zmq.IDENTITY, self.identity)
        self.worker.connect(BACKEND_ADR)

        # RL Init
        self.nA = cfg['RL']['action_num']
        method_class = globals()[cfg['RL']['method'] ]


        # DL Init
        if issubclass(method_class, DRL):
            self.sess = sess
            self.RL = method_class(self.sess, worker_id, main_net)
        elif issubclass(method_class, RL):
            self.RL = method_class()
            pass
        else:
            print('E: Worker::__init__() say error method name={}'.format(cfg['RL']['method'] ))

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

    def predict(self, state):
        return self.RL.choose_action(state)

    def train(self,tag_id, states, actions, rewards, next_state, done ):
        # print('I: train get states.shape={}, type(states)={}'.format(np.shape(states), type(states)))
        # print('I: train get actions.shape={}, type(actions)={}'.format(np.shape(actions), type(actions)))
        # print('I: train get rewards.shape={}, type(rewards)={}'.format(np.shape(rewards), type(rewards)))
        # print('I: train get done = {}'.format(done)) 
       
        self.RL.train(states, actions, rewards, next_state, done)
        
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
            seq = load[0]
            cmd = load[1]

            # print("I: [{}]: Get [{}]'s cmd:({}) , seq:({}) ".format(self.identity, client_id, cmd, seq) )

            if PREDICT_CMD == cmd:
                ''' Predict Section'''
                state = load[2]
                # print("I: [{}]: Get [{}]'s cmd:({}) , seq:({}), state.shape: {}, ".\
                #     format(self.identity, client_id, cmd, seq, np.shape(state)) )

                # print("I: state.shape: {}, type(state)= {} ".\
                #     format(np.shape(state), type(state)) )


                actions = self.predict(state)
                msg = dumps( (seq, actions) )
                self.worker.send_multipart([client_id, _ , msg ])

                
            elif TRAIN_CMD == cmd:
                ''' Train Section'''
                state  = load[2]
                action = load[3]
                reward = load[4]
                next_state = load[5]
                done   = load[6]

                # print("I: [{}]: Get [{}]'s cmd:({}) , seq:({}), state.shape: {}, reward: {}, action: {} ".\
                #     format(self.identity, client_id, cmd, seq, np.shape(state), reward, action) )

                # print("I: [{}]: seq:({}) reward.shape: {}, action.shape: {} ".\
                #     format(self.identity, seq, np.shape(reward), np.shape(action)) )

                tag_id = "{}+{}".format(client_id, seq)
                self.train(tag_id, state, action ,reward, next_state, done )

                ''' Send Finish Train'''
                msg = dumps( seq )
                self.worker.send_multipart([client_id, _ , msg ])

               

            