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
BACKEND_ADR  = "tcp://127.0.0.1:5556"
LRU_READY = "\x01"
ACTION_NA = 4


class Worker(Thread):
    """ Init Client """
    def __init__(self, worker_id):
        Thread.__init__(self)
        # Connect Init
        self.context = zmq.Context(1)
        self.worker = self.context.socket(zmq.REQ)

        self.identity = u"Worker-{}".format(worker_id).encode("ascii")
        self.worker.setsockopt(zmq.IDENTITY, self.identity)
        self.worker.connect(BACKEND_ADR)

        self.nA = ACTION_NA
        self.random = np.random.RandomState(1)
        print "I: (%s) worker ready" % self.identity

    def close_connect(self):
        print('in {} close connect'.format(self.identity) )
        self.worker.setsockopt(zmq.LINGER, 0)
        self.worker.close()
        self.context.term()

    # def __del__(self):
    #     print('in del____')

    def policy(self, s):
        ones = np.ones(self.nA)
        p = ones / self.nA
        return p



    def predict(self, state, greedy = True, epsilon = 0.01):
        """ 
            greedy = True for Train; False for only Predict Run
        """
        #Process Image
        # print(' predict, state.shape={}'.format(state.shape))

        pi = self.policy(state)
        if greedy :
            actions = self.random.randint(0,self.nA)
            # if( self.random.rand() < epsilon ) :
            #     actions = self.random.randint(0,self.nA)
            # else :
            #     actions = np.argmax(pi)
        else :
            actions = [ self.random.choice(self.nA, 1, p=p)[0] for p in pi ]
        

        print('pi={}, actions={}'.format(pi,actions))

        return actions


    def train(self,tag_id, state, reward, action ):
        train_dir = 'train/'
        if not os.path.isdir(train_dir):
            os.mkdir(train_dir)

        # state = np.stack(state,axis=1)

        print('train state.shape={}'.format(state.shape) )
        pic_path =  train_dir + tag_id +'+r'+ str(reward) + '_a'+  str(action) + '_0.jpg'
        cv2.imwrite(pic_path, state[:,:,0])

        pic_path =  train_dir + tag_id +'+r'+ str(reward) + '_a'+  str(action) + '_1.jpg'
        cv2.imwrite(pic_path, state[:,:,1])

        pic_path =  train_dir + tag_id +'+r'+ str(reward) + '_a'+  str(action) + '_2.jpg'
        cv2.imwrite(pic_path, state[:,:,2])
        
        pic_path =  train_dir + tag_id +'+r'+ str(reward) + '_a'+  str(action) + '_3.jpg'
        cv2.imwrite(pic_path, state[:,:,3])

        # print('write path =%s' % pic_path)

    def run(self):
        self.worker.send(LRU_READY)

        cycles = 0
        while True:
            # reda data and blocking
            try:
                client_id, _ , msg   = self.worker.recv_multipart()
            except Exception as e :
                print('self.worker.recv_multipart() Exception = {}'.format(e))
                break
                
            if not msg:
                print('E: Worker say  msg is None')
                break
            # unpack msg
            load = loads(msg)

            # print('len(load) = {}'.format(len(load)))
            seq = load[0]
            cmd = load[1]

            print("I: [{}]: Get [{}]'s cmd:({}) , seq:({}) ".\
                    format(self.identity, client_id, cmd, seq) )

            if PREDICT_CMD == cmd:
                state = load[2]
                actions = self.predict(state)
                msg = dumps( (seq, actions) )
                self.worker.send_multipart([client_id, _ , msg ])

                print("I: [{}]: Get [{}]'s cmd:({}) , seq:({}), state.shape: {}, ".\
                    format(self.identity, client_id, cmd, seq, state.shape) )

            elif TRAIN_CMD == cmd:
                state  = load[2]
                reward = load[3]
                action = load[4]

                tag_id = "{}+{}".format(client_id, seq)
                self.train(tag_id, state, reward, action)
                msg = dumps( seq )
                self.worker.send_multipart([client_id, _ , msg ])

                print("I: [{}]: Get [{}]'s cmd:({}) , seq:({}), state.shape: {}, reward: {}, action: {} ".\
                    format(self.identity, client_id, cmd, seq, state.shape, reward, action) )

            