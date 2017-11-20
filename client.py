#
#   Send raw picture to server.py
#   Get gary image(84x84) from server (use worker)  
#   Save the gray image(84x84)
#   
#   Author:  Kartik, Chen  <kbehouse(at)gmail(dot)com>,
#          

import os
from collections import deque
from threading import Thread

import cv2
import zmq


import numpy as np
from history_buffer import HistoryBuffer
from utility import *
# from config import *
import config

# Connect Parameter setting 
REQUEST_TIMEOUT = 2500
REQUEST_RETRIES = 3
FRONTEND_ADR = "tcp://127.0.0.1:5555"

SEND_TYPE_PREDICT   = 1
SEND_TYPE_TRAIN     = 2


class Client(Thread):
    """ Init Client """
    def __init__(self, client_id):
        Thread.__init__(self)
        
        self.client_id = client_id


        # Connect Init
        self.context = zmq.Context(1)
        print("I: Connecting to server...")
        self.poll = zmq.Poller()
        self.open_connect_pollin()

        # Init other
        self.sequence = 0
        self.retries_left = REQUEST_RETRIES
        
        self.send_type = SEND_TYPE_PREDICT
        
        self.state = []
        self.next_state = None
        self.frame_count = 0

        # Note: 
        # assume TRAIN_RUN_STEPSS = 5, STATE_SHAPE=(84,84), STATE_FRAMES = 4, ACTION_NUM = 4 
        # state_hist_buf shape = (84, 84, 4)
        # state_buf = (5, 84, 84, 4), reward_buf = (5, ), action_buf = (5, 4)
        if config.STATE_FRAMES >= 2:
            self.state_history = HistoryBuffer(config.STATE_SHAPE, config.STATE_FRAMES) 
        
        self.state_buf = []
        self.reward_buf = []
        self.action_buf = []


    def __del__(self):
        self.context.term()  

    def set_state(self, state_func):
        self.state_fn = state_func

    def set_train(self, train_func):
        self.train_fn = train_func

    """ Picture function & picture init """
    def get_pic_list_from_dir(self,dir_name):
        pic_list = []
        for filename in os.listdir(dir_name):
            suffix = filename.split(".")[-1]
            if  suffix == 'jpg' or suffix == 'jpeg' or suffix=='png':
                pic_list.append( filename)
        return pic_list

    def open_connect_pollin(self):
        self.client = self.context.socket(zmq.REQ)
        # self.client = self.context.socket(zmq.DEALER)
        self.client.setsockopt(zmq.IDENTITY, self.client_id)
        self.client.connect(FRONTEND_ADR)
        self.poll.register(self.client, zmq.POLLIN)

    def close_connect(self):
        # Socket is confused. Close and remove it.
        self.client.setsockopt(zmq.LINGER, 0)
        self.client.close()
        self.poll.unregister(self.client)
        # self.retries_left -= 1


    def check_reply_seq_id(self, reply_seq_id):
        if not reply_seq_id:
            return False
        if int(reply_seq_id) == self.sequence:
            # print("I: Server replied OK (%s)" % reply_seq_id)
            self.retries_left = REQUEST_RETRIES
            return True
        else:
            print("E: Malformed reply from server: %s" % reply_seq_id)
            return False

    def send_predict(self): 
        # prepare sequence & cmd
        seq_str = str(self.sequence).encode('utf-8')
        cmd = PREDICT_CMD.encode('utf-8')
        
        state_raw = self.state_fn()
        self.state= self.state_history.add(state_raw) if config.STATE_FRAMES >= 2 else state_raw
        self.state_buf.append(self.state)
        # print("I: send_predict() say Get state_raw.shape: {}, state.shape: {}, self.state_buf.shape={}".\
        #          format(np.shape(state_raw), self.state.shape, np.shape(self.state_buf)))

        # Send data
        msg = dumps( (seq_str,cmd, self.state) )
        self.client.send( msg,copy=False)

    def send_predict_done(self, recv):
        reply_seq_id, reply_action = loads(recv)
        
        #check sequence
        check_result = self.check_reply_seq_id(reply_seq_id)

        # print('I: send_predict_done() say check_result={}, reply_seq_id={}, reply_action={}'.format(check_result,reply_seq_id,reply_action) )            

        if check_result:
            self.action  = reply_action
            self.action_buf.append(self.action)
            # self.send_type = SEND_TYPE_TRAIN

            # Train
            self.reward, self.done, self.next_state = self.train_fn(self.action)
            self.reward_buf.append(self.reward)
            self.frame_count += 1

            self.state = []

            # ELSE: continue to predict and get data
            if self.frame_count >= config.TRAIN_RUN_STEPS or self.done:
                self.send_type = SEND_TYPE_TRAIN

            return True
        else:
            print('W: send_predict_done() say check_result = False!!')
            return False

    def send_train_data(self):
        # prepare sequence & cmd
        seq_str = str(self.sequence).encode('utf-8')
        cmd = TRAIN_CMD.encode('utf-8')

        msg = dumps( (seq_str, cmd, self.state_buf, self.action_buf, self.reward_buf, self.next_state,  self.done) )
        #                               (5,7)     ,   (5, 2)                (5,1)           (7,)             value 

        # Send data
        # print("I: send_train_data()  cmd = (%s), seq (%s) " % (cmd,seq_str) )
        self.client.send( msg,copy=False)

    def send_train_data_done(self, recv):
        reply_seq_id = loads(recv)
        # print('I: send_train_data_done() say reply_seq_id={}'.format(reply_seq_id) )            
        #check sequence
        check_result = self.check_reply_seq_id(reply_seq_id)
        if check_result:
            self.send_type = SEND_TYPE_PREDICT
            
            self.frame_count = 0
            self.reward_buf = []
            self.state_buf  = []
            self.action_buf = []

            return True
        else:
            print('W: send_train_data_done() say check_result = False!!')
            return False

    def retry_connect(self, main_cb, done_cb):
        main_cb()
        # expect_reply = True
        while self.retries_left >= 0:
            socks = dict(self.poll.poll(REQUEST_TIMEOUT))
            if socks.get(self.client) == zmq.POLLIN:
                try:
                    recv = self.client.recv()
                except Exception as e:
                    self.close_connect()
                    print('Client::run() recv = self.client.recv() Exception = {}'.format(e))
                    # expect_reply = False
                    return False
                    break
                # reply_seq_id, reply_im = loads(recv)

                result = done_cb(recv)
                if result:
                    self.retries_left = REQUEST_RETRIES
                    # expect_reply = False
                    return True
                else:
                    print('W: retry_connect Get FAIL Result' )              
                    # return False
            else:
                
                self.close_connect()
                self.retries_left -= 1
                if self.retries_left <= 0:
                    print("E: Server seems to be offline, abandoning")
                    return False
                    break

                print("W: No response from server, retrying...{}...".format(self.retries_left))
                # retry open connect and send again
                self.open_connect_pollin()
                main_cb()

    def run(self):
        retry_result = True
        while retry_result:
            if self.send_type == SEND_TYPE_PREDICT:
                self.sequence += 1
                retry_result = self.retry_connect(self.send_predict, self.send_predict_done)
            else:
                retry_result = self.retry_connect(self.send_train_data, self.send_train_data_done)
                