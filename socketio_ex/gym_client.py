from socketIO_client import SocketIO, BaseNamespace
import cv2
import os
import gym
import numpy as np
import scipy.misc
from utility import *
import json

Game_Name = 'Breakout-v0'
STATE_SHAPE = (84,84)

#--------------Alread have id, connect again -------------#
class ClientSpace(BaseNamespace):
    def on_connect(self):
        print('GymSpace say connect')
        self.env_init()

    def on_disconnect(self):
        print(' GymSpace say disconnect')


    def on_action_response(self, action):
        print('client get action data = {}'.format(action))

        self.next_state, r, d, _ = self.env.step(action)

        self.state = self.next_state
        self.send_state( self.state)
        
    def env_init(self):
        print('ENV Init')
        self.env = gym.make(Game_Name) 
        self.state = self.env.reset()
        self.send_state( self.state)
    
    def send_state(self, state):
        state_process = self.state_preprocess(state)
        # msg = dumps(self.state_process)
        dic ={'state': state_process.tolist()}
        self.emit('get_action',dic)
        # self.emit('get_action',{'state':self.state_process})

    

    def state_preprocess(self, state_im):
        y = 0.2126 * state_im[:, :, 0] + 0.7152 * state_im[:, :, 1] + 0.0722 * state_im[:, :, 2]
        y = y.astype(np.uint8)
        resized = scipy.misc.imresize(y, STATE_SHAPE)

        return resized
        


def connect_with_ns(ns):
    print('defins ns ={}'.format(ns))
    new_client = socketIO.define(ClientSpace, ns)

#--------------Get id -------------#
def on_connect():
    print('client say connect')

def on_reconnect():
    print('client say connect')

def on_disconnect():
    print('disconnect')

def on_session_response(new_id):
    print('Get id = {}'.format(new_id ))
    new_ns = '/' + str(new_id)  + '/predict'
    connect_with_ns(new_ns)

socketIO = SocketIO('127.0.0.1', 5000)
socketIO.on('connect', on_connect)
socketIO.on('disconnect', on_disconnect)
socketIO.on('reconnect', on_reconnect)
# Listen
socketIO.on('session_response', on_session_response)
socketIO.emit('session')
socketIO.wait()


