#
#   Send raw picture to server.py
#   Get gary image(84x84) from server (use worker)  
#   Save the gray image(84x84)
#   Modify from ZMQ example (http://zguide.zeromq.org/py:lpclient)
#   
#   Author:  Kartik, Chen  <kbehouse(at)gmail(dot)com>,
#          

import cv2
import os
from client import Client
import gym

class Atari:
    """ Init Client """
    def __init__(self, game_name):
        self.done = True


        self.env = gym.make(game_name) 
        self.env.reset()
        self.client = Client('Client-1')
        # self.client.set_action_size(self.env.action_space.n)   #ex: the Atari Breackout action size is 4
        self.client.set_state(self.get_state)
        self.client.set_train(self.train)
        self.client.start()

        

    def get_state(self):
        print('in State, self.done={}'.format(self.done))
        if self.done:
            self.done = False
            return self.env.reset()
        else:
            return self.state

    def train(self,action):
        self.state, reward, self.done, _ = self.env.step(action)
        return (reward, self.done)

if __name__ == '__main__':
   Atari('Breakout-v0') 
   