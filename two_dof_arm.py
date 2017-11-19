#
#   Send raw picture to server.py
#   Get gary image(84x84) from server (use worker)  
#   Save the gray image(84x84)
#   Modify from ZMQ example (http://zguide.zeromq.org/py:lpclient)
#   
#   Author:  Kartik, Chen  <kbehouse(at)gmail(dot)com>,
#          

import os
from client import Client
from arm_env import ArmEnv
import time
MAX_EP_STEP = 300
GLOBAL_EP = 0


START_TIME = time.time()

class TwoDofArm:
    """ Init Client """
    def __init__(self, client_id):
        
        self.done = True

        self.env = ArmEnv(mode='hard')
        self.env.reset()
        
        self.client = Client(client_id)
        self.client.set_state(self.get_state)
        self.client.set_train(self.train)
        self.client.start()

        self.done = True

        

    def get_state(self):
        # print('in State, self.done={}'.format(self.done))
        if self.done:
            self.done = False
            self.ep_t = 0
            self.ep_r = 0
            return self.env.reset()
        else:
            return self.state

    def train(self,action):
        global GLOBAL_EP, START_TIME
        self.state, reward, self.done, _  = self.env.step(action)
        self.ep_t += 1
        self.ep_r += reward
        if self.ep_t == MAX_EP_STEP - 1: self.done = True

        if self.client.client_id == 'Client-0':
            self.env.render()


        if self.done:
            use_secs = time.time() - START_TIME
            time_str = '%3dh%2dm%2ds' % (use_secs/3600, (use_secs%3600)/60, use_secs % 60 )
            print('%s -> EP:%4d, STEP:%3d, EP_R:%8.2f, t:%s' % (self.client.client_id,  GLOBAL_EP, self.ep_t, self.ep_r, time_str))
                
            
            GLOBAL_EP += 1

        return (reward, self.done, self.state)

if __name__ == '__main__':
    for i in range(4):
        TwoDofArm('Client-%d' % i ) 