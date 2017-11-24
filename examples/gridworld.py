# Run with QLearning
#   python server.py config/gridworld_QLearning.yaml
#   python examples/gridworld.py config/gridworld_QLearning.yaml
# Run With SARSA
#   python server.py config/gridworld_SARSA.yaml
#   python examples/gridworld.py config/gridworld_SARSA.yaml

import sys, os
import Queue # matplotlib cannot plot in main thread,
import time
import gym  #OpenAI gym
# append this repo's root path
sys.path.append(os.path.abspath(os.path.dirname(__file__)+'/../'))
from client import Client
import envs

GLOBAL_EP = 0
START_TIME = time.time()

class Gridworld_EX:
    """ Init Client """
    def __init__(self, client_id):
        # env setting
        self.env = gym.make('gridworld-v0')
        self.env.reset()
        # client setting
        self.client = Client(client_id)
        self.client.set_state(self.get_state)
        self.client.set_train(self.train)
        
    def get_state(self):
        if self.done:
            self.done = False
            self.ep_use_step = 0
            self.next_state =  self.env.reset()

        self.state = self.next_state
        
        return self.state

    def train(self,action):
        self.next_state, self.reward, self.done, _  = self.env.step(action)
        self.ep_use_step += 1
        self.log_and_show()
        return (self.reward, self.done, self.next_state)


    def log_and_show(self):
        global GLOBAL_EP, START_TIME

        if GLOBAL_EP % 50 == 0:
            self.callback_queue.put(self.show)
            self.callback_queue.join()

        if self.done:
            use_secs = time.time() - START_TIME
            time_str = '%3dh%3dm%3ds' % (use_secs/3600, (use_secs%3600)/60, use_secs % 60 )
            print('%s -> EP:%4d, STEP:%3d, r: %4.2f, t:%s' % (self.client.client_id,  GLOBAL_EP,  self.ep_use_step, self.reward, time_str))
                
            GLOBAL_EP += 1

    def start(self):
        self.done = True
        self.callback_queue = Queue.Queue()
        self.client.start()

        # because matplotlib cannot plot in main thread, you need to do that
        while True:
            #blocking 
            callback = self.callback_queue.get() #blocks until an item is available
            callback()
            self.callback_queue.task_done()

    def show(self):
        global GLOBAL_EP
        self.env._render(title = 'Episode: %d' % GLOBAL_EP)

if __name__ == '__main__':
    g = Gridworld_EX('Client-0') 
    g.start()
        