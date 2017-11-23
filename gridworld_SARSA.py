from client import Client
from env.maze_env import Maze
import sys, time

GLOBAL_EP = 0
START_TIME = time.time()

class Gridworld_SARSA:
    """ Init Client """
    def __init__(self, client_id):
        
        self.done = True

        self.env = Maze()
        self.env.reset()
        
        
        self.client = Client(client_id)
        self.client.set_state(self.get_state)
        self.client.set_train(self.train)
        self.client.start()

        self.client.join()

        self.env.mainloop()
        

    def get_state(self):
        # print('in State, self.done={}'.format(self.done))
        if self.done:
            self.done = False
            self.ep_t = 0
            return self.env.reset()
        else:
            return self.state

    def train(self,action):
        
        self.state, self.reward, self.done, _  = self.env.step(action)

        self.ep_t += 1
        # if self.ep_t == MAX_EP_STEP - 1: self.done = True

        self.log_and_show()

        return (self.reward, self.done, self.state)


    def log_and_show(self):
        global GLOBAL_EP, START_TIME
        
        self.ep_r += self.reward
        self.env.render()

        if self.done:
            use_secs = time.time() - START_TIME
            time_str = '%3dh%3dm%3ds' % (use_secs/3600, (use_secs%3600)/60, use_secs % 60 )
            print('%s -> EP:%4d, STEP:%3d, EP_R:%8.2f, t:%s' % (self.client.client_id,  GLOBAL_EP, self.ep_t, self.reward, time_str))
                
            GLOBAL_EP += 1

if __name__ == '__main__':
    Gridworld_SARSA('Client-0') 