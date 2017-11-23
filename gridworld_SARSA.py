from client import Client
import gym
import envs
import sys, time

GLOBAL_EP = 0
START_TIME = time.time()

print("HIHIHI")

class Gridworld_SARSA:
    """ Init Client """
    def __init__(self, client_id):
        
        self.done = True

        self.env = gym.make('gridworld-v0')
        self.env.set_visual(True)
        self.env.reset()
        
        self.client = Client(client_id)
        self.client.set_state(self.get_state)
        self.client.set_train(self.train)
        self.client.start()

        self.env.after(100, self.show)

    def get_state(self):
        # print('in State, self.done={}'.format(self.done))
        if self.done:
            self.done = False
            self.ep_use_step = 0
            self.next_state =  self.env.reset()
            self.state = self.next_state
        else:
            self.state = self.next_state
        
        return self.state

    def train(self,action):
        
        self.next_state, self.reward, self.done, _  = self.env.step(action)

        self.ep_use_step += 1
        self.log_and_show()

        return (self.reward, self.done, self.next_state)


    def log_and_show(self):
        global GLOBAL_EP, START_TIME

        if GLOBAL_EP % 10 == 0:
            self.show()

        if self.done:
            use_secs = time.time() - START_TIME
            time_str = '%3dh%3dm%3ds' % (use_secs/3600, (use_secs%3600)/60, use_secs % 60 )
            print('%s -> EP:%4d, STEP:%3d, r: %4.2f, t:%s' % (self.client.client_id,  GLOBAL_EP,  self.ep_use_step, self.reward, time_str))
                
            GLOBAL_EP += 1


    def show():
        
        self.env.render()

if __name__ == '__main__':
    Gridworld_SARSA('Client-0') 

        