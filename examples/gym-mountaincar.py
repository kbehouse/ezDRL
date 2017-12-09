# Run with NatureDQN
#   python server.py               config/mountaincar_NatureDQN.yaml
#   python examples/gym-mountaincar.py config/mountaincar_NatureDQN.yaml
#   
# Run with DoubleDQN
#   python server.py               config/mountaincar_DoubleDQN.yaml
#   python examples/gym-mountaincar.py config/mountaincar_DoubleDQN.yaml
#   
# Run with DuelingDQN
#   python server.py               config/mountaincar_DuelingDQN.yaml
#   python examples/gym-mountaincar.py config/mountaincar_DuelingDQN.yaml
#   
#   Author:  Shengru, Xiao  <stemsgrpy(at)gmail(dot)com>,
#          

import sys, os
import Queue # matplotlib cannot plot in main thread,
import time
import gym  #OpenAI gym
# append this repo's root path
sys.path.append(os.path.abspath(os.path.dirname(__file__)+'/../'))
from client import Client, EnvSpace
import envs
class MountainCar(EnvSpace):

    def env_init(self):
        self.EP_MAXSTEP = 10000
        self.env = gym.make('MountainCar-v0')
        self.state = self.env.reset()
        self.send_state_get_action(self.state)

    def on_predict_response(self, action):
        next_state, reward, done, _ = self.env.step(action)
        #if self.ep_use_step >= self.EP_MAXSTEP:
        #    done = True

        #if self.ep % 50 == 25:
        #    self.env._render(title = 'Episode: %3d, Step: %3d' % (self.ep,self.ep_use_step+1))

        self.env.render()

        position, velocity = next_state

        # the higher the better
        reward = abs(position - (-0.5))     # r in [0, 1]

        self.send_train_get_action(self.state, action, reward, done, next_state)
        self.state = next_state 
        if done:
            self.state =  self.env.reset()
            self.send_state_get_action(self.state)

if __name__ == '__main__':
    c = Client(MountainCar,'Main-Env')
    c.start()
        