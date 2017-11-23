"""
First Version:
https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/3_Sarsa_maze/RL_brain.py
"""

import numpy as np
import pandas as pd
from Base import RL
import six
from abc import ABCMeta,abstractmethod
from config import cfg

class TD(RL):
    def __init__(self, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = list(range(cfg['RL']['action_num']))
        m = cfg['RL']['method']
        self.lr      = cfg[m]['LR']
        self.gamma   = cfg[m]['gamma']  # reward_decay
        self.epsilon = cfg[m]['epsilon-greedy']

        print('self.lr={}, self.gamma={}, self.epsilon={}'.\
            format(self.lr, self.gamma, self.epsilon))

        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def check_state_exist(self, state):
        # print('state={}, type = {}'.format(state, type(state)))
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

    def choose_action(self, observation):
        # print("in TD choose_action")
        self.check_state_exist(observation)
        # action selection
        if np.random.rand() < self.epsilon:
            # choose best action
            state_action = self.q_table.ix[observation, :]
            state_action = state_action.reindex(np.random.permutation(state_action.index))     # some actions have same value
            action = state_action.argmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    @abstractmethod
    def train(self, states, actions, rewards, next_state, done):
        pass


# off-policy
# Q-Learning
class QLearning(TD):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(QLearning, self).__init__(actions, learning_rate, reward_decay, e_greedy)

    # def learn(self, s, a, r, s_, done):
    def train(self, s, a, r, s_, done):
        self.check_state_exist(s_)
        q_predict = self.q_table.ix[s, a]
        # if s_ != 'terminal':
        if done:
            q_target = r + self.gamma * self.q_table.ix[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)  # update


# on-policy
class SARSA(TD):

    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.5):
        super(SARSA, self).__init__(actions, learning_rate, reward_decay, e_greedy)
        self.next_action = None
    
    
    def choose_action(self, observation):
        # print("in SARSA choose_action, self.next_action={}".format(self.next_action))
        if self.next_action == None:
            return super(SARSA,self).choose_action(observation)  
        else:
            return self.next_action

    # def learn(self, s, a, r, s_,  done):
    def train(self, s, a, r, s_, done):
        # print('in Learn')
        self.check_state_exist(s_)
        self.next_action = super(SARSA,self).choose_action(s_)  
        q_predict = self.q_table.ix[s, a]
        # if s_ != 'terminal':
        if done:
            # if r > 0 and self.epsilon < 0.99:
            #     self.epsilon = self.epsilon + 0.001 if self.epsilon < 0.99 else self.epsilon
            q_target = r + self.gamma * self.q_table.ix[s_, self.next_action]  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)  # update
