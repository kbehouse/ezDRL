# python TD_test.py config/gridworld_SARSA.yaml
import gym
import envs
from DRL.TD import SARSA,QLearning
import sys



def update(env):
    for episode in range(1, 900):
        # initial observation
        observation = env.reset()

        # RL choose action based on observation
        action = RL.choose_action(observation)
        # print('----------------%3d----------------' % episode)
        step = 0
        if episode %10 == 0:
            env.set_visual(True)
        else:
            env.set_visual(False)

        while True:
            action = RL.choose_action(observation)
            
            # RL take action and get next observation and reward
            observation_, reward, done,_ = env.step(action)
            
            step += 1
            # print("({}) reward = {}, done={}".format(step,reward, done))
            RL.train(observation, action, reward, str(observation_), done)

            # swap observation and action
            observation = observation_
            # break while loop when end of this episode
            if done:
                print("episode={}, use steps={}, reward = {}, epsilon={}".\
                    format(episode, step,reward, RL.epsilon))
                if  episode % 50 ==0:
                    RL.show_qtalbe()
                
                sys.stdout.flush()
                break

    # end of game
    print('game over')
    env.destroy()

if __name__ == "__main__":
    env = gym.make('gridworld-v0')
    env.set_visual(False)
    RL = SARSA()
    # RL = QLearning()

    update(env)