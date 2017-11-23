from envs.maze_env import Maze
from DRL.TD import SARSA
import sys

def update():
    for episode in range(900):
        # initial observation
        observation = env.reset()

        # RL choose action based on observation
        action = RL.choose_action(str(observation))
        # print('----------------%3d----------------' % episode)
        step = 0
        while True:
            action = RL.choose_action(str(observation))
            
            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)
            
            step += 1
            # print("({}) reward = {}, done={}".format(step,reward, done))
            RL.train(str(observation), action, reward, str(observation_), done)

            # swap observation and action
            observation = observation_
            # break while loop when end of this episode
            if done:
                print("episode={}, use steps={}, reward = {}, epsilon={}".\
                    format(episode, step,reward, RL.epsilon))
                
                sys.stdout.flush()
                break

    # end of game
    print('game over')
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    RL = SARSA(actions=list(range(env.n_actions)), e_greedy=0.9)

    # env.after(100, update)
    env.after(10, update)
    env.mainloop()