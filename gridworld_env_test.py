import gym
# import gym_gridworld
import envs
import time
import getch

# 1: up, 2: down, 3: left, 4: right

# env = gym.make('FrozenLake-v0')
env = gym.make('gridworld-v0')
_ = env.reset()

sleep_time = 1

def show_all(env,r = 0 , d= False, info = None ):
    print('env.get_agent_state()={}'.format(env.get_agent_state()))
    print('env.get_start_state()={}'.format(env.get_start_state()))
    print('env.get_target_state()={}'.format(env.get_target_state()))
    print("r={},d={}, info={}".format( r,d, info))


print('env.action_space={}'.format(env.action_space))


env.render()
show_all(env)
time.sleep(sleep_time)

s, r, d, info = env.step(2)
env.render()
show_all(env,r ,d, info)
time.sleep(sleep_time)

# while True:
s, r, d, info = env.step(4)
env.render()
show_all(env,r ,d, info)
time.sleep(sleep_time)

s, r, d, _ = env.step(4)
env.render()
show_all(env,r ,d, info)
time.sleep(sleep_time)


s, r, d, _ = env.step(0)
env.render()
show_all(env,r ,d, info)
time.sleep(sleep_time)


s, r, d, _ = env.step(4)
env.render()
show_all(env,r ,d, info)
time.sleep(sleep_time)