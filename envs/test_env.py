from maze_env import Maze
import time



up = 0
down = 1
right = 2
left  =3

env = Maze()

print('range(env.n_actions)',range(env.n_actions))
print('list(range(env.n_actions))',list(range(env.n_actions)))



# 0
o = env.reset()
env.render()
print('o = {}'.format(o))
time.sleep(1)

# 1
observation_, reward, done = env.step(right)
print('observation_ = {}, reward={}, done={}'.format(observation_, reward, done) )
o = env.render()
time.sleep(1)

# 2
observation_, reward, done = env.step(right)
print('observation_ = {}, reward={}, done={}'.format(observation_, reward, done) )
o = env.render()
time.sleep(1)


env.mainloop()




