import gym
import time

env = gym.make('CartPole-v1', render_mode='human')

observation = env.reset()[0]
for t in range(100):
    env.render()
    action = env.action_space.sample()
    observation, reward, term, trun, info = env.step(action)
    if term or trun:
        break
    time.sleep(0.1)

env.close()