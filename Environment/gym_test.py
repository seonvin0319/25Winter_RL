import gym
import time

env = gym.make('CartPole-v1', render_mode='human', new_step_api=True)

observation = env.reset()[0]
for t in range(100):
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    if done:
        break
    time.sleep(0.1)

env.close()