import gym
import torch
import time
import os
import imageio
from algorithms.drl.dqn import DQN
from gym.wrappers import RecordVideo

# env = gym.make('CartPole-v1', render_mode='human')
env = gym.make('CartPole-v1', render_mode='rgb_array')
# env = RecordVideo(env, video_folder='videos', episode_trigger=lambda e: True)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

model_path = "results/dqn/model.pth"

agent = DQN(state_dim, action_dim)
agent.load_model(model_path)
agent.eval()

if not os.path.exists("gif"):
    os.makedirs("gif")

for episode in range(5):
    print(f"\nEpisode {episode + 1}")
    observation, _ = env.reset()
    frames = []
    for t in range(800):
        frame = env.render()
        frames.append(frame)

        action = agent.sample_action(observation)
        observation, reward, term, trun, info = env.step(action)
        if term or trun:
            print(f"Finished in {t+1} steps")
            break
        time.sleep(0.1)

    filename = f"gif/episode_{episode + 1}.gif"
    imageio.mimsave(filename, frames, fps = 30)
    print(f"saved {filename} with {len(frames)} frames")
    
env.close()