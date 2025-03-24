import gym
import csv

class GymTrainer:
    def __init__(self, env_name, render_mode=None):
        self.env_name = env_name
        self.render_mode = render_mode
        self.env = None

    def get_env_info(self):
        """환경의 state와 action 차원을 반환"""
        env = gym.make(self.env_name)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        print('-'*30)
        print('Environment Info')
        print(f'Environment: {self.env_name}')
        print(f'State Dimension: {state_dim}')
        print(f'Action Dimension: {action_dim}')
        print('-'*30)
        env.close()
        return state_dim, action_dim

    def _init_env(self):
        """환경 초기화"""
        self.env = gym.make(self.env_name, render_mode=self.render_mode, new_step_api=True)
        return self.env.reset()

    def _save_to_csv(self, csv_dir, data):
        """CSV 파일에 데이터 저장"""
        with open(csv_dir, 'a') as f:
            writer = csv.writer(f)
            if isinstance(data, list):
                writer.writerow(data)
            else:
                writer.writerow([data])

    def train(self, agent, max_episode_num, max_episode_length, batch_size,
              make_csv=True, csv_dir=None, save_model=True, model_dir=None,
              load_model_path=None):
        """학습 수행"""
        observation = self._init_env()

        if load_model_path is not None:
            agent.load_model(load_model_path)

        if make_csv:
            csv_dir = './train_data.csv' if csv_dir is None else csv_dir + '.csv'
            with open(csv_dir, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(agent._loss_info())

        for episode in range(max_episode_num):
            episode_return = 0
            steps = 1
            observation = self._init_env()
            for t in range(max_episode_length):
                action = agent.sample_action(observation)
                next_observation, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                if agent.replay_buffer is not None:
                    agent.replay_buffer.push(observation, action, reward, next_observation, done)

                train_loss = agent.update(batch_size)
                observation = next_observation

                if make_csv and train_loss is not None:
                    self._save_to_csv(csv_dir, train_loss)

                episode_return += reward
                steps += 1

                if done:
                    break

            if save_model:
                model_dir = './model.pth' if model_dir is None else model_dir + '.pth'
                agent.save_model(model_dir)

            print(f'Episode {episode} Return: {episode_return} Steps: {steps}')

        self.env.close()

    def test(self, agent, max_episode_num, max_episode_length):
        """테스트 수행"""
        observation = self._init_env()
        agent.eval()

        for episode in range(max_episode_num):
            episode_return = 0
            steps = 1

            for t in range(max_episode_length):
                action = agent.sample_action(observation)
                next_observation, reward, done, info = self.env.step(action)
                observation = next_observation
                episode_return += reward
                steps += 1

                if done:
                    break

            print(f'Episode {episode} Return: {episode_return} Steps: {steps}')

        self.env.close()