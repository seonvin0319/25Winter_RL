from algorithms.drl.dqn import DQN
# 나중에 다른 딥러닝 알고리즘들도 추가될 수 있음
# from algorithms.ddpg import DDPG
# from algorithms.ppo import PPO
# etc...

class DeepRL:
    def __init__(self, env_trainer):
        self.env_trainer = env_trainer
        self.state_dim, self.action_dim = env_trainer.get_env_info()

    def train(self, method, **kwargs):
        """
        Parameters
        ----------
        method : str
            Method to use for training. One of ["DQN", "DDPG", "PPO", etc.]
        **kwargs : dict
            Additional arguments for specific methods
            - DQN: hidden_dim, lr, gamma, epsilon, epsilon_min, epsilon_decay,
                  buffer_size, update_freq, target_hard_update, etc.
            - DDPG: (향후 추가될 파라미터들)
            - PPO: (향후 추가될 파라미터들)
        """
        if method.upper() == "DQN":
            agent = DQN(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                hidden_dim=kwargs.get('hidden_dim', 128),
                lr=kwargs.get('lr', 1e-3),
                gamma=kwargs.get('gamma', 0.99),
                eps=kwargs.get('epsilon', 1.0),
                eps_min=kwargs.get('epsilon_min', 0.01),
                eps_decay=kwargs.get('epsilon_decay', 0.995),
                buffer_capacity=kwargs.get('buffer_size', 10000),
                update_frequency=kwargs.get('update_freq', 100),
                target_net_hard_update=kwargs.get('target_hard_update', False)
            )

            # 학습 실행
            self.env_trainer.train(
                agent=agent,
                max_episode_num=kwargs.get('max_episodes', 10000),
                max_episode_length=kwargs.get('max_steps', 1000),
                batch_size=kwargs.get('batch_size', 64),
                make_csv=kwargs.get('make_csv', True),
                csv_dir=kwargs.get('csv_path', './train_data.csv'),
                save_model=kwargs.get('save_model', True),
                model_dir=kwargs.get('model_path', './model.pth'),
                load_model_path=kwargs.get('load_model', None)
            )

            # 테스트 실행
            self.env_trainer.test(
                agent=agent,
                max_episode_num=kwargs.get('max_episodes', 10000),
                max_episode_length=kwargs.get('max_steps', 1000)
            )

            return agent

        # elif method.upper() == "DDPG":
        #     향후 다른 알고리즘들 추가
        else:
            raise ValueError(f"Invalid method: {method}")