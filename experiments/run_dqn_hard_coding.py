from algorithms.drl.dqn import DQN
from environment.gym_trainer import GymTrainer

def main():
    # Gym 트레이너 생성
    trainer = GymTrainer(
        env_name='CartPole-v1',
        render_mode=None  # None, 'human', 'rgb_array'
    )

    # 환경 정보 가져오기
    state_dim, action_dim = trainer.get_env_info()

    # DQN 에이전트 생성
    dqn = DQN(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=128,
        device=None,
        lr=1e-3,
        gamma=0.99,
        eps=1.0,
        eps_min=0.01,
        eps_decay=0.995,
        buffer_capacity=10000,
        update_frequency=100,
        target_net_hard_update=False
    )

    # 학습 실행
    trainer.train(
        agent=dqn,
        max_episode_num=10000,
        max_episode_length=1000,
        batch_size=64,
        make_csv=False,
        csv_dir='./train_data.csv',
        save_model=False,
        model_dir='./model.pth',
        load_model_path=None
    )

    # 테스트 실행
    trainer.test(
        agent=dqn,
        max_episode_num=10000,
        max_episode_length=1000
    )

if __name__ == '__main__':
    main()