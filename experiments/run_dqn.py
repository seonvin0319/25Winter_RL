import argparse
import yaml
from pathlib import Path
from algorithms.drl.dqn import DQN
from environment.gym_trainer import GymTrainer

def load_config(config_path):
    """YAML 설정 파일 로드"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def parse_args():
    """커맨드 라인 인자 파싱"""
    parser = argparse.ArgumentParser(description='Deep RL Training Arguments')
    parser.add_argument('--config', type=str, default='./setup/dqn_arg.yaml',
                        help='path to config file')

    # 환경 설정
    parser.add_argument('--env-name', type=str, help='override environment name')
    parser.add_argument('--render-mode', type=str, choices=[None, 'human', 'rgb_array'],
                        help='override render mode')

    # 모델 설정
    parser.add_argument('--hidden-dim', type=int, help='override hidden dimension')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'],
                        help='override device')

    # 학습 설정
    parser.add_argument('--lr', type=float, help='override learning rate')
    parser.add_argument('--gamma', type=float, help='override discount factor')
    parser.add_argument('--epsilon', type=float, help='override initial epsilon')
    parser.add_argument('--epsilon-min', type=float, help='override minimum epsilon')
    parser.add_argument('--epsilon-decay', type=float, help='override epsilon decay rate')
    parser.add_argument('--buffer-size', type=int, help='override buffer size')
    parser.add_argument('--update-freq', type=int, help='override update frequency')
    parser.add_argument('--target-hard-update', action='store_true', help='use hard update')
    parser.add_argument('--batch-size', type=int, help='override batch size')
    parser.add_argument('--max-episodes', type=int, help='override max episodes')
    parser.add_argument('--max-steps', type=int, help='override max steps')

    # 저장/로드 설정
    parser.add_argument('--save-model', action='store_true', help='save model')
    parser.add_argument('--model-path', type=str, help='override model path')
    parser.add_argument('--make-csv', action='store_true', help='save csv')
    parser.add_argument('--csv-path', type=str, help='override csv path')
    parser.add_argument('--load-model', type=str, help='override load model path')

    return parser.parse_args()

def update_config(config, args):
    """커맨드 라인 인자로 설정 업데이트"""
    # 환경 설정
    if args.env_name:
        config['env']['name'] = args.env_name
    if args.render_mode:
        config['env']['render_mode'] = args.render_mode

    # 모델 설정
    if args.hidden_dim:
        config['model']['hidden_dim'] = args.hidden_dim
    if args.device:
        config['model']['device'] = args.device

    # 학습 설정
    train_args = ['lr', 'gamma', 'epsilon', 'epsilon_min', 'epsilon_decay',
                 'buffer_size', 'update_freq', 'batch_size', 'max_episodes', 'max_steps']

    for arg in train_args:
        val = getattr(args, arg.replace('-', '_'))
        if val is not None:
            config['train'][arg] = val

    if args.target_hard_update:
        config['train']['target_hard_update'] = True

    # 저장/로드 설정
    if args.save_model:
        config['save']['model'] = True
    if args.model_path:
        config['save']['model_path'] = args.model_path
    if args.make_csv:
        config['save']['make_csv'] = True
    if args.csv_path:
        config['save']['csv_path'] = args.csv_path
    if args.load_model:
        config['save']['load_model'] = args.load_model

    return config

def main():
    args = parse_args()
    config = load_config(args.config)
    config = update_config(config, args)

    # Gym 트레이너 생성
    trainer = GymTrainer(
        env_name=config['env']['name'],
        render_mode=config['env']['render_mode']
    )

    # 환경 정보 가져오기
    state_dim, action_dim = trainer.get_env_info()

    # DQN 에이전트 생성
    dqn = DQN(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=config['model']['hidden_dim'],
        device=config['model']['device'],
        lr=config['train']['lr'],
        gamma=config['train']['gamma'],
        eps=config['train']['epsilon'],
        eps_min=config['train']['epsilon_min'],
        eps_decay=config['train']['epsilon_decay'],
        buffer_capacity=config['train']['buffer_size'],
        update_frequency=config['train']['update_freq'],
        target_net_hard_update=config['train']['target_hard_update']
    )

    # 학습 실행
    trainer.train(
        agent=dqn,
        max_episode_num=config['train']['max_episodes'],
        max_episode_length=config['train']['max_steps'],
        batch_size=config['train']['batch_size'],
        make_csv=config['save']['make_csv'],
        csv_dir=config['save']['csv_path'],
        save_model=config['save']['model'],
        model_dir=config['save']['model_path'],
        load_model_path=config['save']['load_model']
    )

    # 테스트 실행
    trainer.test(
        agent=dqn,
        max_episode_num=config['train']['max_episodes'],
        max_episode_length=config['train']['max_steps']
    )

if __name__ == '__main__':
    main()