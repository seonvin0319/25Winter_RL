import numpy as np
from environment.coin_toss import Env
from algorithms.tabular.base import TabularAgent

def main():
    # 환경 및 에이전트 초기화
    env = Env()
    agent = TabularAgent(env)
    
    # 하이퍼파라미터 설정
    gamma = 0.99
    alpha = 0.1
    max_episode_num = 1000
    max_episode_length = 100
    epsilon = 0.1
    
    # 각 알고리즘 실행 및 결과 출력
    print("Running Dynamic Programming...")
    v_dp, policy_dp = agent.control("policy iteration", 
                                  gamma=gamma, threshold=1e-6, max_iter=1000)
    
    print("\nRunning Monte Carlo...")
    q_mc, policy_mc = agent.control("on-policy MC",
                                  gamma=gamma, max_episode_num=max_episode_num,
                                  max_episode_length=max_episode_length)
    
    print("\nRunning TD Learning...")
    q_sarsa, policy_sarsa = agent.control("SARSA",
                                        gamma=gamma, alpha=alpha,
                                        max_episode_num=max_episode_num,
                                        max_episode_length=max_episode_length,
                                        epsilon=epsilon)
    
    # 결과 비교 및 출력
    print("\nResults Comparison:")
    print(f"DP Policy: {policy_dp}")
    print(f"MC Policy: {policy_mc}")
    print(f"SARSA Policy: {policy_sarsa}")

if __name__ == "__main__":
    main() 