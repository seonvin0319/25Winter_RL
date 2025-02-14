import numpy as np
from Environment import coin_toss
from Algorithms.Tabular import Tabular

env = coin_toss.Env()
agent = Tabular(env)

# Hyperparameters
gamma = 0.99
threshold_dp = 1e-3
max_iter_dp = 1000
alpha = 0.1
max_episode_num = 100
max_episode_length = 1000
epsilon = 0.1


print('--------------------------------')
print("Prediction Value Function of Uniform Policy:")

uniform_policy = agent.policy_class.uniform_random_policy()
true_v = env.true_v(uniform_policy, gamma = gamma)
print(f'True V: {true_v}')
v_dp = agent.prediction(uniform_policy, "DP", gamma = gamma, threshold = threshold_dp, max_iter = max_iter_dp)
print(f'DP: {v_dp}')
v_onp_mc = agent.prediction(uniform_policy, "on-policy MC", max_episode_num = max_episode_num, max_episode_length = max_episode_length, gamma = gamma)
print(f'On-policy MC: {v_onp_mc}')
v_td = agent.prediction(uniform_policy, "TD", gamma = gamma, alpha = alpha, max_episode_num = max_episode_num, max_episode_length = max_episode_length)
print(f'TD: {v_td}')
print('--------------------------------')

print("Finding Optimal Policy and Value Function:")
v_optimal_policy_iter, optimal_policy_policy_iter = agent.control("policy iteration", gamma = gamma, threshold = threshold_dp, max_iter = max_iter_dp)
print(f'Optimal Policy V Policy Iteration: {v_optimal_policy_iter}')
v_optimal_value_iter, optimal_policy_value_iter = agent.control("value iteration", gamma = gamma, threshold = threshold_dp, max_iter = max_iter_dp)
print(f'Optimal Policy V Value Iteration: {v_optimal_value_iter}')

print('--------------------------------')
print("Finding Optimal Policy and Q-function:")
true_v_optimal = env.true_v(optimal_policy_policy_iter, gamma = gamma)
true_q_optimal = env.true_q(true_v_optimal)
print(f'True Q: {true_q_optimal}')

q_optimal_onp_mc, optimal_policy_onp_mc = agent.control("on-policy MC", max_episode_num = max_episode_num, max_episode_length = max_episode_length, gamma = gamma, epsilon = epsilon)
print(f'Optimal Policy Q on-policy MC: {q_optimal_onp_mc}')
q_optimal_off_mc, optimal_policy_off_mc = agent.control("off-policy MC", max_episode_num = max_episode_num, max_episode_length = max_episode_length, gamma = gamma, epsilon = epsilon)
print(f'Optimal Policy Q off-policy MC: {q_optimal_off_mc}')
q_optimal_sarsa, optimal_policy_sarsa = agent.control("SARSA", gamma = gamma, alpha = alpha, max_episode_num = max_episode_num, max_episode_length = max_episode_length, epsilon = epsilon)
print(f'Optimal Policy Q SARSA: {q_optimal_sarsa}')
q_optimal_q_learning, optimal_policy_q_learning = agent.control("Q-LEARNING", gamma = gamma, alpha = alpha, max_episode_num = max_episode_num, max_episode_length = max_episode_length, epsilon = epsilon)
print(f'Optimal Policy Q Q-LEARNING: {q_optimal_q_learning}')
q_optimal_double_q_learning, optimal_policy_double_q_learning = agent.control("DOUBLE-Q-LEARNING", gamma = gamma, alpha = alpha, max_episode_num = max_episode_num, max_episode_length = max_episode_length, epsilon = epsilon)
print(f'Optimal Policy Q DOUBLE-Q-LEARNING: {q_optimal_double_q_learning}')

print('--------------------------------')
