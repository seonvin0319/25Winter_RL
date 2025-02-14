import numpy as np
from Environment import coin_toss
from Algorithms.Monte_Calro import on_policy_first_visit_mc_control, off_policy_mc_control

env = coin_toss.Env()
gamma = 0.9
q, policy = on_policy_first_visit_mc_control(env, max_episode_num = 100, max_episode_length = 1000, gamma = gamma, epsilon = 0.1)
print(f'On-policy q: {q}')
print(f'On-policy policy: {policy}')
v_true = env.true_v(policy, gamma = gamma)
q_true = env.true_q(v_true)
print(f'True Q: {q_true}')



q_off, policy_off = off_policy_mc_control(env, max_episode_num = 100, max_episode_length = 1000, gamma = gamma, epsilon = 0.1)
print(f'Off-policy q: {q_off}')
print(f'Off-policy policy: {policy_off}')
v_true_off = env.true_v(policy_off, gamma = gamma)
q_true_off = env.true_q(v_true_off)
print(f'True Q: {q_true_off}')
