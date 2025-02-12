import numpy as np

def policy_eval(env, policy, gamma = 0.99, threshold = 1e-3, max_iter = 100, history = False):
    if history:
        v_history = []
        v_history.append(np.zeros((env.state_space_dim, 1)))
    v = np.zeros((env.state_space_dim, 1)) # S x 1
    iter_num = 0
    v_old = v.copy()
    while True:
        for state in range(env.state_space_dim):
            v_next = 0
            for action in range(env.action_space_dim):
                pi_sa = policy[state, action]
                for next_state in range(env.state_space_dim):
                    p_sas = env.transition_prob[state, action, next_state]
                    v_next += pi_sa * p_sas * (env.reward[state, next_state] + gamma * v[next_state])
            v[state] = v_next
        if history:
            v_history.append(v.copy())
        if np.max(np.abs(v - v_old)) < threshold:
            break
        v_old = v.copy()
        iter_num += 1
        if iter_num > max_iter:
            break
    if history:
        return v, v_history
    return v

def policy_improvement(env, policy, v, gamma = 0.99, threshold = 1e-3, max_iter = 10):
    new_policy = np.zeros_like(policy) # S x A x 1
    iter_num = 0
    while True:
        for state in range(env.state_space_dim):
            pi_s = policy[state].copy()
            bellman_v = np.zeros(env.action_space_dim) # A
            for action in range(env.action_space_dim):
                p_sas = env.transition_prob[state, action]
                for next_state in range(env.state_space_dim):
                    bellman_v[action] += p_sas[next_state] * (env.reward[state, next_state] + gamma * v[next_state])
            argmax_action = np.argmax(bellman_v)
            new_policy[state, argmax_action] = 1
            error = np.max(np.abs(new_policy[state] - pi_s))
            policy[state] = new_policy[state]
        if error < threshold:
            break
        iter_num += 1
        if iter_num > max_iter:
            break
    return policy

def policy_iteration(env, init_policy, gamma = 0.99, threshold = 1e-3, max_iter = 10):
    policy_old = init_policy
    while True:
        v = policy_eval(env, policy_old, gamma = gamma, threshold = threshold, max_iter = max_iter)
        policy = policy_improvement(env, policy_old, v, gamma = gamma, threshold = threshold, max_iter = max_iter)
        if np.max(np.abs(policy - policy_old)) < threshold:
            break
        policy_old = policy
    return policy

if __name__ == '__main__':
    from Environment.coin_toss import Env
    import matplotlib.pyplot as plt

    p_head = 0.5 # Unbiased coin
    r_head = 1
    r_tail = 0
    gamma = 0.9
    env = Env(p_head = p_head, r_head = r_head, r_tail = r_tail)
    policy = np.ones((env.state_space_dim, env.action_space_dim)) / env.action_space_dim

    v, v_history = policy_eval(env, policy, gamma = gamma, history = True)
    v_theo1, v_theo2 = env.true_v(policy, gamma = gamma)
    v_history_array = np.array(v_history)
    v1, v2 = v_history_array[:, 0], v_history_array[:, 1]

    optimal_policy = policy_iteration(env, policy, gamma = gamma, threshold = 1e-3, max_iter = 10)
    _, v_history_optimal = policy_eval(env, optimal_policy, gamma = gamma, history = True)
    v_history_optimal_array = np.array(v_history_optimal)
    v1_optimal, v2_optimal = v_history_optimal_array[:, 0], v_history_optimal_array[:, 1]
    v_theo1_optimal, v_theo2_optimal = env.true_v(optimal_policy, gamma = gamma)

    plt.figure()
    plt.plot(v1, label = '$\hat{v}(H)$', color = 'r')
    plt.axhline(v_theo1, color = 'r', label = '$v(H)$', linestyle = '--')
    plt.plot(v2, label = '$\hat{v}(T)$', color = 'b')
    plt.axhline(v_theo2, color = 'b', label = '$v(T)$', linestyle = '--')
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.title('Policy Iteration for Uniform Policy')
    plt.legend()

    plt.figure()
    plt.plot(v1_optimal, label = '$\hat{v}(H)$', color = 'r')
    plt.axhline(v_theo1_optimal, color = 'r', label = '$v(H)$', linestyle = '--')
    plt.plot(v2_optimal, label = '$\hat{v}(T)$', color = 'b')
    plt.axhline(v_theo2_optimal, color = 'b', label = '$v(T)$', linestyle = '--')
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.title('Policy Iteration for Optimal Policy')
    plt.legend()
    plt.show()