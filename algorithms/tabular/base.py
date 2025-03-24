from algorithms.tabular.dynamic_programming import Dynamic_Programming
from algorithms.tabular.monte_carlo import Monte_Carlo
from algorithms.tabular.td import TD
from utils.tabular_utils import TabularPolicy

class TabularAgent:
    """Tabular 방식의 강화학습 알고리즘들을 관리하는 클래스"""

    VALID_PREDICTION_METHODS = {
        "DP": "dynamic_programming.policy_eval",
        "on-policy MC": "monte_carlo.first_visit_mc_prediction",
        "off-policy MC": "monte_carlo.off_policy_mc_prediction",
        "TD": "td.td_prediction"
    }

    VALID_CONTROL_METHODS = {
        "policy iteration": "dynamic_programming.policy_iteration",
        "value iteration": "dynamic_programming.value_iteration",
        "on-policy MC": "monte_carlo.on_policy_first_visit_mc_control",
        "off-policy MC": "monte_carlo.off_policy_mc_control",
        "SARSA": "td.sarsa",
        "Q-LEARNING": "td.q_learning",
        "DOUBLE-Q-LEARNING": "td.double_q_learning"
    }

    def __init__(self, env):
        self.env = env
        self.dynamic_programming = Dynamic_Programming(env)
        self.monte_carlo = Monte_Carlo(env)
        self.td = TD(env)
        self.policy_class = TabularPolicy(env)

        # 메서드 매핑 초기화
        self._init_method_mappings()

    def _init_method_mappings(self):
        """메서드 매핑 딕셔너리 초기화"""
        self.prediction_methods = {
            "DP": self.dynamic_programming.policy_eval,
            "on-policy MC": self.monte_carlo.first_visit_mc_prediction,
            "off-policy MC": self.monte_carlo.off_policy_mc_prediction,
            "TD": self.td.td_prediction
        }

        self.control_methods = {
            "policy iteration": self.dynamic_programming.policy_iteration,
            "value iteration": self.dynamic_programming.value_iteration,
            "on-policy MC": self.monte_carlo.on_policy_first_visit_mc_control,
            "off-policy MC": self.monte_carlo.off_policy_mc_control,
            "SARSA": self.td.sarsa,
            "Q-LEARNING": self.td.q_learning,
            "DOUBLE-Q-LEARNING": self.td.double_q_learning
        }

        # 별칭 추가
        self.control_methods.update({
            "sarsa": self.td.sarsa,
            "q-learning": self.td.q_learning,
            "double-q-learning": self.td.double_q_learning
        })

    def prediction(self, policy, method, **kwargs):
        """
        상태 가치 함수를 예측하는 메서드

        Parameters
        ----------
        policy : numpy.ndarray
            Policy matrix of shape (state_space_dim, action_space_dim)
        method : str
            사용할 예측 방법. 다음 중 하나:
            - "DP": gamma, threshold, max_iter, history
            - "on-policy MC": max_episode_num, max_episode_length, gamma
            - "off-policy MC": max_episode_num, max_episode_length, gamma
            - "TD": gamma, alpha, max_episode_num, max_episode_length
        **kwargs : dict
            각 메서드에 필요한 추가 인자들
        """
        if method not in self.prediction_methods:
            raise ValueError(f"Invalid method: {method}. Valid methods are {list(self.prediction_methods.keys())}")

        return self.prediction_methods[method](policy, **kwargs)

    def control(self, method, **kwargs):
        """
        최적 정책을 탐색

        Parameters
        ----------
        method : str
            사용할 제어 방법
        **kwargs : dict
            알고리즘별 필요 인자
        """
        if method not in self.control_methods:
            raise ValueError(f"Invalid method: {method}. Valid methods are {list(self.control_methods.keys())}")

        return self.control_methods[method](**kwargs)