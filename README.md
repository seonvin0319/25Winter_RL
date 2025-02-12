# IISL 2025 겨울 강화학습 스터디

## 스터디 일정
- **2025-02-12**  
  - Ch.3 (Finite MDP), Ch.4 (DP)
- **2025-02-19**  
  - Ch.5 (MC), Ch.6 (TD)
- **2025-02-26**  
  - Ch.13 (Policy Gradient)
- **2025-03-05**  
  - DQN

---

## 파일 구조 (업데이트 예정)
```
├── Algorithms
│   ├── Dynamic_Programming.py
├── Environment
│   └── coin_toss.py
├── LICENSE
├── README.md
└── main.py
```

---

## Environment

### `coin_toss.py`

#### Parameters
- `init_state`: 초기 상태
- `p_head`: 앞면(Head)이 나올 확률
- `r_head`: 앞면(Head)일 때의 보상
- `r_tail`: 뒷면(Tail)일 때의 보상

#### State Space
- `{0, 1}` (0: Head, 1: Tail)

#### Action Space
- `{0, 1}` (0: Flip, 1: Pass)

#### Methods
- `step(action)`:  
  - 주어진 `action`을 수행하고 `(next_state, reward)`를 반환

- `reset(state)`:  
  - 환경을 특정 `state`로 초기화  
  - 만약 `state == None`이면 `init_state`로 리셋

- `true_v(policy)`:  
  - 주어진 `policy`에 대한 이론적 가치 함수 계산

---

## Algorithms

### Dynamic_Programming

#### Funtions
- `policy_eval(env, policy, history = False)`:
  - `Dynamic Programming`에 따라 주어진 `env`에서의 `policy evaluation`을 수행하고 `V(s)`를 반환
  - 만약 `history == True`이면 `V(s)`와 함께 iteration의 `history`를 반환

- `policy_improvement(env, policy, v)`:
  - `Dynamic Programming`에 따라 주어진 `env, policy, v(s)`에 대해 policy improvement를 수행하고 새로운 `policy`를 반환

- `policy_iteration(env, init_policy)`:
  -`init_policy`로 부터의 `policy_iteration`을 통해 `optimal policy`를 반환