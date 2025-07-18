# AI를 이용한 물 온도 맞추기

## 시작하기에 앞서
맨 처음에 릴레이 프로젝트의 취지를 제대로 이해하지 못해 퀘스트를 너무 어렵게 드린 것 같습니다 😥 혼동을 피하기 위해 몇 가지 가이드를 드릴게요

**DQN 코드를 완성시키지 않으셔도 됩니다! 이 프로젝트의 목적은 샤워기 온도 조절이라는 현실의 문제를 어떻게 AI를 통해 해결할 수 있는지가 목적이지 실제로 DQN을 구현하라는 것이 아닙니다**
그냥 강화학습이라는 이론이 있고 이런 이론을 통해서 현실의 문제를 해결할 수도 있겠구나 취지로 봐주시면 좋을 것 같습니다 아래 체크리스트를 참고해 주시면 좋을 것 같습니다
그리고 만약 코드를 작성하시고 싶으시다면 AI를 통해서 작성해 보시는 것을 추천합니다(DQN을 직접 튜닝하는 것은 관련 지식이 없으면 매우 어렵기 때문에 권장하지 않습니다)

시도 정도는 해보시되 너무 어렵거나 잘 작동하지 않으면 할 수 있는데까지만 해봐도 될 것 같습니다 그리고 이번 기회에 코드 작성 및 분석에 AI를 적극적으로 활용해 보면 좋을 것 같습니다

- [ ] Python, PyTorch, Gym 환경 구축하고 AI가 짠 초안 코드 실행해 보기

저도 아래 코드를 실행해본 것은 아니어서 코드가 제대로 작동할지 장담은 못하겠지만 한번 시도해 보시면 좋을 것 같습니다

- [ ] 만약 제대로 작동하지 않는다면 작동할 수준으로 코드 수정해보기
**모델의 정확도가 높지 않더라도 학습이 이루어진다면 성공으로 간주합니다!**

이 이후부터는 선택에 따라 진행하시면 됩니다

- [ ] (추가 미션) DQN 환경 시각화해보기

코드가 어느 정도 돌아간다면 시각화를 통해 샤워기 시뮬레이션이 원하는대로 동작하는지 확인해 봅니다

- [ ] (추가 미션) DQN 모델 튜닝해서 정확도 올리기
AI를 통해서 어느 정도 튜닝을 시도해 볼 수는 있을 것 같습니다. 예를 들어 코드를 전달받고 모델의 결과를 feedback으로 넣는 식으로 하면 어느 정도 튜닝은 될 것 같습니다. 아니면 Claude Code나 Cursor 등의 AI 코딩 툴을 사용해서 에이전트에게 위임하는 것도 가능은 할 것 같습니다
만약 진행하기 어렵다면 진행하지 않으셔도 됩니다

## 배경 목적

AI를 이용해서 샤워기의 온도를 사용자가 가장 만족할 수 있는 물 온도로 맞추는 것을 목적으로 합니다

## 달성 기준

Python을 통해 샤워기 환경을 성공적으로 구현한 경우 달성한 것으로 간주합니다

## 이것이 필요한 이유?

샤워를 하는 시간에서 약 10분 이상이 물 온도 맞추는데 소모됩니다

- 대부분의 수도꼭지는 아날로그 식이며 수도꼭지가 돌아간 각도와 타이밍(보일러가 얼마나 데워졌는지 / 온수를 얼마나 보유하고 있는지)에 따라 최적의 물 온도까지 도달하기 위한 시간이 천차만별입니다
- 특히 아침에 출근할 때와 같이 빠르게 씻고 나가야 하는데 물 온도 맞추느라 허송세월하면 정작 대충 씻고 나가야 할 수도 있습니다!
    - 물이 너무 뜨거울 경우: 두피와 머리카락에 좋지 않다고 알려져 있고(검증 필요) 화상의 위험이 있습니다
    - 물이 너무 차가울 경우: 너무 차가운 물은 심장에 무리를 줍니다
    - 공통: 사람에 따라 다르겠지만 적정하지 못한 수온은 원활하게 씻는 데에 있어서 방해가 됩니다
- 생각보다 물 온도를 맞추는 것은 쉽지 않습니다
    - 보일러가 데워져 있지 않거나 온수가 부족할 경우 물이 따뜻해지지 않습니다
    - 수도꼭지를 돌린 각도에 맞는 온도까지 물이 도달하는데 시간이 오래 걸립니다
    - 예를 들어, 수도꼭지를 40도 온도에 맞게 돌린다고 바로 40도가 되는 것이 아니고 현재 온도 → 40도까지 도달하는데 시간이 소요됩니다. 이 과정에서 어 물이 안데워지네 하고 수도꼭지를 더 돌릴 경우 40도를 넘는 온도까지 올라갈 위험이 있습니다. → Lazy Evaluation..?
    - 수도꼭지를 얼마만큼 돌려야 하는지 그리고 정해진 만큼 수도꼭지를 돌렸을 때 실제로 그 온도에 정확히 도달하는지에 대해서는 명확히 알려진 바가 없습니다 그렇지만 우리는 크게 개의치 않는데 계속 시도하다 보면 어차피 최적값에 도달함을 이미 경험적으로 알고 있기 때문입니다
        - 일종의 휴리스틱 알고리즘이라고 생각합니다
- 당연한 말이지만 물 온도가 적절하지 않으면 아예 씻지 못할 수도 있습니다

## 부스트캠프에 있어서 어떤 도움이 되나요?

샤워를 빠르게 잘 마치면 컨디션 증진, 학습 시간 확보, 기분 좋게 하루를 시작할 수 있어서 능률 향상 등의 장점이 있습니다

## 기술 검토

이 퀘스트를 수행하는데 있어서 크게 두 가지 방법이 있습니다

1. 실제로 AI를 통해 물 온도 맞추기 시스템을 구상하기
2. Python을 통해서 물 온도 맞추기 시뮬레이션을 구현해 보기

### 실제로 구현하는 경우

<img width="710" height="196" alt="image" src="https://github.com/user-attachments/assets/1371177d-5e3f-4d3f-bd84-ec5db5967e9c" />


처음에 구상할 때는 크게 생각하지 못했는데, 안전상의 이슈가 있을 수 있습니다.

이전에 Arduino를 통해 브레드 보드에서 실습을 진행해 보았을 때 +, - 극을 거꾸로 연결해 LED가 타버린 적이 있었습니다. 전기를 다루는 일은 위험하므로 실제로 하기에는 무리가 있을 것 같습니다.

또한 소프트웨어적으로 구현할 경우에는 버그가 발생하거나 invalid한 값이 나왔을 때(e.g. temperature = 99) 문제가 발생하지 않지만 이를 통해 실제 물 온도를 제어할 경우 소프트웨어젹 결험이 큰 부상으로 이어질 수 있다는 생각이 들었습니다.

이를 방지하기 위해서 Gemini에서는 초기 테스트할 경우에 냉수로만 테스트하는 것을 강력히 권장한다고 하였는데, 이 원리는 실제 프로그램 개발에도 사용될 수 있어서 잘 알아두면 좋을 것 같습니다.

**참고**

<img width="713" height="376" alt="image" src="https://github.com/user-attachments/assets/9a82045e-4a75-4ae3-b6bb-aff848e48899" />



그리고 이를 실제로 구현하려면 Arduino 등의 장비가 필요해 보이는데 릴레이 프로젝트 취지에는 맞지 않는 것 같아서 현실의 구조를 모방한 샌드박스 환경에서 퀘스트를 진행해볼까 합니다.

### Python을 통해서 시뮬레이션을 구현해볼 경우

소프트웨어만으로 이 문제를 해결하는 것은 훨씬 간단합니다. 우선 하드웨어나 안전과 같은 문제에 대해서 생각할 필요가 없어지고 AI로 어떤 일을 달성하는 것은 Python과 PyTorch, TensorFlow 등을 통해 쉽게 구현할 수 있기 때문입니다.

또한 이 프로젝트를 고도화할 것이 아니라면 시중에 나와 있는 모델로도 충분히 목표를 달성 가능할 것으로 보여서 Python으로 구현하는 방안을 한번 생각해 보았습니다. 이는 강화학습을 통해 구현할 수 있을 것으로 보입니다.

### 강화 학습 설명 간단하게

강화 학습이란 AI가 취할 수 있는 행동 중 여러 행동을 시도해 보고 시도한 행동에 따른 보상(Reward)을 이용해서 학습하는 훈련 방식입니다. 계속해서 시도하다 보면 AI는 더 나은 보상을 얻기 위한 방식으로 모델 훈련을 진행하게 되며 이는 인간이 학습하는 방식과도 유사합니다. 대표적으로는 자율주행, 게임에 주로 사용되는 이론입니다.

### Exploration vs Exploitation

강화학습에서 사용되는 이론 중 하나입니다. Exploration은 탐색을, Exploitation은 경험을 의미합니다. 쉽게 비유하자면 점심 메뉴를 고를 때 Exploration은 처음 가보는 식당을, Exploitation은 자주 가는 맛집을 가보는 것을 의미합니다.

<aside>
💡

**물 온도 맞추기에 있어서의 딜레마?**

물 온도 맞추기라는 주제는 다른 주제들과 다르게 Exploration을 할 필요가 없다고 생각합니다. 왜냐하면 이 과정은 아예 가보지 않은 식당을 가보는 것과 같이 기존의 패턴과는 다른 아예 새로운 것을 시도해 봐야 하는데 샤워기의 물 온도를 맞출 때 이걸 시도하다가는 너무 낮은 온도나 너무 높은 온도가 될 수 있기 때문이죠. 따라서 이 이론을 사용해야 하는지는 조금 더 고민이 필요해 보입니다.

</aside>

## 구현

위에서 말했던 것과 같이 Python을 이용해서 어떻게 구현해야 하는지를 간략하게 정리해 보았습니다. Gymnasium을 이용해서 시뮬레이션 환경을 구축해 봅니다.

[Gymnasium Documentation](https://gymnasium.farama.org/)

### 환경 세팅

강화학습을 진행할 때는 AI 에이전트에 있어서 상태(State), 행동(Action), 보상(Reward)이 필요합니다

- 상태: 강화학습 환경에서 현재 상황을 파악할 수 있는 여러 지표들을 의미하며 여기서는 물 온도, 수도꼭지가 돌아간 정도가 필요합니다. Q. 보일러의 현재 상태, 보유한 온수량, 경과한 시간 등도 필요할까요?
- 행동: 에이전트가 취할 수 있는 행동을 의미합니다. 여기서는 수도꼭지를 돌리는 일에 해당합니다. 정확히는 다음과 같습니다.
    - 온도를 올리는 경우: 수도꼭지를 좌측으로 돌립니다. 이 때 돌리는 각도를 지정할 수 있습니다.
    - 온도를 내리는 경우: 수도꼭지를 우측으로 돌립니다. 마찬가지로 돌리는 각도를 지정할 수 있습니다.
    - 가만히 있기: 아무것도 안하고 가만히 있어도 됩니다. 이 경우에도 시간은 흘러가고 보일러나 온수의 상태가 변할 수 있으므로 괜찮습니다.
- 보상: 에이전트가 받는 보상입니다. 간략하게 점수로 나타내며 사용자가 원하는 온도에 가까워질 경우 +, 멀어질 경우는 -를 적용합니다.
    - 특별한 case로 원하는 온도에 도달했지만 너무 오래 걸리는 경우(== 샤워가 너무 늦게 끝난 경우)는 -를 적용합니다.

### 예시 코드

위 요구사항에 맞춰서 Python 코드를 구현할 수 있는지 검증하기 위해서 Gemini 2.5 Pro를 이용해서 코드 초안을 한번 작성해 보았습니다. 여기서는 DQN 모델을 사용했습니다.

<aside>
💡

**Gemini 프롬프트**

파이썬과 Gym을 이용해서 구현하는 코드에 대해서 다음 요구사항을 적용해서 한번 간략하게 초안을 작성해봐

- 샤워기의 수도꼭지는 좌우로만 돌릴 수 있으며 좌로 돌릴 경우 온도 상승, 반대의 경우 온도가 하강함
- 보일러가 있다고 가정하며 보일러가 데워지는데 시간이 걸리고 초기 온도로부터 보일러가 데워지면 온도가 상승함. 이 때 시간이 소요됨.
- 수도꼭지를 돌렸다고 바로 온도에 반영되는 것이 아니라 수도꼭지를 돌려서 온도 상승 -> 보일러 가열 -> 가열된 온수가 나오게 되면 최종적으로 온도가 상승했다고 판단
- 반대의 경우는 온도가 내려가게끔 이 때의 보일러 동작은 일반적으로 알려진 방식을 적용해줘
- Result 공식을 내가 말한 요구사항에 맞게 바꿔줘
- 모델은 초기 실험하기 좋은 형태로 구성해줘
- PyTorch를 사용해줘
- 나머지 요구사항은 알려준 것과 동일함
- 주석을 가능한 상세하게 넣어줘
</aside>

<aside>
❗

아래 코드는 검증이 필요합니다

</aside>

```python
import gymnasium as gym
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 1. 환경(Environment) 정의: 샤워기 시뮬레이터
# ----------------------------------------------------
# Gymnasium의 Env를 상속받아 우리만의 샤워기 환경을 만듭니다.
# 보일러 딜레이, 물의 혼합 등 물리 현상을 코드로 시뮬레이션합니다.

class ShowerEnv(gym.Env):
    def __init__(self):
        super(ShowerEnv, self).__init__()

        # --- 환경 파라미터 정의 ---
        self.TARGET_TEMP = 38.0  # 목표 온도 (섭씨)
        self.COLD_WATER_TEMP = 15.0 # 냉수 기본 온도
        self.MAX_BOILER_TEMP = 80.0 # 보일러가 낼 수 있는 최대 온도

        # 보일러 반응 속도 (값이 작을수록 느리게 반응)
        self.BOILER_HEATING_RATE = 0.05

        # 수도꼭지 위치 (0: 완전 냉수, 10: 완전 온수)
        self.MIN_KNOB_POS = 0
        self.MAX_KNOB_POS = 10

        # 행동 공간(Action Space): AI가 할 수 있는 행동 정의
        # 0: 온도를 낮춤 (오른쪽으로 돌림)
        # 1: 가만히 있음
        # 2: 온도를 높임 (왼쪽으로 돌림)
        self.action_space = gym.spaces.Discrete(3)

        # 관찰 공간(Observation Space): AI가 관찰할 수 있는 상태
        # [현재 물 온도, 현재 보일러 온도]
        # AI는 이 두 가지 정보를 보고 다음 행동을 결정합니다.
        low = np.array([0, 0], dtype=np.float32)
        high = np.array([100, 100], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)
        
        # 에피소드 당 최대 스텝 수
        self.max_steps = 200

    def _get_obs(self):
        """현재 관찰 상태를 반환하는 내부 함수"""
        return np.array([self._current_water_temp, self._boiler_temp], dtype=np.float32)

    def reset(self, seed=None, options=None):
        """환경을 초기 상태로 리셋하는 함수"""
        super().reset(seed=seed)

        # 시스템 상태 초기화
        self._current_water_temp = self.COLD_WATER_TEMP
        self._boiler_temp = self.COLD_WATER_TEMP
        
        # 수도꼭지는 중간(5)에서 시작
        self._knob_position = 5.0

        # 현재 스텝 수 초기화
        self.current_step = 0
        
        return self._get_obs(), {}

    def step(self, action):
        """행동을 실행하고 다음 상태, 보상 등을 반환하는 핵심 함수"""
        self.current_step += 1

        # --- 1. 행동(Action) 해석 및 수도꼭지 위치 변경 ---
        if action == 0:  # 온도 낮추기
            self._knob_position -= 0.5
        elif action == 2:  # 온도 높이기
            self._knob_position += 0.5
        
        # 수도꼭지 위치가 범위를 벗어나지 않도록 제한
        self._knob_position = np.clip(self._knob_position, self.MIN_KNOB_POS, self.MAX_KNOB_POS)

        # --- 2. 보일러의 지연된 반응 시뮬레이션 (핵심 로직) ---
        # 수도꼭지 위치에 따라 보일러의 '목표 온도'가 설정됩니다.
        # 보일러는 이 목표 온도를 향해 점진적으로 온도를 변화시킵니다.
        target_boiler_temp = self.COLD_WATER_TEMP + (self.MAX_BOILER_TEMP - self.COLD_WATER_TEMP) * (self._knob_position / self.MAX_KNOB_POS)
        
        # 보일러 실제 온도는 목표 온도를 서서히 따라갑니다. (가열/냉각 지연 효과)
        self._boiler_temp += (target_boiler_temp - self._boiler_temp) * self.BOILER_HEATING_RATE

        # --- 3. 최종 물 온도 계산 (Result 공식) ---
        # 최종 샤워기 물 온도는 '기본 냉수'와 '보일러에서 데워진 물'이 혼합된 결과입니다.
        # 혼합 비율은 현재 수도꼭지 위치에 의해 결정됩니다.
        mixing_ratio = self._knob_position / self.MAX_KNOB_POS
        self._current_water_temp = (1 - mixing_ratio) * self.COLD_WATER_TEMP + mixing_ratio * self._boiler_temp

        # --- 4. 보상(Reward) 설계 ---
        # 목표 온도와의 차이가 작을수록 높은 보상을 받습니다.
        temp_diff = abs(self._current_water_temp - self.TARGET_TEMP)
        
        # 거리에 반비례하는 보상. 가까울수록 1에 가까워지고 멀수록 0에 가까워짐
        reward = 1.0 / (1.0 + temp_diff)
        
        # 목표 온도에 매우 근접하면 큰 추가 보상
        if temp_diff < 0.5:
            reward += 10.0

        # --- 5. 에피소드 종료 조건 확인 ---
        terminated = self.current_step >= self.max_steps
        
        return self._get_obs(), reward, terminated, False, {}

# 2. DQN 에이전트(Agent) 정의
# ----------------------------------------------------
# AI의 두뇌 역할을 하는 신경망(Q-Network)과 학습 로직을 포함합니다.

# state, action, next_state, reward를 저장하기 위한 튜플
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    """경험(Transition)을 저장하고 재사용하기 위한 메모리"""
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class QNetwork(nn.Module):
    """상태(state)를 입력받아 각 행동(action)의 가치(Q-value)를 출력하는 신경망"""
    def __init__(self, n_observations, n_actions):
        super(QNetwork, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class DQNAgent:
    def __init__(self, n_observations, n_actions):
        # --- 하이퍼파라미터 ---
        self.BATCH_SIZE = 128    # 한 번에 학습할 데이터 샘플 수
        self.GAMMA = 0.99      # 미래 보상에 대한 할인율
        self.EPS_START = 0.9   # 초기 탐험(exploration) 확률
        self.EPS_END = 0.05    # 최종 탐험 확률
        self.EPS_DECAY = 1000  # 탐험 확률 감소 속도
        self.TAU = 0.005       # 목표 네트워크(target network) 업데이트 속도
        self.LR = 1e-4         # 학습률

        self.n_actions = n_actions
        self.steps_done = 0

        # 메인 네트워크(policy_net)와 목표 네트워크(target_net) 생성
        self.policy_net = QNetwork(n_observations, n_actions)
        self.target_net = QNetwork(n_observations, n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict()) # 가중치 복사

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.memory = ReplayMemory(10000)

    def select_action(self, state):
        """엡실론-그리디(Epsilon-Greedy) 방식으로 행동 선택"""
        sample = random.random()
        # 엡실론 값은 에피소드가 진행될수록 점차 감소
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        
        if sample > eps_threshold:
            # 학습된 정책에 따라 최적의 행동 선택 (exploitation)
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            # 무작위로 행동 선택 (exploration)
            return torch.tensor([[random.randrange(self.n_actions)]], dtype=torch.long)

    def train_model(self):
        """메모리에서 샘플을 뽑아 모델을 학습시키는 함수"""
        if len(self.memory) < self.BATCH_SIZE:
            return

        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # 1. 현재 상태(state)에서 특정 행동(action)을 했을 때의 Q-value 계산
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # 2. 다음 상태(next_state)의 최대 Q-value 계산
        next_state_values = torch.zeros(self.BATCH_SIZE)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        
        # 3. 기대 Q-value 계산 (벨만 방정식)
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # 4. 손실(Loss) 계산 및 역전파
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100) # Gradient 클리핑
        self.optimizer.step()

# 3. 훈련 루프(Training Loop)
# ----------------------------------------------------
if __name__ == '__main__':
    env = ShowerEnv()

    # 상태 및 행동 공간의 크기 가져오기
    state, info = env.reset()
    n_observations = len(state)
    n_actions = env.action_space.n

    agent = DQNAgent(n_observations, n_actions)

    num_episodes = 500  # 총 훈련 에피소드 수
    episode_rewards = []

    for i_episode in range(num_episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        total_reward = 0

        for t in count():
            action = agent.select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            
            total_reward += reward
            reward = torch.tensor([reward])
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

            # 경험을 메모리에 저장
            agent.memory.push(state, action, next_state, reward)

            state = next_state

            # 모델 학습
            agent.train_model()

            # 목표 네트워크 가중치 업데이트
            target_net_state_dict = agent.target_net.state_dict()
            policy_net_state_dict = agent.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*agent.TAU + target_net_state_dict[key]*(1-agent.TAU)
            agent.target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_rewards.append(total_reward)
                print(f"Episode {i_episode+1}/{num_episodes}, Total Reward: {total_reward:.2f}")
                break

    print('훈련 완료')
    env.close()

    # 결과 시각화
    plt.figure(figsize=(10, 5))
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.plot(episode_rewards)
    plt.show()
```

### DQN 모델을 사용한 이유?

- 샤워기 온도는 연속적인 state를 가지기 때문에 이산적인 state를 처리하는 기존의 Q 테이블 사용 불가
- 리플레이 버퍼 사용
- 목표 네트워크 분리
