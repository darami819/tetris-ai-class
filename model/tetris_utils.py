# tetris_utils.py
import numpy as np

class TetrisEnv:
    def __init__(self):
        self.state_size = 100  # 더미 값
        self.action_size = 5   # 더미 값

    def reset(self):
        return np.zeros(self.state_size)

    def step(self, action):
        next_state = np.random.rand(self.state_size)
        reward = np.random.randint(0, 10)
        done = np.random.rand() > 0.95
        return next_state, reward, done

    def render(self):
        print(".", end='')  # 간단한 출력

    def get_possible_actions(self):
        return [{'y': np.random.randint(0, 20)} for _ in range(self.action_size)]

    def get_action_from_index(self, idx):
        return idx  # 간단한 맵핑

def preprocess_state(state):
    return state / np.linalg.norm(state)  # 정규화 예시
