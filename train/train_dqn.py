import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from tetris_utils import TetrisEnv, preprocess_state

# 하이퍼파라미터
EPISODES = 500
GAMMA = 0.95
LR = 0.001
BATCH_SIZE = 64
MEMORY_SIZE = 10000
TARGET_UPDATE = 10

# DQN 모델 정의
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# 경험 저장소
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# 학습 루프
def train():
    env = TetrisEnv()
    input_dim = env.state_size
    output_dim = env.action_size

    policy_net = DQN(input_dim, output_dim)
    target_net = DQN(input_dim, output_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayBuffer(MEMORY_SIZE)

    for episode in range(EPISODES):
        state = preprocess_state(env.reset())
        total_reward = 0
        done = False

        while not done:
            if np.random.rand() < max(0.1, 1.0 - episode / 200):  # ε-greedy
                action_idx = np.random.randint(output_dim)
            else:
                with torch.no_grad():
                    q_values = policy_net(torch.FloatTensor(state).unsqueeze(0))
                    action_idx = torch.argmax(q_values).item()

            action = env.get_action_from_index(action_idx)
            next_state_raw, reward, done = env.step(action)
            next_state = preprocess_state(next_state_raw)

            memory.push(state, action_idx, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(memory) >= BATCH_SIZE:
                batch = memory.sample(BATCH_SIZE)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.FloatTensor(states)
                actions = torch.LongTensor(actions).unsqueeze(1)
                rewards = torch.FloatTensor(rewards).unsqueeze(1)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones).unsqueeze(1)

                q_values = policy_net(states).gather(1, actions)
                next_q_values = target_net(next_states).max(1)[0].unsqueeze(1)
                expected_q = rewards + (1 - dones) * GAMMA * next_q_values

                loss = nn.MSELoss()(q_values, expected_q)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # 주기적으로 target network 동기화
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        print(f"Episode {episode} - Total Reward: {total_reward}")

    # 모델 저장
    torch.save(policy_net.state_dict(), "model/dqn_tetris_model.pth")
    print("Model saved to model/dqn_tetris_model.pth")

if __name__ == "__main__":
    train()
