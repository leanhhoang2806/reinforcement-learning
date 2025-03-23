import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import collections

# Maze Environment (5x5 grid)
maze_size = 5
exit_position = (4, 4)
actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
num_actions = len(actions)

# Initialize Q-table
Q = np.zeros((maze_size, maze_size, num_actions))

# Hyperparameters
alpha = 0.5  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration probability
episodes = 2000

# Reward function
def get_reward(state):
    if state == exit_position:
        return 10  # Goal reached
    return -1  # Small penalty to encourage shorter paths

# Choose an action using epsilon-greedy policy
def choose_action(state):
    if random.random() < epsilon:
        return random.randint(0, num_actions - 1)  # Random action
    return np.argmax(Q[state])  # Best known action



for episode in range(episodes):
    state = (0,0) # start position

    while state != exit_position:
        action_index = choose_action(state)
        next_state = (state[0] + actions[action_index][0], state[1] + actions[action_index][1])

        if 0 <= next_state[0] < maze_size and 0 <= next_state[1] < maze_size:
            reward = get_reward(next_state)
        
        else:
            next_state = state
            reward = -5
        
        # Q-learning update
        Q[state][action_index] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state][action_index])
        state = next_state

print("Trained Q-values:")
print(Q)

# Visualizing the learned policy
policy = np.zeros((maze_size, maze_size), dtype=str)
arrows = ['→', '↓', '←', '↑']
for i in range(maze_size):
    for j in range(maze_size):
        if (i, j) == exit_position:
            policy[i, j] = '🏁'
        else:
            best_action = np.argmax(Q[i, j])
            policy[i, j] = arrows[best_action]

print("\nLearned Policy:")
for row in policy:
    print(" ".join(row))




# ########## Neural network implementation ################

# Maze Environment (5x5 grid)
maze_size = 5
exit_position = (4, 4)
actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
num_actions = len(actions)

# Hyperparameters
gamma = 0.9  # Discount factor
alpha = 0.001  # Learning rate
epsilon = 0.1  # Exploration rate
episodes = 100
batch_size = 32
memory_size = 5000

# Experience Replay Memory
memory = collections.deque(maxlen=memory_size)

# Reward function
def get_reward(state):
    return 10 if state == exit_position else -1 


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )
    

    def forward(self, x):
        return self.fc(x)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} for training neural network")
model = DQN().to(device)
optimizer = optim.Adam(model.parameters(), lr=alpha)
loss_fn = nn.MSELoss()

def encode_state(state):
    return torch.tensor(state, dtype=torch.float32).to(device)

def choose_action(state):
    if random.random() < epsilon:
        return random.randint(0, num_actions - 1)  # Random action (exploration)
    with torch.no_grad():
        return torch.argmax(model(encode_state(state))).item()  # Best action (exploitation)

for episode in range(episodes):
    state = (0, 0)  # Start at (0,0)

    while state != exit_position:
        action_index = choose_action(state)
        next_state = (state[0] + actions[action_index][0], state[1] + actions[action_index][1])

        # Check for out-of-bounds moves
        if 0 <= next_state[0] < maze_size and 0 <= next_state[1] < maze_size:
            reward = get_reward(next_state)
        else:
            next_state = state  # Stay in place if hitting a wall
            reward = -5  # Penalty for hitting walls

        # Store experience in memory
        memory.append((state, action_index, reward, next_state))

        # Train on a batch
        if len(memory) >= batch_size:
            batch = random.sample(memory, batch_size)
            state_batch = torch.stack([encode_state(s) for s, _, _, _ in batch])
            action_batch = torch.tensor([a for _, a, _, _ in batch], dtype=torch.int64, device=device)
            reward_batch = torch.tensor([r for _, _, r, _ in batch], dtype=torch.float32, device=device)
            next_state_batch = torch.stack([encode_state(ns) for _, _, _, ns in batch])

            q_values = model(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze()
            next_q_values = model(next_state_batch).max(1)[0].detach()
            target_q_values = reward_batch + gamma * next_q_values

            loss = loss_fn(q_values, target_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        state = next_state  # Move to the next state

# Evaluate learned policy
print("\nLearned Q-values:")
with torch.no_grad():
    for i in range(maze_size):
        for j in range(maze_size):
            state = encode_state((i, j))
            best_action = torch.argmax(model(state)).item()
            print(f"State ({i},{j}): Action {best_action}")

policy = np.full((maze_size, maze_size), "⬛", dtype=str)
arrows = ['→', '↓', '←', '↑']
for i in range(maze_size):
    for j in range(maze_size):
        if (i, j) == exit_position:
            policy[i, j] = '🏁'
        else:
            state = encode_state((i, j))
            best_action = torch.argmax(model(state)).item()
            policy[i, j] = arrows[best_action]

print("\nLearned Policy:")
for row in policy:
    print(" ".join(row))