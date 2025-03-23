import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
import torch

class SnakeEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 10}
    
    def __init__(self, grid_size=(10, 10)):
        super(SnakeEnv, self).__init__()
        self.grid_size = grid_size
        self.action_space = spaces.Discrete(4)  # 0: Up, 1: Down, 2: Left, 3: Right
        self.observation_space = spaces.Box(low=0, high=1, shape=(grid_size[0], grid_size[1], 1), dtype=np.float32)
        
        self.snake = [(5, 5)]
        self.food = (np.random.randint(grid_size[0]), np.random.randint(grid_size[1]))
        self.direction = (0, 1)
        self.done = False
        
        self.window = None
        self.clock = None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.snake = [(5, 5)]
        self.food = (np.random.randint(self.grid_size[0]), np.random.randint(self.grid_size[1]))
        self.direction = (0, 1)
        self.done = False
        return self._get_observation(), {}
    
    def step(self, action):
        if action == 0:
            self.direction = (-1, 0)
        elif action == 1:
            self.direction = (1, 0)
        elif action == 2:
            self.direction = (0, -1)
        elif action == 3:
            self.direction = (0, 1)
        
        head = (self.snake[0][0] + self.direction[0], self.snake[0][1] + self.direction[1])
        
        if (head in self.snake or head[0] < 0 or head[0] >= self.grid_size[0] or head[1] < 0 or head[1] >= self.grid_size[1]):
            self.done = True
            return self._get_observation(), -10, self.done, False, {}
        
        self.snake.insert(0, head)
        
        reward = 0
        if head == self.food:
            reward = 10
            self.food = (np.random.randint(self.grid_size[0]), np.random.randint(self.grid_size[1]))
        else:
            self.snake.pop()
        
        return self._get_observation(), reward, self.done, False, {}
    
    def _get_observation(self):
        obs = np.zeros(self.grid_size, dtype=np.float32)
        for segment in self.snake:
            obs[segment] = 1
        obs[self.food] = 2
        return np.expand_dims(obs, axis=-1)
    
    def render(self):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((300, 300))
            pygame.display.set_caption("Snake Game")
            self.clock = pygame.time.Clock()
        
        self.window.fill((0, 0, 0))
        cell_size = 300 // self.grid_size[0]
        
        for segment in self.snake:
            pygame.draw.rect(self.window, (0, 255, 0), (segment[1] * cell_size, segment[0] * cell_size, cell_size, cell_size))
        pygame.draw.rect(self.window, (255, 0, 0), (self.food[1] * cell_size, self.food[0] * cell_size, cell_size, cell_size))
        
        font = pygame.font.SysFont(None, 24)
        score_text = font.render(f"Score: {len(self.snake) - 1}", True, (255, 255, 255))
        self.window.blit(score_text, (10, 10))
        
        pygame.display.flip()
        self.clock.tick(self.metadata['render_fps'])
    
    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None

# Train the RL model using GPU
def train_model():
    env = make_vec_env(SnakeEnv, n_envs=1)
    model = DQN("MlpPolicy", env, verbose=1, learning_rate=0.001, buffer_size=50000, batch_size=32, gamma=0.99, target_update_interval=500, device='cuda')
    model.learn(total_timesteps=100000)
    model.save("dqn_snake")
    env.close()

# Load and play with trained model
def play_trained_model(model_path="dqn_snake"):
    model = DQN.load(model_path, device='cuda')
    env = SnakeEnv()
    obs, _ = env.reset()
    
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        env.render()
    
    env.close()

if __name__ == "__main__":
    train_model()
    play_trained_model()
