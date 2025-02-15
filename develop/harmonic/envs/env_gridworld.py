import numpy as np
import random
import gym
from gym import spaces

class gridWorld(gym.Env):
    def __init__(self, grid_size=(5, 5), start=(0, 0), goal=(4, 4), obstacles=None, max_steps=20,
                 randomize=False, possible_starts=None, possible_goals=None, mode="train"):
        super(gridWorld, self).__init__()
        self.grid_size = grid_size
        self.start = start
        self.goal = goal
        self.obstacles = set(obstacles) if obstacles else set()
        self.max_steps = max_steps
        self.current_step = 0
        self.state = self.start
        
        self.observation_space = spaces.Box(low=0, high=1, shape=(grid_size[0] * grid_size[1],), dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        
        self.goal_reward = 10
        self.step_reward = -0.01
        self.obstacle_penalty = -1
        self.timeout_penalty = 0
        self.illegal_penalty = -0.2
        
        self.randomize = randomize
        self.possible_starts = possible_starts
        self.possible_goals = possible_goals
        self.mode = mode

    def get_state_index(self, state):
        return state[0] * self.grid_size[1] + state[1]

    def to_one_hot(self, index):
        state = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        state[index] = 1.0
        return state

    def reset(self):
        if self.randomize:
            if self.possible_starts is not None:
                self.start = random.choice(self.possible_starts)
            if self.possible_goals is not None:
                self.goal = random.choice(self.possible_goals)
        self.state = self.start
        self.current_step = 0
        return self.to_one_hot(self.get_state_index(self.state))

    def step(self, action):
        moves = {
            0: (-1, 0),  #left
            1: (1, 0),   #right
            2: (0, -1),  #down
            3: (0, 1)    #up
        }
        dx, dy = moves[action]
        new_state = (self.state[0] + dx, self.state[1] + dy)

        if not (0 <= new_state[0] < self.grid_size[0] and 0 <= new_state[1] < self.grid_size[1]):
            reward = self.illegal_penalty
            done = False
            info = {}
            new_state = self.state
        elif new_state in self.obstacles:
            reward = self.obstacle_penalty
            done = False
            info = {"info": "Hit Obstacle"}
        elif new_state == self.goal:
            reward = self.goal_reward
            done = True
            info = {"info": "Goal Reached"}
        elif self.current_step >= self.max_steps - 1:
            reward = self.timeout_penalty
            done = True
            info = {"info": "Max Steps Reached"}
        else:
            reward = self.step_reward
            done = False
            info = {"info": "Step Taken"}

        self.state = new_state
        self.current_step += 1
        if self.current_step > self.max_steps:
            done = True
        return self.to_one_hot(self.get_state_index(new_state)), reward, done, info

    def render(self, mode='human'):
        grid = np.zeros(self.grid_size)
        for obs in self.obstacles:
            grid[obs] = -1
        grid[self.start] = 1
        grid[self.goal] = 2
        print(grid)
