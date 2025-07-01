import numpy as np
import matplotlib.pyplot as plt
import random
import copy
import time
import os
import sys
import json
import logging

class Engine:
    def __init__(self, local_setup_info:dict):
        # Store setup info
        self.maze_name = local_setup_info.get("maze_name", "umaze")
        self.max_steps = local_setup_info.get("action_limit", 100)
        self.reward_signal = local_setup_info.get("reward_signal", [1, 0, -0.05])
        
        # Initialize maze
        if self.maze_name == "umaze":
            self.maze = np.array([
                [1, 1, 1, 1, 1],
                [1, 0, 0, 0, 1], 
                [1, 0, 1, 0, 1], 
                [1, 0, 1, 0, 1],
                [1, 1, 1, 1, 1],
            ], dtype=np.uint8)
            self.valid_goals = np.array([[3, 3]])
            self.start_position = (3, 1)
        elif self.maze_name == "double_t_maze":
            self.maze = np.array([
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
                [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1], 
                [1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1], 
                [1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1], 
                [1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1], 
                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], 
                [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1], 
                [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1], 
                [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1], 
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            ], dtype=np.uint8)
            self.valid_goals = np.array([[8, 6]])
            self.start_position = (1, 1)
        elif self.maze_name == "medium":
            self.maze = np.array([
                [1, 1, 1, 1, 1, 1, 1, 1], 
                [1, 0, 0, 1, 0, 0, 0, 1],
                [1, 0, 0, 1, 0, 1, 0, 1],
                [1, 1, 0, 0, 0, 0, 1, 1], 
                [1, 1, 1, 0, 1, 0, 0, 1], 
                [1, 0, 0, 0, 0, 1, 0, 1], 
                [1, 0, 0, 1, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
            ], dtype=np.uint8)
            self.valid_goals = np.array([[6, 6]])
            self.start_position = (1, 1)
        elif self.maze_name == "large":
            self.maze = np.array([
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 0, 1, 0, 1],
                [1, 0, 1, 0, 1, 0, 1, 0, 1],
                [1, 0, 0, 0, 1, 0, 0, 0, 1],
                [1, 0, 1, 1, 1, 0, 1, 1, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 1, 1, 0, 1, 1, 1, 0, 1],
                [1, 0, 0, 0, 1, 0, 0, 0, 1],
                [1, 0, 1, 0, 1, 1, 1, 1, 1],
                [1, 0, 1, 0, 1, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 1, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1]
            ], dtype=np.uint8)
            self.valid_goals = np.array([[10, 7]])
            self.start_position = (1, 1)
        elif self.maze_name == "random":
            # Generate random maze
            width = local_setup_info.get("maze_width", 81)
            height = local_setup_info.get("maze_height", 51)
            complexity = local_setup_info.get("maze_complexity", 0.75)
            density = local_setup_info.get("maze_density", 0.75)
            
            # Only odd shapes
            shape = ((height // 2) * 2 + 1, (width // 2) * 2 + 1)
            # Adjust complexity and density relative to maze size
            complexity = int(complexity * (5 * (shape[0] + shape[1])))
            density = int(density * ((shape[0] // 2) * (shape[1] // 2)))
            
            # Build actual maze
            self.maze = np.zeros(shape, dtype=np.uint8)
            # Fill borders
            self.maze[0, :] = self.maze[-1, :] = 1
            self.maze[:, 0] = self.maze[:, -1] = 1
            
            # Make aisles
            for i in range(density):
                x, y = np.random.randint(0, shape[1]//2 + 1) * 2, np.random.randint(0, shape[0]//2 + 1) * 2
                self.maze[y, x] = 1
                for j in range(complexity):
                    neighbours = []
                    if x > 1:             neighbours.append((y, x - 2))
                    if x < shape[1] - 2:  neighbours.append((y, x + 2))
                    if y > 1:             neighbours.append((y - 2, x))
                    if y < shape[0] - 2:  neighbours.append((y + 2, x))
                    if len(neighbours):
                        y_,x_ = neighbours[np.random.randint(0, len(neighbours))]
                        if self.maze[y_, x_] == 0:
                            self.maze[y_, x_] = 1
                            self.maze[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = 1
                            x, y = x_, y_
            
            # Set start and goal positions
            positions = np.argwhere(self.maze == 0).tolist()
            self.start_position = tuple(positions[0])
            self.valid_goals = np.array([positions[-1]])

        else:
            raise ValueError(f"Unknown maze name: {self.maze_name}")

        # Define actions
        self.actions = {
            0: (0, -1), # left
            1: (0, 1),  # right
            2: (-1, 0), # down
            3: (1, 0), # up
            4: (-1,-1), # down+left
            5: (1,-1), # up+left
            6: (-1,1), # down+right
            7: (1,1) # up+right
        }

        # Initialize state
        self.position = None
        self.goal = None
        self.num_steps = 0
        self.move_history = []

    def reset(self, start_obs = None):
        # Reset position and state
        start_positions = {
            'umaze': (3, 1),
            'double_t_maze': (1, 1),
            'medium': (1, 1),
            'large': (1, 1),
        }

        self.num_steps = 0
        self.move_history = []
        
        # Set goal position
        self.goal = self.valid_goals[0].tolist()
        
        # Set start position
        if start_obs is not None:
            self.position = start_obs.split("_")
            self.position = [int(i) for i in self.position]
        elif self.maze_name == 'random':
            # For random maze, start at a random position
            positions = np.argwhere(self.maze == 0).tolist()
            positions.remove(self.goal)
            self.position = random.choice(positions)
        else:
            # Fixed start positions for predefined mazes
            self.position = list(start_positions.get(self.maze_name, self.start_position))
        
        output_position = f"{self.position[0]}_{self.position[1]}"
        return output_position

    def step(self, state, action):
        state = state.split("_")
        state = [int(i) for i in state]
        # Update position based on action
        if action in self.actions:
            new_pos = (state[0] + self.actions[action][0], 
                      state[1] + self.actions[action][1])
            
            # Check if move is valid
            if (0 <= new_pos[0] < self.maze.shape[0] and 
                0 <= new_pos[1] < self.maze.shape[1] and
                self.maze[new_pos[0], new_pos[1]] == 0):
                self.position = list(new_pos)
        
        # Update history
        self.move_history.append(action)
        self.num_steps += 1
        
        # Check termination
        terminated = False
        if (self.position[0] == self.goal[0] and 
            self.position[1] == self.goal[1]):
            reward = self.reward_signal[0]  # Success reward
            terminated = True
        elif self.num_steps >= self.max_steps:
            reward = self.reward_signal[1]  # Timeout reward
            terminated = True
        elif action not in self.actions:
            reward = self.reward_signal[2]  # Invalid action penalty
        else:
            reward = self.reward_signal[2]  # Step penalty
            
        info = {"goal":self.goal}
        output_position = f"{self.position[0]}_{self.position[1]}"
        return output_position, reward, terminated, info

    def legal_move_generator(self, state = None):
        if state is None:
            state = self.position
        else:
            state = state.split("_")
            state = [int(i) for i in state]
        
        legal_moves = []
        for action, (dy, dx) in self.actions.items():
            new_pos = (state[0] + dy, state[1] + dx)
            if (0 <= new_pos[0] < self.maze.shape[0] and 
                0 <= new_pos[1] < self.maze.shape[1] and
                self.maze[new_pos[0], new_pos[1]] == 0):
                legal_moves.append(action)
                
        return legal_moves

    def render(self, state = None):
        if state is None:
            state = self.position
        else:
            state = state.split("_")
                    
        fig, ax = plt.subplots()
        ax.imshow(self.maze, cmap='binary')
        
        # Plot goal
        ax.plot(self.goal[1], self.goal[0], 'g*', markersize=15)
        
        # Plot current position
        ax.plot(int(state[1]), int(state[0]), 'ro', markersize=10)
        
        plt.grid(True)
        return fig

    def close(self):
        plt.close('all')