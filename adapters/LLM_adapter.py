from elsciRL.adapters.LLM_state_generators.text_ollama import OllamaAdapter
import numpy as np


class Adapter:
    def __init__(self, setup_info):
        # Initialize language encoder
        self.LLM_adapter = OllamaAdapter(
            model_name=setup_info.get('model_name', 'llama3.2'),
            base_prompt=setup_info.get('system_prompt', 'You are navigating a maze with x,y positions and proximity to the walls.'),
            context_length=2000,
            action_history_length=setup_info.get('action_history_length', 5),
            encoder=setup_info.get('encoder', 'MiniLM_L6v2')
        )
        
        # Store setup info
        self.describe_mode = setup_info.get("describe_mode", "give_position")
        
        # Build observation mapping functions
        self.delta_descriptions = {
            "to your right": (0, 1),
            "to your left": (0, -1),
            "above you": (-1, 0),
            "below you": (1, 0)
        }

        self.maze_name = setup_info.get("maze_name", "umaze")
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
            width = setup_info.get("maze_width", 81)
            height = setup_info.get("maze_height", 51)
            complexity = setup_info.get("maze_complexity", 0.75)
            density = setup_info.get("maze_density", 0.75)
            
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

        self.goal = self.valid_goals[0]

    def describe_objects(self, object: str, relations: list):
        if len(relations) == 0:
            return f"There are no {object}s near you."
        if len(relations) == 1:
            return f"There is a {object} {relations[0]}."
        return f"There are {object}s {', '.join(relations)}."

    def describe_observation(self, maze, position, goal_position, initial_position=None, move_history=None):
        goal_description = f"The goal is at position {' '.join(str(goal_position[0]))}, {' '.join(str(goal_position[1]))}."
        position = position.split("_")
        position = [int(i) for i in position]

        walls = []
        for k, (dy, dx) in self.delta_descriptions.items():
            if maze[position[0]+dy, position[1]+dx] == 1:
                walls.append(k)
        
        wall_description = self.describe_objects("wall", walls)
        
        if self.describe_mode == "give_position":
            curr_position_description = f"Your current position is at position {' '.join(str(position[0]))}, {' '.join(str(position[1]))}."
            return f"{goal_description} {curr_position_description} {wall_description}\n"
        elif self.describe_mode == "only_walls":
            return f"{wall_description}\n"
        else:
            return f"{goal_description} {wall_description}\n"

    def adapter(self, state, legal_moves=[], episode_action_history=[], encode=True, indexed=False):
        # Get maze and goal from engine
        maze = self.maze
        goal = self.goal
        
        # Generate text description
        text_description = self.describe_observation(
            maze=maze,
            position=state,
            goal_position=goal,
            move_history=episode_action_history
        )

        # Use the elsciRL LLM adapter to transform and encode
        state_encoded = self.LLM_adapter.adapter(
            state=state, 
            legal_moves=legal_moves, 
            episode_action_history=episode_action_history, 
            encode=encode, 
            indexed=indexed
        )
        
        return state_encoded
        