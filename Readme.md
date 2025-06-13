# elsciRL Application: Maze

This is a recreation of the maze problems defined by Xidon Feng et al. in '[Natural Language Reinforcement Learning](https://arxiv.org/abs/2402.07157)' (2023). 

Code was sourced from [waterhorse1/Natural-langauge-RL](https://github.com/waterhorse1/Natural-language-RL) and transformed to work with the [elsciRL framework](https://elsci.org).


## Configuration

maze_name:
    - 'umaze'
    - 'double_t_maze'
    - 'medium'
    - 'large'
    - 'random'
        - inputs for: size, complexity & density 

reward_signal e.g. [1,0,-0.05]:
    - success
    - timeout
    - action cost

describe_mode:
    - give_position
    - only_walls