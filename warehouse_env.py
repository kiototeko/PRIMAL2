import gym
from gym import spaces
import numpy as np
from enum import Enum


class Action(Enum):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3
    NOOP = 4


class WarehouseEnv(gym.Env):
    def __init__(self, obstacle_map, agent_map):
        super().__init__()
        assert obstacle_map.size == agent_map.size

        self.agent_state = {}
        self.agent_goal = {}
        rows, cols = np.nonzero(agent_map)
        for i, (row, col) in enumerate(zip(rows, cols)):
            self.agent_state[i] = (row, col)
            self.agent_goal[i] = None
            

        self.n_agents = len(self.agent_state)
        self.obstacle_map = obstacle_map
        self.agent_map = agent_map
        self.goal_map = np.zeros_like(agent_map)
        self.action_space = spaces.Discrete(5)

    def _observe(self, agent):

        goal = self.agent_goal[agent]
        state = self.agent_state[agent]

        agent_channel = np.zeros_like(self.agent_map)
        agent_channel[state] = 1
        other_agent_channel = np.zeros_like(self.agent_map)
        for k, v in self.agent_state.items():
            if k != agent:
                other_agent_channel[v] = 1
        goal_channel = np.zeros_like(self.goal_map)
        goal_channel[goal] = 1
        other_goal_channel = np.zeros_like(self.goal_map)
        for k, v in self.agent_goal.items():
            if k != agent:
                other_goal_channel[v] = 1
        return np.stack(
            [agent_channel, other_agent_channel, goal_channel, other_goal_channel],
            axis=0,
        )

    def _occupied(self, row, col):
        if self.obstacle_map[row, col] == 1:
            return True
        for _, v in self.agent_state.items():
            if v == (row, col):
                return True
        return False

    def assign_goal(self, agent, goal):
        self.agent_goal[agent] = goal
    
    def random_without_repetition(self, from_list): #random goal positions without repeting
        R, C = self.agent_map.shape
        obstacles = [(row,col) for row,col in np.transpose(np.nonzero(self.obstacle_map))]
        while True:
            r=(np.random.randint(0,R),np.random.randint(0,C))
            if r not in from_list and r not in obstacles:
                return r

    def step(self, agent, action):
        row, col = self.agent_state[agent]
        R, C = self.agent_map.shape
        if action == Action.RIGHT:
            s_prime = row, min(col + 1, C-1) #Should be C-1 as index goes from 0 to C-1
        elif action == Action.UP:
            s_prime = max(0, row - 1), col
        elif action == Action.LEFT:
            s_prime = row, max(0, col - 1)
        elif action == Action.DOWN:
            s_prime = min(row + 1, R-1), col #Should be R-1 as index goes from 0 to R-1
        elif action == Action.NOOP:
            s_prime = row, col
        else:
            raise ValueError("Invalid action.")
        if not self._occupied(*s_prime):
            self.agent_state[agent] = s_prime
        observation = self._observe(agent)
        reward = 0
        if self.agent_state[agent] == self.agent_goal[agent]:
            reward = 1
            goal = self.random_without_repetition(self.agent_goal.values()) 
            self.assign_goal(agent, goal) #Assign new goal
            done = True #Done should be true when the agent gets to a goal
        else:
            done = False
        return observation, reward, done, {}
    
            

    def action2dir(self, action):
        checking_table = {Action.NOOP: (0, 0), Action.RIGHT: (0, 1), Action.UP: (-1, 0), Action.LEFT: (0, -1), Action.DOWN: (1, 0)}
        return checking_table[action]
    
    def dir2action(self,direction):
        checking_table = {(0, 0): Action.NOOP, (0, 1): Action.RIGHT, (-1, 0): Action.UP, (0, -1): Action.LEFT, (1, 0): Action.DOWN}
        return checking_table[direction]

    def listValidActions(self, agent): #this function is used on PRIMAL 2
        available_actions = []
        R, C = self.agent_map.shape


        for action in Action:
            
            if action == Action.NOOP: #staying in place is always valid
                available_actions.append(action.value)
                continue
            
            direction = self.action2dir(action)
            new_pos = tuple(map(lambda i, j: i + j, direction, self.agent_state[agent]))
            
            if new_pos.count(-1) > 0 or new_pos[0] > R-1 or new_pos[1] > C-1 or self._occupied(*new_pos):
                continue
            available_actions.append(action.value)
            
        return available_actions


    def reset(self):
        self.agent_state = {}
        self.agent_goal = {}
        for i, (row, col) in enumerate(np.transpose(np.nonzero(self.agent_map))):
            self.agent_state[i] = (row, col)
            goal = self.random_without_repetition(self.agent_goal.values()) 
            self.assign_goal(i, goal) #Assign new goal
        #return self._observe() the _observe() function has as an argument an agent

    def render(self, mode="human"):
        pass

    def close(self):
        pass
