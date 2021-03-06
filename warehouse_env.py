import gym
from gym import spaces
import numpy as np
from enum import Enum
from PIL import Image
from matplotlib import cm

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
        self.past_agent_state = {}
        self.agent_goal = {}
        rows, cols = np.nonzero(agent_map)
        for i, (row, col) in enumerate(zip(rows, cols)):
            self.agent_state[i] = (row, col)
            self.agent_goal[i] = None
            self.past_agent_state[i] = None

        self.n_agents = len(self.agent_state)
        self.obstacle_map = obstacle_map
        self.agent_map = agent_map
        self.goal_map = np.zeros_like(agent_map)
        self.action_space = spaces.Discrete(5)
        
        action_space_map = {}
        action_space_map[0] = Action.RIGHT
        action_space_map[1] = Action.UP
        action_space_map[2] = Action.LEFT
        action_space_map[3] = Action.DOWN
        action_space_map[4] = Action.NOOP
        self.action_space_map = action_space_map

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
            [agent_channel, other_agent_channel, goal_channel, other_goal_channel, self.obstacle_map],
            axis=0,
        )

    def _occupied(self, row, col):
        if self.obstacle_map[row, col] != 0:
            return True
        for _, v in self.agent_state.items():
            if v == (row, col):
                return True
        return False

    def assign_goal(self, agent, goal):
        if self._occupied(goal[0], goal[1]):
            raise ValueError("Attempting to assgin goal to occupied location: {}, {}.".format(agent, goal))
        self.agent_goal[agent] = goal

    def step(self, agent, action):
#         action = self.action_space_map[action] if isinstance(action, int) else action
#         print(action, action == Action.NOOP, action == Action.RIGHT, 
#               action == Action.LEFT, action == Action.UP, action == Action.DOWN)
        row, col = self.agent_state[agent]
        R, C = self.agent_map.shape
        R = R-1
        C = C-1
        if action == 0:
            s_prime = row, min(col + 1, C)
        elif action == 1:
            s_prime = max(0, row - 1), col
        elif action == 2:
            s_prime = row, max(0, col - 1)
        elif action == 3:
            s_prime = min(row + 1, R), col
        elif action == 4:
            s_prime = row, col
        else:
            raise ValueError("Invalid action.")
        
        tmp_state = self.agent_state[agent]
        if not self._occupied(*s_prime):
            self.agent_state[agent] = s_prime
            if self.past_agent_state[agent] != None and self.agent_state[agent] == self.past_agent_state[agent]:
                reward = -2.0
            else:
                reward = -0.3
        else:
            reward = -0.5
            
        self.past_agent_state[agent] = tmp_state
        
        observation = self._observe(agent)
        
        if self.agent_state[agent] == self.agent_goal[agent]:
            reward = 20.0
            # Lazy retry, fix me
            new_goal_location_occupied = True
            while_count = 0
            while new_goal_location_occupied:
                while_count += 1
                new_goal_location = self.get_new_goal_location(excluding_location=self.agent_goal[agent])
                if not self._occupied(new_goal_location[0], new_goal_location[1]):
                    new_goal_location_occupied = False
                if while_count > 800:
                    raise ValueError("Probably in an infinite while loop.")
                
            self.assign_goal(agent, new_goal_location)
            done = True
        else:
            done = False
            
        
        return observation, reward, done, {}
    
    def get_new_goal_location(self, excluding_location=None):
        obstacle_map_copy = self.obstacle_map.copy()
        if excluding_location is not None:
            obstacle_map_copy[excluding_location] = 1
        for i in range(len(self.agent_goal)): #for planner, two agents cannot have the same goal
            if self.agent_goal[i] is not None:
                obstacle_map_copy[self.agent_goal[i]] = 1
        empty_locations = np.argwhere((self.agent_map == 0) & (obstacle_map_copy == 0))
        choice_location = np.random.choice(empty_locations.shape[0])
        x, y = empty_locations[choice_location]
        return (x, y)
        
    def reset(self):
        self.agent_state = {}
        self.past_agent_state = {}
        self.agent_goal = {}
        rows, cols = np.nonzero(self.agent_map)
        for i, (row, col) in enumerate(zip(rows, cols)):
            self.agent_state[i] = (row, col)
            self.agent_goal[i] = None
            self.past_agent_state[i] = None
        return self._observe(agent = 0)

    def render(self, mode="human", zoom_size=8, agent_id=0, other_agents_same=False):
        image_array = np.zeros_like(self.agent_map)
        # Map agents, agent_id will mapped to 1
        for k, v in self.agent_state.items():
            if agent_id is not None:
                if agent_id == k:
                    image_array[v] = 1
                else:
                    if not other_agents_same:
                        image_array[v] = k+2
                    else:
                        image_array[v] = 2
            else:
                image_array[v] = k+1
        
        # Color obstacles and "zoom in" by repeating appropiately
        max_agent_id = max(self.agent_goal, key=int)
        image_array = np.where(self.obstacle_map != 0, max_agent_id + 3, image_array)
        image_array_copy = np.repeat(image_array, zoom_size, axis=0)
        image_array_copy2 = np.repeat(image_array_copy, zoom_size, axis=1)
        
        # Set inner goal boxes for each agent
        inner_box_size = int(zoom_size/4)
        outer_box_size = int(zoom_size/3)
        for k, v in self.agent_goal.items():
            x = int((v[0] * zoom_size) + zoom_size/2) 
            y = int((v[1] * zoom_size) + zoom_size/2) 

            image_array_copy2[(x-outer_box_size):(x+outer_box_size), 
                              (y-outer_box_size):(y+outer_box_size)] = max_agent_id + 4
            if agent_id is not None:
                if agent_id == k:
                    image_array_copy2[(x-inner_box_size):(x+inner_box_size), 
                                      (y-inner_box_size):(y+inner_box_size)] = 1
                else:
                    if not other_agents_same:
                        image_array_copy2[(x-inner_box_size):(x+inner_box_size), 
                                          (y-inner_box_size):(y+inner_box_size)] = k+2
                    else:
                        image_array_copy2[(x-inner_box_size):(x+inner_box_size), 
                                          (y-inner_box_size):(y+inner_box_size)] = 2
            else:
                image_array_copy2[(x-inner_box_size):(x+inner_box_size), 
                                  (y-inner_box_size):(y+inner_box_size)] = k+1
        
        scalar_cm = cm.ScalarMappable(cmap="jet_r")
        color_array = np.uint8(scalar_cm.to_rgba(image_array_copy2)*255)

        #color background gray and obstacles black
        color_array[(image_array_copy2 == 0)] = [190,190,190,255]
        color_array[(image_array_copy2 == (max_agent_id + 3))] = [0,0,0,255]
        color_array[(image_array_copy2 == (max_agent_id + 4))] = [255,255,255,255]
        if agent_id is not None:
            color_array[(image_array_copy2 == 1)] = [255,0,0,255]

        im = Image.fromarray(color_array)
        return im
    
    def close(self):
        pass
    
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

class WarehouseEnvRuntimeError(Exception):
    def __init__(self, message, errors):

        # Call the base class constructor with the parameters it needs
        super().__init__(message)

        # Now for your custom code...
        self.errors = errors
        
        raise errors
