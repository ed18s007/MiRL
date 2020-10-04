import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class gridworld(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.grid = np.zeros([12,12])
        self.grid[2:9,3:9] = -1
        self.grid[3:8,4:8] = -2
        self.grid[4:7,5:7] = -3
        self.grid[8,8], self.grid[7,8]  = 0,0
        self.grid[7,7], self.grid[6,7]  = -1,-1
        self.grid[6,6], self.grid[6,5], self.grid[5,6]   = -2,-3, -2
        # print(self.grid)
        # self.grid[5,0], self.grid[6,0],self.grid[10,0],self.grid[11,0] = 1,1,1,1
        # print(self.grid)
        self.action_space = spaces.Discrete(4) # 0 - LEFT, 1 - RIGHT, 2 - UP, 3 -DOWN
        self.observation_space = spaces.Box(low=0, high=12, shape=(12,12))
        self.east_wind = True
        self.goal_pos = None
        self.seed()
        self.state = None

    def goal(self, goal):
        if (goal=='A'):
            self.grid[0,11] = 10
            self.goal_pos = [0,11]
            # print(self.grid,self.goal_pos)
            return self.goal_pos
        elif (goal=='B'):
            self.grid[2,9] = 10
            self.goal_pos =  [2,9]
            # print(self.grid,self.goal_pos)
            return self.goal_pos
        elif (goal=='C'):
            self.east_wind = False
            self.grid[6,7] = 10
            self.goal_pos =  [6,7]
            # print(self.grid, self.goal_pos)
            return self.goal_pos


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        #Fill your code here
        # Return the next state and the reward, along with 2 additional quantities : False, {}
        state = self.state
        prob_sel_action = [0.1/3,0.1/3,0.1/3,0.1/3]
        prob_sel_action[action] = 0.9
        act_prob = action
        act_prob = int(np.random.choice(range(4), 1, p=prob_sel_action))
        if self.east_wind:
            act_prob = int(np.random.choice([act_prob,1],1,p=[0.5,0.5]))
            # print("in east_wind act_prob", act_prob)
        trans_pos = [[0,-1], [0,1],[-1,0],[1,0]] # 0 - LEFT, 1 - RIGHT, 2 - UP, 3 -DOWN
        # print("initially self.state",self.state)
        dx, dy = trans_pos[act_prob][0], trans_pos[act_prob][1]
        x,y = self.state[0]+dx, self.state[1]+dy
        if (x<0 or x>11 or y<0 or y>11):
            next_state = self.state
        else:
            next_state = [x,y]
            self.state = next_state
        reward = self.grid[next_state[0],next_state[1]]
        # print("initially next_state",next_state)
        done = ((next_state[0] == self.goal_pos[0])  and (next_state[1] == self.goal_pos[1] ) )
        return next_state, reward, done, {}

    def reset(self):
        start_positions = [[5,0], [6,0],[10,0],[11,0]]
        idx = int(np.random.choice(range(4), 1, p=[0.25, 0.25, 0.25, 0.25]))
        self.state = start_positions[idx]
        return self.state

    # method for rendering
    def render(self, mode='human', close=False):
        pass

# gridworld()