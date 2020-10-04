import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class vchakra(gym.Env):
# class chakra(Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,))
        self.observation_space = spaces.Box(low=-1, high=1, shape=(2,))

        self._seed()
        self.viewer = None
        self.state = None

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        #Fill your code here
        # Return the next state and the reward, along with 2 additional quantities : False, {}
        state = self.state
        # print("here state",state)
        nx, ny = action
        # print("nx, ny", nx, ny)
        dist_xy = np.sqrt(nx**2 + ny**2)
        # print("dist_xy",dist_xy)
        if( -0.025 <= dist_xy <= 0.025):
            dx, dy = action 
        else:
            dx, dy = action * (0.025/dist_xy)
        # print("dx, dy",dx, dy)
        x,y = state
        done = (x==0) and (y ==0)
        if not done:
            reward = -np.sqrt((0.5*x*x)+(5*y*y))
        else:
            reward = 0.0
        if (x+dx)<-1 or (x+dx)>1 :
            x=x
        else:
            x += dx
        if (y+dy)<-1 or (y+dy)>1 :
            y=y
        else:
            y += dy
        # print("next x, y", x, y)
        self.state = np.array([x,y])
        return self.state, reward, done, {}

    def _reset(self):
        while True:
            self.state = self.np_random.uniform(low=-1, high=1, size=(2,))
            # Sample states that are far away
            if np.linalg.norm(self.state) > 0.9:
                break
        return np.array(self.state)

    # method for rendering
    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 800
        screen_height = 800

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            agent = rendering.make_circle(
                min(screen_height, screen_width) * 0.03)
            origin = rendering.make_circle(
                min(screen_height, screen_width) * 0.03)
            trans = rendering.Transform(translation=(0, 0))
            agent.add_attr(trans)
            self.trans = trans
            agent.set_color(1, 0, 0)
            origin.set_color(0, 0, 0)
            origin.add_attr(rendering.Transform(
                translation=(screen_width // 2, screen_height // 2)))
            self.viewer.add_geom(agent)
            self.viewer.add_geom(origin)

        # self.trans.set_translation(0, 0)
        self.trans.set_translation(
            (self.state[0] + 1) / 2 * screen_width,
            (self.state[1] + 1) / 2 * screen_height,
        )

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
