import gym
from gym import error, spaces, utils
import numpy as np
import subprocess, time


class RaindropsGym(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, jar_location, pixels):
        

    def step(self, action):
        if action == 0: # NO OP
            self.game.moveBucket(0,0)
        elif action <= self.pixels:
            self.game.moveBucket(0, action * 10)
        else:
            self.game.moveBucket(1, action * 10)

        reward = self.game.reward()
        observation = self.game.screen()
        game_over = self.game.state()

        observation = np.frombuffer(observation, dtype=np.int8)
        observation.shape = (480, 800, 4)

        return observation, reward, game_over, {}

    def reset(self):
        self.game.reset()
        return self.game.screen()

    def render(self, mode='human'):
        img = self.game.screen()
        arr = np.frombuffer(img, dtype=np.int8)
        arr.shape = (480, 800, 4)
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.SimpleImageViewer()
        self.viewer.imshow(arr)
        return self.viewer.isopen

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

class RaindropsWrapper(gym.ObservationWrapper):
    """This wrapper converts a Box observation into a single integer.
    """
    def __init__(self, env, n_bins=10, low=None, high=None):
        super().__init__(env)
        assert isinstance(env.observation_space, spaces.Box)

        low = self.observation_space.low if low is None else low
        high = self.observation_space.high if high is None else high

        self.n_bins = n_bins
        self.val_bins = [np.linspace(l, h, n_bins + 1) for l, h in
                         zip(low.flatten(), high.flatten())]
        self.observation_space = spaces.Discrete(n_bins ** low.flatten().shape[0])

    def _convert_to_one_number(self, digits):
        return sum([d * ((self.n_bins + 1) ** i) for i, d in enumerate(digits)])

    def observation(self, observation):
        digits = [np.digitize([x], bins)[0]
                  for x, bins in zip(observation.flatten(), self.val_bins)]
        return self._convert_to_one_number(digits)