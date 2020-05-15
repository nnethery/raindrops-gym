import gym
from gym import error, spaces, utils
import numpy as np
import subprocess, time
from raindrops_python.game import Game


class RaindropsGym(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, show_score = False):
        self.show_score = show_score
        self.game = Game()

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