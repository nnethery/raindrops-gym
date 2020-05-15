import gym, time, random
import numpy as np
from envs.raindrops_gym import RaindropsGym

env = RaindropsGym()

while True:
    env.step_frame(env.action_space.sample())
    time.sleep(1/60)