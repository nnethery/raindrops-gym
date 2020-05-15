from gym.envs.registration import register

register(
    id='raindrops-v0',
    entry_point='gym_foo.envs:RaindropsEnv',
)