from gym_minigrid.wrappers import *


def make_env(env_key, seed=None):
    env = gym.make(env_key)
    env = MyOneHotPartialObsWrapper(env)
    env.seed(seed)

    return env
