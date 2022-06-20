import time
import gym
from gym import spaces
import numpy as np
import math
from gym_bb_attack.envs.utils import *
# from service_requests import *
import math

np.seterr(invalid='raise')
import warnings
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

class Attack_Env(gym.Env):
    """ A DRS environment for OPENAI gym"""

    def __init__(self):

        self.encoder = None
        self.decoder = None
        self.n_enc_dims = 4


        self.action_space = spaces.Box(low=-1, high=1,
                                       shape=(self.N_ENC_DIMS,), dtype=np.float32)


        self.observation_space = spaces.Space(
                                            #The image encoding
                                            spaces.Box(low=0, high=1, shape=(self.n_enc_dims,),dtype=np.float32),

                                            #Current position vector wrt. original image
                                            spaces.Box(low=0, high=1, shape=(self.n_enc_dims,), dtype=np.float32),

                                            #Current l_inf distance
                                            spaces.Box(low=0, high=1, shape=(self.n_enc_dims,),dtype=np.float32),

                                            #Image original label
                                            spaces.Discrete(10),

                                            #Image current label
                                            spaces.Discrete(10),)


    def step(self, action):
        reward, done, info = None, None, None
        return self.get_observation_space(), reward, done, info

    def reset(self):
        return self.get_observation_space()

    def get_observation_space(self):
        observation_space = None
        return observation_space



"""
A manaul base case to test the environmnet 
"""
def run_base_case(env, n_episodes):
    return 0

if __name__ == "__main__":
    env = Attack_Env()
    run_base_case(env, 10)
