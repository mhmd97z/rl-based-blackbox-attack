import time
import gym
from gym import spaces
import numpy as np
import math
#from gym_bb_attack.envs.utils import *
from utils import *
import math

np.seterr(invalid='raise')
import warnings
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

class Attack_Env(gym.Env):
    """ A DRS environment for OPENAI gym"""

    def __init__(self):



        # Original Image features
        self.orig_image = None
        self.orig_image_enc = None
        self.orig_image_label = None

        # Purturbation features
        self.relative_pos_vector = None
        self.purt_image_label = None

        #Autoencoder
        self.n_enc_dims = 4
        self.encoder = Encoder(encoded_space_dim=self.n_enc_dims, fc2_input_dim=128)
        load_weights(encoder, "./encoder.pt")
        self.decoder = Decoder(encoded_space_dim=self.n_enc_dims, fc2_input_dim=128)
        load_weights(decoder, "./decoder.pt")

        # Dataset of image to train RL agent on
        data_dir = 'dataset'
        self.train_dataset = torchvision.datasets.MNIST(data_dir, train=True, download=True)


        self.action_space = spaces.Box(low=-1, high=1,
                                       shape=(self.N_ENC_DIMS,), dtype=np.float32)


        self.observation_space = spaces.Space(
                                            #The perturbed image's encoding
                                            spaces.Box(low=0, high=1, shape=(self.n_enc_dims,),dtype=np.float32),


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
        self.orig_image = get_random_image(self.train_dataset)
        self.orig_image_enc = encode_image(self.orig_image, self.encoder)
        self.orig_image_label =

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
