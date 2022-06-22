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

        #Hyperparameters
        self.beta = 0.7
        self.max_steps = 100
        self.n_step = 0
        self.image_dims = 28 * 28
        self.n_classes = 10

        # Original Image features
        self.orig_image = None # Original image
        self.orig_image_label = None # Original image label
        self.target_image_label = None # Target image label

        # Perturbed image features
        self.current_image = None # Perturbed image
        self.current_image_enc = None # Enc of perturbed image
        self.relative_pos_vector = None # Cumulative perturbations matrix
        self.current_image_label = None # Perturbed image label

        #Autoencoder and classifier models and weights
        self.n_enc_dims = 4
        self.encoder = Encoder(encoded_space_dim=self.n_enc_dims, fc2_input_dim=128)
        load_weights(self.encoder, "./encoder.pt")
        self.decoder = Decoder(encoded_space_dim=self.n_enc_dims, fc2_input_dim=128)
        load_weights(self.decoder, "./decoder.pt")
        self.classifier = Classifier()
        load_weights(self.classifier, "./classifier.pth")

        # Dataset of image to train RL agent on
        data_dir = 'dataset'
        self.train_dataset = torchvision.datasets.MNIST(data_dir, train=True, download=True)


        self.action_space = spaces.Box(low=-1, high=1,
                                       shape=(self.n_enc_dims,), dtype=np.float32)

        # This is a
        self.observation_space = spaces.Space((
                                            #The perturbed image's encoding
                                            spaces.Box(low=0, high=1, shape=(self.n_enc_dims,),dtype=np.float32),


                                            #Current l_inf  distance
                                            spaces.Box(low=0, high=1, shape=(self.n_enc_dims,),dtype=np.float32),

                                            #Image original label
                                            spaces.Discrete(10),

                                            #Current image label
                                            spaces.Discrete(10),))


    def step(self, action):
        global TARGET_IMAGE, TARGET_IMAGE_ENC
        print("Env Action: ", action)
        reward = 0
        done = False
        info = {}

        # Getting the updated image
        updated_image_enc = self.current_image_enc + action
        updated_image = decode_image(updated_image_enc, self.decoder)

        # Image out of bounds
        if np.max(updated_image) > 1 or np.min(updated_image) < 0: # If image is out of range
            reward = -100
            print("Image out of bounds")
        else: # If image is in bounds then Perturb the image
            self.relative_pos_vector += np.reshape(updated_image - self.current_image, (self.image_dims))
            self.current_image = updated_image
            self.current_image_enc = updated_image_enc
            self.current_image_label = np.argmax(classify_image(updated_image, self.classifier))
            reward = \
                self.beta * np.log(
                    np.clip(get_class_prob(self.current_image, self.target_image_label, self.classifier), 1e-5, 1)) +\
                (1 - self.beta) * np.log(
                    np.clip(get_class_prob(self.current_image, self.orig_image_label, self.classifier), 1e-5, 1))

        if self.current_image_label == self.target_image_label:
            done = True
            print("Done")
        self.n_step += 1


        return self.get_observation_space(), reward, done, info

    def reset(self):
        self.orig_image = get_random_image(self.train_dataset)
        self.orig_image_label = np.argmax(classify_image(self.orig_image, self.classifier))

        self.current_image = self.orig_image
        self.current_image_label = self.orig_image_label
        self.current_image_enc = encode_image(self.current_image, self.encoder)

        self.target_image_label = np.random.randint(0, 10)
        self.relative_pos_vector = np.zeros(shape=(1, self.image_dims))
        self.n_step = 0

        return self.get_observation_space()

    def get_observation_space(self):

        observation_space = [self.current_image_enc,
                             np.max(np.abs(self.current_image - self.orig_image)),
                             get_one_hot(self.orig_image_label, self.n_classes),
                             get_one_hot(self.current_image_label, self.n_classes)]

        return observation_space



"""
A manaul base case to test the environmnet 
"""
def run_base_case(env, n_episodes):

    for episode in range(n_episodes):
        env.reset()
        done = 0

        target_image, target_image_enc = get_class_image(env.train_dataset, env.target_image_label, env.classifier, env.encoder, env.decoder)

        while not done:
            action = (target_image_enc - env.current_image_enc)/2
            observation, reward, done, info = env.step(action)
            print(f"Reward: {reward}, Done: {done}, Info: {info}")
    return 0

if __name__ == "__main__":
    env = Attack_Env()
    run_base_case(env, 100)