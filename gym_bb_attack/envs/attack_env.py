import time
import gym
from gym import spaces
import numpy as np
import math
from gym_bb_attack.envs.utils import *
#from utils import *
import math

np.seterr(invalid='raise')
import warnings
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

class Attack_Env(gym.Env):
    """ A DRS environment for OPENAI gym"""

    def __init__(self):

        #Hyperparameters
        self.epsilon = 5
        self.alpha = 0.08
        self.beta = 0.7
        self.max_steps = 100
        self.n_step = 0
        self.image_dims = 28 * 28
        self.n_classes = 10

        # Original Image features
        self.orig_image = None # Original image
        self.orig_image_enc = None # Original image encoding
        self.orig_image_label = None # Original image label
        self.target_image_label = None # Target image label

        # Perturbed image features
        self.current_image = None # Perturbed image
        self.current_image_enc = None # Enc of perturbed image
        self.relative_pos_vector = None # Cumulative perturbations matrix
        self.current_image_label = None # Perturbed image label
        self.lp_dist = None

        #Autoencoder and classifier models and weights
        self.n_enc_dims = 15
        self.encoder = Encoder(encoded_space_dim=self.n_enc_dims)
        load_weights(self.encoder, "./encoder.pt")
        self.decoder = Decoder(encoded_space_dim=self.n_enc_dims)
        load_weights(self.decoder, "./decoder.pt")
        self.classifier = Classifier()
        load_weights(self.classifier, "./classifier.pth")
        self.encoder.eval()
        self.decoder.eval()

        # Dataset of image to train RL agent on
        data_dir = 'dataset'
        self.train_dataset = torchvision.datasets.MNIST(data_dir, train=True, download=True)

        self.observation_space = spaces.Tuple((
                                            # The original image's encoding
                                            spaces.Box(low=0, high=1, shape=(self.n_enc_dims,), dtype=np.float32),

                                            #Relative position vector
                                            spaces.Box(low=-1, high=1, shape=(self.n_enc_dims,),dtype=np.float32),

                                            #Current l_P  distance
                                            spaces.Box(low=0, high=100, shape=(1,),dtype=np.float32),

                                            #Image original label
                                            spaces.Discrete(10),))

        # self.action_space = spaces.Box(low=-1, high=1,
        #                                shape=(self.n_enc_dims,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([3 for _ in range(15)])

    def step(self, action):

        action = (np.array(action) - 1) * self.alpha
        reward = 0
        done = False

        info = {"done": 0,
                "success": 0,
                "lp_dist": self.lp_dist,
                "n_step": 0}

        # Getting the updated image
        updated_image_enc = np.clip(self.current_image_enc + action, 0, 1)
        updated_image = np.clip(decode_image(updated_image_enc, self.decoder), 0, 1)

        # # Image out of bounds
        # if np.max(updated_image) > 1 or np.min(updated_image) < 0: # If image is out of range
        #     reward = -100
        #     print("Image out of bounds")
        # else: # If image is in bounds then Perturb the image

        self.relative_pos_vector += np.reshape(updated_image - self.current_image, (self.image_dims))
        self.current_image = updated_image
        self.current_image_enc = updated_image_enc
        self.current_image_label = np.argmax(classify_image(updated_image, self.classifier))

        self.lp_dist = self.calculate_lp_dist(self.relative_pos_vector)
        info["lp_dist"] = self.lp_dist
        info["n_step"] = self.n_step

        if self.lp_dist > self.epsilon:
            reward = -1 - 10 * (self.lp_dist/10)
        else:
            reward = -1 + get_2ndclass_prob(self.current_image, self.orig_image_label, self.classifier)
        #     self.beta * np.log(
        #         np.clip(get_2ndclass_prob(self.current_image, self.orig_image_label, self.classifier), 1e-5, 1)) +\
        #     (1 - self.beta) * np.log(
        #         np.clip(get_class_prob(self.current_image, self.orig_image_label, self.classifier), 1e-5, 1))

        #if self.current_image_label == self.target_image_label or self.n_step >= self.max_steps:
        if (self.current_image_label != self.orig_image_label and self.lp_dist <= self.epsilon) \
                or self.n_step >= self.max_steps:
            if self.current_image_label != self.orig_image_label and self.lp_dist <= self.epsilon:
                info["success"] = 1
            info["done"] = 1
            #print(info)
            done = True
            #print(info)
            #print("Done")
        self.n_step += 1


        return self.get_observation_space(), reward, done, info

    def reset(self):
        self.orig_image = get_random_image(self.train_dataset)
        self.orig_image_label = np.argmax(classify_image(self.orig_image, self.classifier))
        self.orig_image_enc = encode_image(self.orig_image, self.encoder)

        self.current_image = decode_image(self.orig_image_enc, self.decoder)
        self.current_image_label = np.argmax(classify_image(self.current_image, self.classifier))
        self.current_image_enc = self.orig_image_enc

        while self.current_image_label != self.orig_image_label:
            self.orig_image = get_random_image(self.train_dataset)
            self.orig_image_label = np.argmax(classify_image(self.orig_image, self.classifier))
            self.orig_image_enc = encode_image(self.orig_image, self.encoder)

            self.current_image = decode_image(self.orig_image_enc, self.decoder)
            self.current_image_label = np.argmax(classify_image(self.current_image, self.classifier))
            self.current_image_enc = self.orig_image_enc

        # self.target_image_label = np.random.randint(0, 10)
        # while self.target_image_label == self.orig_image_label:
        #     self.target_image_label = np.random.randint(0, 10)

        self.relative_pos_vector = np.reshape(self.current_image - self.orig_image, (self.image_dims))
        self.lp_dist = self.calculate_lp_dist(self.relative_pos_vector)
        self.n_step = 0

        return self.get_observation_space()

    def get_observation_space(self):
        observation_space = [self.orig_image_enc,
                             self.current_image_enc - self.orig_image_enc,
                             np.array([self.calculate_lp_dist(self.relative_pos_vector)]),
                             self.orig_image_label]
                             #, self.current_image_label]
                             #get_one_hot(self.orig_image_label, self.n_classes),
                             #get_one_hot(self.current_image_label, self.n_classes)]

        return observation_space

    def calculate_lp_dist(self, vector):
        lp_dist = np.sqrt(np.square(vector).sum())
        # lp_dist = np.linalg.norm(vector, 2)
        #lp_dist = np.max(np.abs(vector))
        return lp_dist


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
    #run_base_case(env, 100)
    print(env.observation_space.sample())