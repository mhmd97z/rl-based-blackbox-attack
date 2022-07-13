import time
import gym
import matplotlib.pyplot as plt
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

    def __init__(self, config):

        #Hyperparameters ################################################################
        self.epsilon = 5 # Max perturbation size
        self.alpha = 0.03 # Step size for perturbation
        self.beta = 0.99 # Weight for the 2nd highest class
        self.max_steps = 100 # Max steps per episode
        self.n_step = 0 # Current step
        self.image_dims = 28 * 28
        self.n_classes = 10
        self.test = False
        self.done = 0 # Episode done flag
        self.success = 0 # Episode success flag
        self.train_clas = None # Class to train on
        if config != None:
            self.test = config["test"] # Testing or training
            self.train_clas = config["train_class"] # Class to train on

        # Original Image features ########################################################
        self.orig_image = None # Original image
        self.orig_image_enc = None # Original image encoding
        self.orig_image_label = None # Original image label
        self.target_image_label = None # Target image label
        self.encoding_clip_range = [-3, 3]
        self.image_clip_range = [0, 1]
        self.rel_pos_vec_clip_range = 2 * np.array(self.encoding_clip_range)

        # Perturbed image features ########################################################
        self.current_image = None # Perturbed image
        self.current_image_enc = None # Enc of perturbed image
        self.relative_pos_vector = None # Cumulative perturbations matrix
        self.current_image_label = None # Perturbed image label
        self.lp_dist = None

        #Autoencoder/PCA and classifier models and weights #################################
        self.n_enc_dims = 64
        self.n_act_dims = 64

        # For autoencoder model
        self.model_type = 'ae' #Use pca or ae
        self.encoder = Encoder(encoded_space_dim=self.n_enc_dims)
        load_weights(self.encoder, "./encoder.pt")
        self.decoder = Decoder(encoded_space_dim=self.n_enc_dims)
        load_weights(self.decoder, "./decoder.pt")
        self.encoder.eval()
        self.decoder.eval()
        self.classifier = Classifier()
        load_weights(self.classifier, "./classifier.pth")


        # For PCA model
        # self.model_type = 'pca' #Use pca or ae
        # self.encoder = load_pca_model('model.pkl')
        # self.decoder = self.encoder

        # Dataset of image to train RL agent on ##############################################
        data_dir = '/home/m4sulaim/CS886/rl-based-blackbox-attack/dataset'
        self.dataset = None
        transform = transforms.Compose([transforms.ToTensor()])
        if self.test:
            self.dataset = torchvision.datasets.MNIST(data_dir, train=False, download=False, transform=transform)
        else:
            self.dataset = torchvision.datasets.MNIST(data_dir, train=True, download=False, transform=transform)

        self.observation_space = spaces.Tuple((
                                            # The original image's encoding
                                            spaces.Box(low=self.encoding_clip_range[0],
                                                       high=self.encoding_clip_range[1],
                                                       shape=(self.n_act_dims,), dtype=np.float32),

                                            #Relative position vector
                                            spaces.Box(low=self.rel_pos_vec_clip_range[0],
                                                       high=self.rel_pos_vec_clip_range[1],
                                                       shape=(self.n_act_dims,),dtype=np.float32),

                                            #Current l_P  distance
                                            spaces.Box(low=0, high=100, shape=(1,),dtype=np.float32),

                                            #Image original label
                                            spaces.Discrete(10),))

        # self.action_space = spaces.Box(low=-1, high=1,
        #                                shape=(self.n_enc_dims,), dtype=np.float32)
        #self.action_space = spaces.MultiDiscrete([3 for _ in range(self.n_act_dims)])
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(self.n_act_dims,), dtype=np.float32)

    def step(self, action):
        #action = self.action_space.sample() #Samples random action from action space
        #action = (np.array(action) - 1) * self.alpha
        action = np.array(action)
        reward = 0
        done = False

        info = {"done": 0,
                "success": 0,
                "lp_dist": self.lp_dist,
                "n_step": 0,
                "lp_success": 0}

        # Getting the updated image
        updated_image_enc = self.current_image_enc
        updated_image_enc[:self.n_act_dims] = np.clip(self.current_image_enc[:self.n_act_dims] + action,
                                    self.encoding_clip_range[0], self.encoding_clip_range[1])
        updated_image = np.clip(decode_image(updated_image_enc, self.decoder, self.model_type),
                                self.image_clip_range[0], self.image_clip_range[1])

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

        #probs = get_probs(self.current_image, self.classifier)
        #reward = -1 * (self.current_image_label == self.orig_image_label)
        #reward = max(0, 15 - self.lp_dist)/15

        #if self.current_image_label == self.target_image_label or self.n_step >= self.max_steps:
        if (self.current_image_label != self.orig_image_label) \
                or self.n_step >= self.max_steps:

            if self.current_image_label != self.orig_image_label:
                reward += (15 - self.lp_dist)/1.5

            if self.current_image_label != self.orig_image_label:
                info["success"], self.success = 1, 1
                if self.lp_dist < self.epsilon:
                    info["lp_success"] = 1
            info["done"], done, self.done = True, True, True

        self.n_step += 1
        if self.test: print(f"test:{self.test}, obs: {self.get_observation_space()}")

        return self.get_observation_space(), reward, done, info

    def reset(self):
        self.done = False
        self.success = False
        self.orig_image = get_random_image(self.dataset, self.train_clas)
        self.orig_image_label = np.argmax(classify_image(self.orig_image, self.classifier))
        self.orig_image_enc = np.clip(encode_image(self.orig_image, self.encoder, self.model_type),
                                      self.encoding_clip_range[0], self.encoding_clip_range[1])

        self.current_image = np.clip(decode_image(self.orig_image_enc, self.decoder, self.model_type),
                                     self.image_clip_range[0], self.image_clip_range[1])
        self.current_image_label = np.argmax(classify_image(self.current_image, self.classifier))
        self.current_image_enc = self.orig_image_enc

        while self.current_image_label != self.orig_image_label:
            self.orig_image = get_random_image(self.dataset, self.train_clas)
            self.orig_image_label = np.argmax(classify_image(self.orig_image, self.classifier))
            self.orig_image_enc = np.clip(encode_image(self.orig_image, self.encoder, self.model_type),
                                          self.encoding_clip_range[0], self.encoding_clip_range[1])

            self.current_image = np.clip(decode_image(self.orig_image_enc, self.decoder, self.model_type),
                                         self.image_clip_range[0], self.image_clip_range[1])
            self.current_image_label = np.argmax(classify_image(self.current_image, self.classifier))
            self.current_image_enc = self.orig_image_enc

        self.relative_pos_vector = np.reshape(self.current_image - self.orig_image, (self.image_dims))
        self.lp_dist = self.calculate_lp_dist(self.relative_pos_vector)
        self.n_step = 0
        if self.test: print(f"test:{self.test}, obs: {self.get_observation_space()}")
        return self.get_observation_space()

    def get_observation_space(self):
        enc_diff = self.current_image_enc - self.orig_image_enc
        observation_space = [self.orig_image_enc[:self.n_act_dims],
                             enc_diff[:self.n_act_dims],
                             #self.current_image_enc[:self.n_act_dims],
                             np.array([self.calculate_lp_dist(self.relative_pos_vector)], dtype=np.float32),
                             self.orig_image_label]
        return observation_space

    def calculate_lp_dist(self, vector):
        lp_dist = np.sqrt(np.square(vector).sum())
        # lp_dist = np.linalg.norm(vector, 2)
        #lp_dist = np.max(np.abs(vector))
        return lp_dist

    def render(self):
        if self.done == True:
            if self.success == 1:
                plt.figure()
                plt.subplot(1, 2, 1)
                plt.imshow(self.orig_image, cmap='gray')
                plt.title(f"{self.orig_image_label}")
                plt.subplot(1, 2, 2)
                plt.imshow(self.current_image, cmap='gray')
                plt.title(f"{self.current_image_label} ({self.lp_dist})")
                plt.savefig("/home/m4sulaim/CS886/images/image_" + str(time.time()) + ".png")
                print("Saved fig")
        return True

"""
A manaul base case to test the environmnet 
"""
def run_base_case(env, n_episodes):

    for episode in range(n_episodes):
        env.reset()
        done = 0

        target_image, target_image_enc = get_class_image(env.dataset, env.target_image_label, env.classifier, env.encoder, env.decoder)

        while not done:
            action = (target_image_enc - env.current_image_enc)/2
            observation, reward, done, info = env.step(action)
            print(f"Reward: {reward}, Done: {done}, Info: {info}")
    return 0

if __name__ == "__main__":
    env = Attack_Env()
    #run_base_case(env, 100)
    print(env.observation_space.sample())
