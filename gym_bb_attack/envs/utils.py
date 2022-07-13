import matplotlib.pyplot as plt # plotting library
import numpy as np # this module is useful to work with numerical arrays
import pandas as pd
import random
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader,random_split
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import os
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA, KernelPCA
import math
import pickle as pk
#np.random.seed(9000)

class Encoder(nn.Module):

    def __init__(self, encoded_space_dim):
        super().__init__()
        # ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)

        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(True),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Linear(128, encoded_space_dim),
            nn.BatchNorm1d(encoded_space_dim)
        )

    def forward(self, x):
        # x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        x = torch.clamp(x, -3, 3)
        return x


class Decoder(nn.Module):

    def __init__(self, encoded_space_dim):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 512),
            nn.ReLU(True),
            nn.Linear(512, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = x.view((-1, 1, 28, 28))
        return x
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        # 28x28x1 => 26x26x32
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(2, stride=2)

        self.d1 = nn.Linear(7 * 7 * 64, 1024)
        self.d2 = nn.Linear(1024, 625)
        self.d3 = nn.Linear(625, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # flatten
        x = x.flatten(start_dim=1)

        # FC layers
        x = self.d1(x)
        x = F.relu(x)
        x = self.d2(x)
        x = F.relu(x)

        # logits => 10
        logits = self.d3(x)
        out = F.softmax(logits, dim=1)
        return out

def load_weights(model, filename):
    if not os.path.exists(filename):
        filename = '/home/m4sulaim/CS886/rl-based-blackbox-attack/gym_bb_attack/envs/' + filename
    model.load_state_dict(torch.load(filename, map_location=torch.device('cpu')))

def encode_image(image, encoder, model_type='ae'):
    if model_type == 'ae':
        image_tensor = torch.from_numpy(np.array(np.reshape(image, (1, 1, 28, 28)), dtype=np.float32))
        encoding = encoder(image_tensor).detach().numpy()[0]
        return encoding
    elif model_type == 'pca':
        image = np.reshape(image, (-1, 28 * 28))
        encoding = encoder.transform(image)
        #print(np.max(encoding))
        return encoding.squeeze()

def decode_image(encoding, decoder, model_type='ae'):
    if model_type == 'ae':
        encoding = torch.Tensor(encoding).unsqueeze(0)
        decoding = decoder(encoding).detach().numpy()[0][0]
        return decoding
    elif model_type == 'pca':
        decoding = np.reshape(decoder.inverse_transform(encoding), (28, 28))
        return decoding

def get_random_image(train_dataset, clas=None):
    if clas == None:
        rand_index = np.random.randint(len(train_dataset))
        image = np.array(train_dataset[rand_index][0], dtype=np.float32)
        return image.squeeze()
    else:
        rand_index = np.random.randint(len(train_dataset))
        while train_dataset[rand_index][1] != clas:
            rand_index = np.random.randint(len(train_dataset))
        image = np.array(train_dataset[rand_index][0], dtype=np.float32)
        return image.squeeze()

def get_class_image(train_dataset, clas, classifier, encoder, decoder):
    rand_index = np.random.randint(len(train_dataset))
    image = np.array(train_dataset[rand_index][0])
    image_enc = encode_image(image, encoder)
    image_dec = decode_image(image_enc, decoder)
    classification = classify_image(image_dec, classifier)

    while np.argmax(classification) != clas:
        rand_index = np.random.randint(len(train_dataset))
        image = np.array(train_dataset[rand_index][0])
        image_enc = encode_image(image, encoder)
        image_dec = decode_image(image_enc, decoder)
        classification = classify_image(image_dec, classifier)
    return image, image_enc

def get_random_encoded_image(train_dataset, encoder):
    return encode_image(get_random_image(train_dataset), encoder)

def classify_image(image, classifier):
    image = torch.Tensor(image).unsqueeze(0).unsqueeze(0)
    return classifier(image).detach().numpy()[0]

def get_probs(image, classifier):
    image = torch.Tensor(image).unsqueeze(0).unsqueeze(0)
    return classifier(image).detach().numpy()[0]

"""
Returns the highest prob of class that is not the given class
"""
def get_2ndclass_prob(probs, clas):
    sorted_args = np.argsort(probs, axis=0)
    if np.argmax(probs) == clas:
        return probs[sorted_args[-2]]
    return probs[sorted_args[-1]]

def get_one_hot(index, max_index):
    out = np.zeros(max_index)
    out[index] = 1
    return out

def save_image(image, filename="image.png"):
    image = image.reshape(28, 28)
    plt.imshow(image, cmap='gray')
    plt.savefig(filename)
    return 0

def save_orig_adv_image(orig_image, orig_label, adv_image, adv_label, filename="image.png"):
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(orig_image, cmap='gray')
    plt.title(f"Original image - Label: {orig_label}")
    plt.subplot(1, 2, 2)
    plt.imshow(adv_image, cmap='gray')
    plt.title(f"Adversarial image - Label: {adv_label}")
    plt.savefig(filename)
    return 1

def reconstr_img(pca, image):
    rec_im = pca.inverse_transform(pca.transform(image))
    return rec_im

def save_pca_model(train_dataset, n_components, filename):
    pca = PCA(n_components=n_components, random_state=1).\
        fit(np.reshape(train_dataset, (len(train_dataset), 28 * 28)))
    with open(filename, 'wb') as pickle_file:
        pk.dump(pca, pickle_file)
    return 1

def load_pca_model(filename):
    if not os.path.exists(filename):
        filename = '/home/m4sulaim/CS886/rl-based-blackbox-attack/gym_bb_attack/envs/' + filename
    with open(filename, 'rb') as pickle_file:
        pca = pk.load(pickle_file)
    return pca

if __name__=="__main__":
    d = 784
    # encoder = Encoder(encoded_space_dim=d)
    # load_weights(encoder, "./encoder.pt")
    # decoder = Decoder(encoded_space_dim=d)
    # load_weights(decoder, "./decoder.pt")
    # classifier = Classifier()
    # load_weights(classifier, "./classifier.pth")
    #
    data_dir = './dataset'
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
    train_dataset_array = next(iter(train_loader))[0].numpy()
    save_pca_model(train_dataset_array, d, 'model.pkl')
    pca = PCA(n_components=784, random_state=1).fit(np.reshape(train_dataset_array, (len(train_dataset), 28 * 28)))

    ratio = pca.explained_variance_ratio_
    rev_ratio = ratio[::-1]
    cum_ratio = np.cumsum(rev_ratio)
    for i in range(len(ratio)):
        print(f"{i}: {cum_ratio[i]}")

    # m = len(train_dataset)
    #
    # pca = PCA(n_components=64, random_state=1).\
    #     fit(np.reshape(train_dataset.data, (len(train_dataset), 28 * 28)))
    #
    # mse = 0
    # max_v = -1 * math.inf
    # min_v = [math.inf for _ in range(d)]
    # for im in test_dataset.data:
    #     im_flat = np.reshape(im, (-1, 28*28)).numpy()
    #     enc_im = np.clip(pca.transform(im_flat), -3, 3)
    #     rec_im = pca.inverse_transform(enc_im)
    #     mse += np.sqrt(np.sum(np.square(im_flat - rec_im)))
    # print(mse/len(test_dataset))

    #
    # train_data, val_data = random_split(train_dataset, [int(m - m * 0.2), int(m * 0.2)])
    # batch_size = 1
    #
    # train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    # valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

