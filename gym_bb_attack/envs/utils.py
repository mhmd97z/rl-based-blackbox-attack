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


class Encoder(nn.Module):

    def __init__(self, encoded_space_dim, fc2_input_dim):
        super().__init__()

        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True)
        )

        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)

        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(3 * 3 * 32, 128),
            nn.ReLU(True),
            nn.Linear(128, encoded_space_dim)
        )

    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x


class Decoder(nn.Module):

    def __init__(self, encoded_space_dim, fc2_input_dim):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 3 * 3 * 32),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1,
                                      unflattened_size=(32, 3, 3))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3,
                               stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2,
                               padding=1, output_padding=1)
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
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
    model.load_state_dict(torch.load(filename, map_location=torch.device('cpu')))

def encode_image(image, encoder):
    image_tensor = torch.from_numpy(np.array(np.reshape(image, (1, 1, 28, 28)), dtype=np.float32))
    return encoder(image_tensor).detach().numpy()[0]

def decode_image(encoding, decoder):
    encoding = torch.Tensor(encoding).unsqueeze(0)
    return decoder(encoding).detach().numpy()[0][0]

def get_random_image(train_dataset):
    rand_index = np.random.randint(len(train_dataset))
    return np.array(train_dataset[rand_index][0])/255

def get_random_encoded_image(train_dataset, encoder):
    return encode_image(get_random_image(train_dataset), encoder)

def classify_image(image, classifier):
    image = torch.Tensor(image).unsqueeze(0).unsqueeze(0)
    return classifier(image).detach().numpy()[0]

def get_class_prob(image, clas, classifier):
    image = torch.Tensor(image).unsqueeze(0).unsqueeze(0)
    return classifier(image).detach().numpy()[0][clas]

def get_one_hot(index, max_index):
    out = np.zeros(max_index)
    out[index] = 1
    return out

if __name__=="__main__":
    d = 4
    encoder = Encoder(encoded_space_dim=d,fc2_input_dim=128)
    load_weights(encoder, "./encoder.pt")
    decoder = Decoder(encoded_space_dim=d,fc2_input_dim=128)
    load_weights(decoder, "./decoder.pt")
    classifier = Classifier()
    load_weights(classifier, "./classifier.pth")

    data_dir = 'dataset'

    train_dataset = torchvision.datasets.MNIST(data_dir, train=True, download=True)
    test_dataset = torchvision.datasets.MNIST(data_dir, train=False, download=True)

    m = len(train_dataset)

    train_data, val_data = random_split(train_dataset, [int(m - m * 0.2), int(m * 0.2)])
    batch_size = 1

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

