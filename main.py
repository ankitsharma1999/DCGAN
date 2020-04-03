import torch
import torch.nn as nn
import torch.functional as F
from keras.datasets import mnist
from torch import optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.distributions.normal import Normal

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train, y_train, X_test, y_test = torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float(), torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float()


class Generator(nn.Module):

    def __init__(self, batch_size):

        super(Generator, self).__init__()

        num_ip = 100
        
        self.batch_size = batch_size

        self.ip = nn.Linear(num_ip, 7*7*256)
        self.bn_1 = nn.BatchNorm1d(7*7*256, affine=False)
        self.a_1 = nn.LeakyReLU(0.2)

        self.conv_1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.bn_2 = nn.BatchNorm2d(128, affine=False)
        self.a_2 = nn.LeakyReLU(0.2)

        self.conv_2 = nn.ConvTranspose2d(128, 1, kernel_size=2, stride=2)
        self.a_3 = nn.Tanh()

    def forward(self,x):

        # Initializing all the weights according to a normal distribution with 0 mean and 0.02 standard deviation as mentioned in the DCGAN paper.
        self.ip.weight.data.normal_(0,0.02)
        self.conv_1.weight.data.normal_(0,0.02)
        self.conv_2.weight.data.normal_(0,0.02)

        x = self.ip(x)
        x = self.bn_1(x)
        x = self.a_1(x)

        x = x.view(self.batch_size,256,7,7)

        x = self.conv_1(x)
        x = self.bn_2(x)
        x = self.a_2(x)

        x = self.conv_2(x)
        x = self.a_3(x)

        return x

