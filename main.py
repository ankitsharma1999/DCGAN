import torch
import torch.nn as nn
import torch.functional as F
from keras.datasets import mnist
from torch import optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.distributions.normal import Normal

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = torch.from_numpy(X_train).float()
X_test = torch.from_numpy(X_test).float()

X_train, X_test = X_train.view(60000,1,28,28), X_test.view(10000,1,28,28)

batch_size = 120

data_loader = DataLoader(X_train, batch_size=batch_size, shuffle=True)
num_batches = len(data_loader)


class Generator(nn.Module):

    def __init__(self):

        super(Generator, self).__init__()

        num_ip = 100
        

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

        x = x.view(120,256,7,7)

        x = self.conv_1(x)
        x = self.bn_2(x)
        x = self.a_2(x)

        x = self.conv_2(x)
        x = self.a_3(x)

        return x

class Discriminator(nn.Module):

    def __init__(self):

        super(Discriminator, self).__init__()


        self.conv_1 = nn.Conv2d(1,64, kernel_size=2, stride=2)
        self.a_1 = nn.LeakyReLU(0.2)
        self.dp_1 = nn.Dropout2d(0.3)

        self.conv_2 = nn.Conv2d(64, 128, kernel_size=2, stride=2)
        self.a_2 = nn.LeakyReLU(0.2)
        self.dp_2 = nn.Dropout2d(0.3)

        self.lin = nn.Linear(128*7*7, 1)
        self.op = nn.Sigmoid()

    def forward(self, x):

        self.lin.weight.data.normal_(0,0.02)
        self.conv_1.weight.data.normal_(0,0.02)
        self.conv_2.weight.data.normal_(0,0.02)

        x = self.conv_1(x)
        x = self.a_1(x)
        x = self.dp_1(x)

        x = self.conv_2(x)
        x = self.a_2(x)
        x = self.dp_2(x)

        x = x.view(120, 128*7*7)

        x = self.lin(x)
        x = self.op(x)
        return x
