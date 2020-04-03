import torch
from tqdm import tqdm
import torch.nn as nn
import torch.functional as F
from keras.datasets import mnist
from torch import optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

(X_train, _), (_, _) = mnist.load_data()

X_train = (X_train-127.5)/127.5  # Normalizing values between [-1,1]

X_train = torch.from_numpy(X_train).float()

X_train = X_train.view(60000,1,28,28)

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

generator = Generator()
discriminator = Discriminator()

d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)

loss = nn.BCELoss()

def noise(size):
    n = torch.randn(size, 100, requires_grad=True)
    return n

def zeros_target(size):
    n = torch.zeros(size,1)
    return n

def ones_target(size):
    n = torch.ones(size,1)
    return n

def train_discriminator(optimizer, real_data, fake_data):
    N = real_data.size(0)
    optimizer.zero_grad()
    prediction_real = discriminator(real_data)
    error_real = loss(prediction_real, ones_target(N))
    error_real.backward()
    prediction_fake = discriminator(fake_data)
    error_fake = loss(prediction_fake, zeros_target(N))
    error_fake.backward()
    optimizer.step()
    return error_real + error_fake, prediction_real, prediction_fake

def train_generator(optimizer, fake_data):
    N = fake_data.size(0)
    optimizer.zero_grad()
    prediction = discriminator(fake_data)
    error = loss(prediction, ones_target(N))
    error.backward()
    optimizer.step()
    return error


num_epochs = 200
for epoch in range(num_epochs):
    print("Epoch {} Running...".format(epoch+1))

    for real_batch in tqdm(data_loader):
        N = real_batch.size(0)
        # 1. Train Discriminator
        real_data = real_batch
        # Generate fake data and detach 
        # (so gradients are not calculated for generator)
        fake_data = generator(noise(N)).detach()
        # Train D
        d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimizer, real_data, fake_data)

        # 2. Train Generator
        # Generate fake data
        fake_data = generator(noise(N)).detach()
        # Train G
        g_error = train_generator(g_optimizer, fake_data)

    print("Generator Error: {}\nDiscriminator Error: {}\n------------------------------------------------------------------".format(g_error, d_error))

