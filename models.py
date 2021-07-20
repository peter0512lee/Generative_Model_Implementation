import torch
import torch.nn as nn
import torch.nn.functional as F

from padding_same_conv import Conv2d


class Encoder(nn.Module):
    def __init__(self, z_dim, ch=64):
        super(Encoder, self).__init__()
        self.ch = ch
        # (3, 64, 64)
        self.conv1 = Conv2d(3, ch, 5, stride=2)
        # (ch, 32, 32)
        self.conv2 = Conv2d(ch, ch*2, 5, stride=2)
        self.bn2 = nn.BatchNorm2d(ch*2)
        # (ch*2, 16, 16)
        self.conv3 = Conv2d(ch*2, ch*4, 5, stride=2)
        self.bn3 = nn.BatchNorm2d(ch*4)
        # (ch*4, 8, 8)
        self.conv4 = Conv2d(ch*4, ch*8, 5, stride=2)
        self.bn4 = nn.BatchNorm2d(ch*8)
        # (ch*8, 4, 4)
        self.fc11 = nn.Linear(4*4*ch*8, z_dim)
        self.fc12 = nn.Linear(4*4*ch*8, z_dim)

    def forward(self, x):
        output = F.relu(self.conv1(x))
        output = F.relu(self.bn2(self.conv2(output)))
        output = F.relu(self.bn3(self.conv3(output)))
        output = F.relu(self.bn4(self.conv4(output)))
        output = output.view(-1, 4*4*64*8)
        z_mu = self.fc11(output)
        z_logvar = self.fc12(output)

        return z_mu, z_logvar


class Decoder(nn.Module):
    def __init__(self, z_dim, ch=64):
        super(Decoder, self).__init__()
        self.ch = ch
        self.fc1 = nn.Linear(z_dim, 8*8*ch*8)
        # (ch*8, 8, 8)
        self.conv1 = Conv2d(ch*8, ch*4, 3)
        # (ch*4, 8, 8)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2 = Conv2d(ch*4, ch*2, 3)
        self.bn2 = nn.BatchNorm2d(ch*2)
        # (ch*2, 32, 32)
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv3 = Conv2d(ch*2, ch, 3)
        self.bn3 = nn.BatchNorm2d(ch)
        # (ch, 16, 16)
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv4 = Conv2d(ch, 3, 3)

    def forward(self, z):
        output = F.relu(self.fc1(z))
        output = output.view(-1, 512, 8, 8)
        output = F.relu(self.conv1(output))
        output = F.relu(self.bn2(self.conv2(self.up2(output))))
        output = F.relu(self.bn3(self.conv3(self.up3(output))))
        output = self.conv4(self.up4(output))
        x_samp = torch.sigmoid(output)

        return x_samp


class Generator(nn.Module):
    def __init__(self, z_dim, ch=64):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # in: latent_size x 1 x 1

            nn.ConvTranspose2d(z_dim, 512, kernel_size=4,
                               stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # out: 512 x 4 x 4

            nn.ConvTranspose2d(512, 256, kernel_size=4,
                               stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # out: 256 x 8 x 8

            nn.ConvTranspose2d(256, 128, kernel_size=4,
                               stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # out: 128 x 16 x 16

            nn.ConvTranspose2d(128, 64, kernel_size=4,
                               stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # out: 64 x 32 x 32

            nn.ConvTranspose2d(64, 3, kernel_size=4,
                               stride=2, padding=1, bias=False),
            nn.Tanh()
            # out: 3 x 64 x 64
        )

    def forward(self, z):
        return self.main(z)


class Generator_(nn.Module):
    def __init__(self, z_dim, ch=64):
        super(Generator_, self).__init__()
        self.ch = ch
        self.fc1 = nn.Linear(z_dim, 8*8*ch*8)
        # (ch*8, 8, 8)
        self.conv1 = Conv2d(ch*8, ch*4, 3)
        # (ch*4, 8, 8)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2 = Conv2d(ch*4, ch*2, 3)
        self.bn2 = nn.BatchNorm2d(ch*2)
        # (ch*2, 32, 32)
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv3 = Conv2d(ch*2, ch, 3)
        self.bn3 = nn.BatchNorm2d(ch)
        # (ch, 16, 16)
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv4 = Conv2d(ch, 3, 3)

    def forward(self, z):
        output = F.relu(self.fc1(z).view(-1, 512, 8, 8))
        output = F.relu(self.conv1(output))
        output = F.relu(self.bn2(self.conv2(self.up2(output))))
        output = F.relu(self.bn3(self.conv3(self.up3(output))))
        output = self.conv4(self.up4(output))
        x_samp = torch.sigmoid(output)

        return x_samp


class Discriminator(nn.Module):
    def __init__(self, ch=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # in: 3 x 64 x 64

            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 64 x 32 x 32

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 128 x 16 x 16

            nn.Conv2d(128, 256, kernel_size=4,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 256 x 8 x 8

            nn.Conv2d(256, 512, kernel_size=4,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 512 x 4 x 4

            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
            # out: 1 x 1 x 1

            nn.Flatten(),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)


class Discriminator_(nn.Module):
    def __init__(self, ch=64):
        super(Discriminator_, self).__init__()
        self.ch = ch
        self.leakyReLU = nn.LeakyReLU(0.2, inplace=True)
        # (3, 64, 64)
        self.conv1 = Conv2d(3, ch, 5, stride=2)
        # (ch, 32, 32)
        self.conv2 = nn.utils.spectral_norm(Conv2d(ch, ch*2, 5, stride=2))
        # (ch*2, 16, 16)
        self.conv3 = nn.utils.spectral_norm(Conv2d(ch*2, ch*4, 5, stride=2))
        # (ch*4, 8, 8)
        self.conv4 = nn.utils.spectral_norm(Conv2d(ch*4, ch*8, 5, stride=2))
        # (ch*8, 4, 4)
        self.fc1 = nn.Linear(4*4*ch*8, 1)

    def forward(self, x):
        output = self.leakyReLU(self.conv1(x))
        output = self.leakyReLU(self.conv2(output))
        output = self.leakyReLU(self.conv3(output))
        output = self.leakyReLU(self.conv4(output))
        output = output.view(-1, 4*4*64*8)
        d_logit = self.fc1(output)
        d_prob = torch.sigmoid(d_logit)
        return d_prob, d_logit
