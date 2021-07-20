import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import models
from read_celeba import *

# Decide which device we want to run on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")s

# Construct Model
z_dim = 128
netGen = models.Generator_(z_dim).to(device)
netDis = models.Discriminator_().to(device)
optGen = optim.Adam(netGen.parameters(), lr=4e-4, betas=(0.5, 0.999))
optDis = optim.Adam(netDis.parameters(), lr=1e-4, betas=(0.5, 0.999))

# Create Results Folder
model_name = "gan"
out_folder = "out/" + model_name + "/"
if not os.path.exists(out_folder):
    os.makedirs(out_folder)
save_folder = "save/" + model_name + "/"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Main Loop
num_epochs = 20
z_fixed = torch.randn(36, z_dim, device=device)
# z_fixed = torch.randn(64, z_dim, 1, 1, device=device)
print("Start Training ...")
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):

        data = data[0].to(device)
        b_size = data.size(0)

        # Discriminator
        netGen.zero_grad()
        netDis.zero_grad()

        '''
        TODO: Compute probability of real data and construct labels
        '''
        # Pass real images through discriminator
        d_real, _ = netDis(data)
        d_label = torch.ones(data.size(0), 1, device=device)
        d_real_loss = nn.BCELoss()(d_real, d_label)
        # d_real_loss.backward()

        '''
        TODO: Compute probability of fake data and construct labels
        '''
        # Generate fake images
        latent = torch.randn(b_size, z_dim, device=device)
        fake_img = netGen(latent)

        # Pass fake images through discriminator
        d_fake, _ = netDis(fake_img)
        d_label = torch.zeros(fake_img.size(0), 1, device=device)
        d_fake_loss = nn.BCELoss()(d_fake, d_label)
        # d_fake_loss.backward()

        d_loss = d_real_loss + d_fake_loss
        d_loss.backward(retain_graph=True)
        optDis.step()

        # Generator
        netGen.zero_grad()
        '''
        TODO: Compute probability of fake data and construct labels
        '''
        d_label = torch.ones(b_size, 1, device=device)
        d_fake, _ = netDis(fake_img)
        g_loss = nn.BCELoss()(d_fake, d_label)
        g_loss.backward()
        optGen.step()

        # Show Information
        if i % 100 == 0:
            print("[%d/%d][%s/%d] D_loss: %.4f | G_loss: %.4f"
                  % (epoch+1, num_epochs, str(i).zfill(4), len(dataloader), d_loss.item(), g_loss.item()))

        if i % 500 == 0:
            print("Generate Images & Save Models ...")
            # Output Images
            x_fixed = netGen(z_fixed).cpu().detach()
            plt.figure(figsize=(6, 6))
            plt.imshow(np.transpose(vutils.make_grid(
                x_fixed, nrow=6, padding=2, normalize=True).cpu(), (1, 2, 0)))
            plt.axis("off")
            plt.savefig(out_folder+str(epoch).zfill(2)+"_" +
                        str(i).zfill(4)+".jpg", bbox_inches="tight")
            plt.close()
            # Save Model
            torch.save(netGen.state_dict(), save_folder+"netGen.pt")
            torch.save(netDis.state_dict(), save_folder+"netDis.pt")
