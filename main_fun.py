# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 13:47:23 2019

@author: xtan

"""



from IPython import get_ipython
get_ipython().magic('reset -sf')

from utils import MyDataset
#patchSet = MyDataset('Patterns21.mat')
patchSet = MyDataset('Patterns21_reso2.mat')

pat = 21

batch_size = 32
num_workers = 0

import torch
# build DataLoaders for patches
train_loader = torch.utils.data.DataLoader(dataset=patchSet,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=num_workers)


# obtain one batch of training images
dataiter = iter(train_loader)
images = dataiter.next()

import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(25, 4))
plot_size=20
for idx in np.arange(plot_size):
    ax = fig.add_subplot(2, plot_size/2, idx+1, xticks=[], yticks=[])
    ax.imshow(np.reshape(images[idx,:], (pat,pat)), cmap='gray')
    
    
    
# define hyperparams
conv_dim = 32
z_size = 50

from discriminator import Discriminator
from generator import Generator

# define discriminator and generator
D = Discriminator(in_channels = 1, conv_dim=conv_dim)
G = Generator(z_size=z_size, conv_dim=conv_dim)

print(D)
print()
print(G)





import torch.optim as optim

# params
lr = 0.0002
beta1=0.5
beta2=0.999 # default value

# Create optimizers for the discriminator and generator
d_optimizer = optim.Adam(D.parameters(), lr, [beta1, beta2])
g_optimizer = optim.Adam(G.parameters(), lr, [beta1, beta2])



import pickle as pkl

# training hyperparams
num_epochs = 500

# keep track of loss and generated, "fake" samples
samples = []
losses = []

print_every = 300

# Get some fixed data for sampling. These are images that are held
# constant throughout training, and allow us to inspect the model's performance
sample_size=16
fixed_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
fixed_z = torch.from_numpy(fixed_z).float()

from losses import real_loss, fake_loss

# train the network
for epoch in range(num_epochs):
    
    for batch_i, real_images in enumerate(train_loader):
                
        batch_size = real_images.size(0)
        
        # important rescaling step
        #real_images = scale(real_images)
        
        # ============================================
        #            TRAIN THE DISCRIMINATOR
        # ============================================
        
        d_optimizer.zero_grad()
        
        # 1. Train with real images

        # Compute the discriminator losses on real images 
        
        D_real = D(real_images)
        d_real_loss = real_loss(D_real)
        
        # 2. Train with fake images
        
        # Generate fake images
        z = np.random.uniform(-1, 1, size=(batch_size, z_size))
        z = torch.from_numpy(z).float()
        
       
        fake_images = G(z)
        
        # Compute the discriminator losses on fake images            
        D_fake = D(fake_images)
        d_fake_loss = fake_loss(D_fake)
        
        # add up loss and perform backprop
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        d_optimizer.step()
        
        
        # =========================================
        #            TRAIN THE GENERATOR
        # =========================================
        g_optimizer.zero_grad()
        
        # 1. Train with fake images and flipped labels
        
        # Generate fake images
        z = np.random.uniform(-1, 1, size=(batch_size, z_size))
        z = torch.from_numpy(z).float()
        fake_images = G(z)
        
        # Compute the discriminator losses on fake images 
        # using flipped labels!
        D_fake = D(fake_images)
        g_loss = real_loss(D_fake) # use real loss to flip labels
        
        # perform backprop
        g_loss.backward()
        g_optimizer.step()

        # Print some loss stats
        if batch_i % print_every == 0:
            # append discriminator loss and generator loss
            losses.append((d_loss.item(), g_loss.item()))
            # print discriminator and generator loss
            print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                    epoch+1, num_epochs, d_loss.item(), g_loss.item()))
            
    ## AFTER EACH EPOCH##    
    # generate and save sample, fake images
    G.eval() # for generating samples
  
    samples_z = G(fixed_z)
    samples.append(samples_z)
    G.train() # back to training mode


torch.save(G.state_dict(), 'weights_G_reso2_31.pth')

# Save training generator samples
with open('train_samples_31.pkl', 'wb') as f:
    pkl.dump(samples, f)

#with open('train_samples_reso2.pkl', 'wb') as f:
#    pkl.dump(samples, f)



fig, ax = plt.subplots()
losses = np.array(losses)
plt.plot(losses.T[0], label='Discriminator', alpha=0.5)
plt.plot(losses.T[1], label='Generator', alpha=0.5)
plt.title("Training Losses")
plt.legend()


# helper function for viewing a list of passed in sample images
def view_samples(epoch, samples):
    fig, axes = plt.subplots(figsize=(16,4), nrows=2, ncols=8, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples[epoch]):
        img = img.detach().cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        #img = ((img +1)*255 / (2)).astype(np.uint8) # rescale to pixel range (0-255)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(img.reshape((pat,pat)))
        
_ = view_samples(-1, samples)






