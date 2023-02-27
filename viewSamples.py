# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 18:00:08 2019

@author: xtan
"""

import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np

pat = 21

# Load samples from generator, taken while training
#with open('train_samples.pkl', 'rb') as f:
#     samples = pkl.load(f)
     
     
with open('train_samples_reso2.pkl', 'rb') as f:
     samples = pkl.load(f)



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
        

    


rows = 10 # split epochs into 10, so 100/10 = every 10 epochs
cols = 6
fig, axes = plt.subplots(figsize=(7,12), nrows=rows, ncols=cols, sharex=True, sharey=True)

for sample, ax_row in zip(samples[::int(len(samples)/rows)], axes):
    for img, ax in zip(sample[::int(len(sample)/cols)], ax_row):
        img = img.detach().cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        img[img>=0]=1
        img[img<0] =-1
        #sz = int(math.sqrt(img.shape[0]))
        ax.imshow(np.squeeze(img))
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        
        
        
        
        
        
        
        
    

import torch
from generator import Generator
conv_dim = 32
z_size = 50

G = Generator(z_size=z_size, conv_dim=conv_dim)

#G.load_state_dict(torch.load('weights_G.pth'))

G.load_state_dict(torch.load('weights_G_reso2.pth'))

# randomly generated, new latent vectors
sample_size= 100
rand_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
rand_z = torch.from_numpy(rand_z).float()





G.eval() # eval mode
# generated samples
rand_images = G(rand_z)

# 0 indicates the first set of samples in the passed in list
# and we only have one batch of samples, here
view_samples(0, [rand_images]) 
    
    
Y2 = G(rand_z)
view_samples(0, [Y2]) 

Y2[Y2>=0]=1
Y2[Y2<0] =-1
    
view_samples(0, [Y2]) 
    
Y2[Y2==-1]=0
    




