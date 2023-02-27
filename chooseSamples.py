# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 15:15:07 2019

@author: xtan
"""
import numpy as np
import torch
from generator import Generator
import matplotlib.pyplot as plt
from imresize_matlab import graythresh
conv_dim = 32
z_size = 50
pat = 21


G = Generator(z_size=z_size, conv_dim=conv_dim)

G.load_state_dict(torch.load('weights_G.pth'))
#G.load_state_dict(torch.load('weights_G_reso2.pth'))


# randomly generated, new latent vectors
sample_size= 3200
rand_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
rand_z = torch.from_numpy(rand_z).float()

def view_samples(epoch, samples):
    fig, axes = plt.subplots(figsize=(16,16), nrows=5, ncols=5, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples[epoch]):
        img = img.detach().cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        #img = ((img +1)*255 / (2)).astype(np.uint8) # rescale to pixel range (0-255)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(img.reshape((pat,pat)))
        
G.eval() # eval mode
# generated samples
Y         = G(rand_z)
Y2        = Y
Y2[Y2>=0] = 1
Y2[Y2<0]  =-1
view_samples(0, [Y2])


Y2[Y2==-1]=0
Y2 = Y2.reshape((Y2.shape[0],-1))
torch.save(Y2, "newpatches.pt")


