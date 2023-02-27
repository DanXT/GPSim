# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 23:14:33 2019

@author: xtan
"""
from IPython import get_ipython
get_ipython().magic('reset -sf')

from utils import MyDataset
patchSet = MyDataset('Patterns21.mat')
#patchSet = MyDataset('Patterns21_reso2.mat')

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

fig = plt.figure(figsize=(16, 4))
plot_size=16
for idx in np.arange(plot_size):
    ax = fig.add_subplot(2, plot_size/2, idx+1, xticks=[], yticks=[])
    ax.imshow(np.reshape(images[idx,:], (pat,pat)))




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