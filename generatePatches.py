# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 02:22:58 2019

@author: xtan
"""

import numpy as np
import torch
import torch.optim as optim
import pickle as pkl
from GANs_fun import Discriminator, Generator, real_loss, fake_loss, TVnorm
import matplotlib.pyplot as plt
import random
import math



def gan4patches(X, opt):
    
    Y1 = X
    
    pat = opt.pat
    out = np.int32(X)
    out[out==0]=-1
    
    X = out
    X = np.reshape(X, (X.shape[0],1,pat,pat))
    
    batch_size = 64
    num_batch = int(np.floor(X.shape[0]/batch_size))
      
    X = torch.from_numpy(X).float()
    
    # Discriminator hyperparams

#    batch_size = 64
#    num_batch  = torch.floor(X.shape[0]/64)
    # Size of input image to discriminator (28*28)
    input_size = pat**2  #225
    # Size of discriminator output (real or fake)
    d_output_size = 1
    # Size of last hidden layer in the discriminator
    d_hidden_size = 32

    # Generator hyperparams

    # Size of latent vector to give to generator
    z_size = 100
    # Size of discriminator output (generated image)
    g_output_size = pat**2
    # Size of first hidden layer in the generator
    g_hidden_size = 32

    # instantiate discriminator and generator
    D = Discriminator(input_size, d_hidden_size, d_output_size)
    G = Generator(z_size, g_hidden_size, g_output_size)
    
    

    # Optimizers
    lr = 0.0005

    # Create optimizers for the discriminator and generator
    d_optimizer = optim.Adam(D.parameters(), lr)
    g_optimizer = optim.Adam(G.parameters(), lr)



    # training hyperparams
    num_epochs = 800

    # keep track of loss and generated, "fake" samples
    samples = []
    losses = []

    print_every = 400

    miu = 0.0001

    # Get some fixed data for sampling. These are images that are held
    # constant throughout training, and allow us to inspect the model's performance
    sample_size= 1600
    fixed_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
    fixed_z = torch.from_numpy(fixed_z).float()


    #
    #container_z = np.random.uniform(-1,1,size=(500,z_size))
    #container_z = torch.from_numpy(container_z).float()

    # train the network
    D.train()
    G.train()
    for epoch in range(num_epochs):
        
        d_loss_avg = 0.0
        g_loss_avg = 0.0
    
    
        for b_i in range(num_batch):
            real_images = X[b_i*batch_size:(b_i+1)*batch_size,:,:,:]
            # print(b_i*batch_size,(b_i+1)*batch_size)
        
            # ============================================
            #            TRAIN THE DISCRIMINATOR
            # ============================================
        
            d_optimizer.zero_grad()
        
            # 1. Train with real images

            # Compute the discriminator losses on real images 
            # smooth the real labels
            D_real = D(real_images)
            d_real_loss = real_loss(D_real, smooth=False)
            
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
        
        
        #        if torch.isnan(D.fc1.weight.grad).any():
        #            aaaa=1
        
        
        
        
        
        # =========================================
        #            TRAIN THE GENERATOR
        # =========================================
            g_optimizer.zero_grad()
        
        # 1. Train with fake images and flipped labels
        
        # Generate fake images
        #aa= [randint(0,499) for  p in range(0,64)]
            
            # from fixed samples, choose 16 of them
            ind_z = random.sample(range(0, sample_size), batch_size) 
            z = fixed_z[ind_z,:]
            

            #z = np.random.uniform(-1, 1, size=(batch_size, z_size))
            #z = torch.from_numpy(z).float()
            fake_images = G(z)
        
        # total variation loss
            TV_loss = TVnorm(fake_images)
        
        
        
        
        # Compute the discriminator losses on fake images 
        # using flipped labels!
            D_fake = D(fake_images)
            g_loss1 = real_loss(D_fake)
            reg = miu*TV_loss
            g_loss = g_loss1 + reg# use real loss to flip labels
        
        #print(G.fc1.weight.grad)
        # perform backprop
            g_loss.backward()
        
        
        
        
            g_optimizer.step()
        
        
            d_loss_avg = d_loss_avg + d_loss.item()
            g_loss_avg = g_loss_avg + g_loss.item()
        
        

        # Print some loss stats
        # print discriminator and generator loss
            print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f} | TV_loss: {:6.4f}'.format(
                    epoch+1, num_epochs, d_loss.item(), g_loss1.item(), reg.item()))
        #    print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
        #             epoch+1, num_epochs, d_loss.item(), g_loss1.item()))

    
    ## AFTER EACH EPOCH##
    # append discriminator loss and generator loss
        d_loss_avg = d_loss_avg/num_batch
        g_loss_avg = g_loss_avg/num_batch
        losses.append((d_loss_avg, g_loss_avg))
    
    # generate and save sample, fake images
        G.eval() # eval mode for generating samples
        samples_z = G(fixed_z[0:16,:])
        #samples_z = G(fixed_z)

        samples.append(samples_z)
        G.train() # back to train mode


# Save training generator samples
    with open('train_samples.pkl', 'wb') as f:
        pkl.dump(samples, f)
    

    fig, ax = plt.subplots()
    losses = np.array(losses)
    plt.plot(losses.T[0], label='Discriminator')
    plt.plot(losses.T[1], label='Generator')
    plt.title("Training Losses")
    plt.legend()


# helper function for viewing a list of passed in sample images
    def view_samples(epoch, samples):
        fig, axes = plt.subplots(figsize=(7,7), nrows=4, ncols=4, sharey=True, sharex=True)
        for ax, img in zip(axes.flatten(), samples[epoch]):
            img = img.detach()
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            sz2 = samples[epoch].shape[1]
            sz  = int(math.sqrt(sz2))
            im = ax.imshow(img.reshape((sz,sz)), cmap='Greys_r')
        

# Load samples from generator, taken while training
    with open('train_samples.pkl', 'rb') as f:
        samples = pkl.load(f)
    
# -1 indicates final epoch's samples (the last in the list)
    view_samples(-1, samples)

    rows = 10 # split epochs into 10, so 100/10 = every 10 epochs
    cols = 6
    fig, axes = plt.subplots(figsize=(7,12), nrows=rows, ncols=cols, sharex=True, sharey=True)

    for sample, ax_row in zip(samples[::int(len(samples)/rows)], axes):
        for img, ax in zip(sample[::int(len(sample)/cols)], ax_row):
            img = img.detach()
            sz = int(math.sqrt(img.shape[0]))
            ax.imshow(img.reshape((sz,sz)), cmap='Greys_r')
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
        
# randomly generated, new latent vectors
    sample_size=16
    rand_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
    rand_z = torch.from_numpy(rand_z).float()

    G.eval() # eval mode
# generated samples
    rand_images = G(rand_z)

# 0 indicates the first set of samples in the passed in list
# and we only have one batch of samples, here
    view_samples(0, [rand_images]) 
    
    
    Y2 = G(fixed_z)
    view_samples(0, [Y2]) 

    Y2[Y2>=0]=1
    Y2[Y2<0] =-1
    
    view_samples(0, [Y2]) 
    
    Y2[Y2==-1]=0
    
    
    Y = np.concatenate((Y1, Y2.detach()), axis=0)
    
    #with open('patches%s.pkl' % str(opt.m), 'wb') as f:
    #    pkl.dump(Y, f)
    
    
    return Y
    
    