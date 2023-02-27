# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 18:24:27 2019

@author: xtan
"""
import torch
import torch.nn as nn

def real_loss(D_out, smooth=False):
    batch_size = D_out.size(0)
    # label smoothing
    if smooth:
        # smooth, real labels = 0.9
        labels = torch.ones(batch_size)*0.9
    else:
        labels = torch.ones(batch_size) # real labels = 1
   
    # binary cross entropy with logits loss
    criterion = nn.BCEWithLogitsLoss()
    # calculate loss
    #print(D_out.shape)
    #print(labels.shape)
    D_out = torch.squeeze(D_out, 1)
    loss = criterion(D_out, labels)
    return loss

def fake_loss(D_out):
    batch_size = D_out.size(0)
    labels = torch.zeros(batch_size) # fake labels = 0
  
    criterion = nn.BCEWithLogitsLoss()
    # calculate loss
    D_out = torch.squeeze(D_out, 1)
    loss = criterion(D_out, labels)
    return loss