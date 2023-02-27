# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 14:05:08 2019

@author: xtan
"""


from scipy.io import loadmat
import torch
from torch.utils.data import Dataset



class MyDataset(Dataset):
    def __init__(self, mat_path):
        data = loadmat(mat_path)
        x    = data['X']
        self.images = torch.Tensor(x) 
        # self.images = torch.from_numpy(x)

    def __getitem__(self, index):
        x = self.images[index]
        return x

    def __len__(self):
        return len(self.images)
    
    


