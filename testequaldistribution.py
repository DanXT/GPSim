# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 13:45:37 2020

@author: xtan
"""

import torch

def getDistMatrix(T,R,W=1):

    nT = T.shape[0]
    nR = R.shape[0]
    
    T = T.view(nT,-1)
    R = R.view(nR,-1)
        
    C = torch.sum(T**2,1).unsqueeze(1) + torch.sum(R**2,1).unsqueeze(0) - 2*torch.matmul(T,R.t())
    return W*C 

import numpy as np

n1 = 20
n2 = 30
d  = 100
n1n2 = n1*n2
n = n1 + n2

#rand_z = np.random.randint(10, size=(n1, d))
rand_z = np.random.normal(size=(n1, d))
rand_z = torch.from_numpy(rand_z).float()


rand_y = np.random.randint(10, size=(n2, d))
#rand_y = np.random.normal(size=(n2, d))
rand_y = torch.from_numpy(rand_y).float()





M12 = getDistMatrix(rand_z,rand_y)  # 0,1 == 2,3,4
M11 = getDistMatrix(rand_z,rand_z)  # 0,1
M11[torch.eye(n1).byte()] = 0
M22 = getDistMatrix(rand_y,rand_y)  # 2,3,4
M22[torch.eye(n2).byte()] = 0


M12 = torch.sqrt(M12)
M11 = torch.sqrt(M11)
M22 = torch.sqrt(M22)



# pooled n1 and n2 samples
A = np.vstack((M12, M22))
B = np.vstack((M11, M12.t()))
M = np.hstack((B, A))





# compute e-statistic
mm12 = torch.sum(M12)
mm11 = torch.sum(M11)
mm22 = torch.sum(M22)

a1 = n1n2/n
b1 = 2/n1n2
c1 = 1/n1**2
d1 = 1/n2**2

e_stat = a1* (b1*mm12-c1*mm11-d1*mm22)



# 
nTest = 10000
C_value = 0

for i in range(nTest):
    idx = torch.randperm(n)
    M_new = np.zeros((n,n))
    for i in range(n):
        xx = M[idx[i], idx]
        M_new[i,:] = xx
        
        
    mm11_new=np.sum(M_new[0:n1,0:n1]) 
    mm12_new=np.sum(M_new[0:n1,n1:n])
    mm22_new=np.sum(M_new[n1:n,n1:n])
    e_sample = a1* (b1*mm12_new-c1*mm11_new-d1*mm22_new)
    
    if e_sample > e_stat:
        C_value = C_value + 1

print(C_value/nTest)


