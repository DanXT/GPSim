# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 08:16:21 2019

@author: xtan
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 20:42:53 2019

@author: xtan
"""

#from IPython import get_ipython
#get_ipython().magic('reset -sf')


from scipy.io import loadmat


dic = loadmat('TI.mat')
out = dic['out']

import numpy as np

out = np.int8(out)
#out[out==0]=-1

dimx_TI = 101
dimy_TI = 101
out = out[0:dimx_TI,0:dimy_TI]

# coherence map
idcm = range(dimx_TI*dimy_TI-1,-1,-1)
# coherence map
cm1 = np.reshape(idcm, [dimx_TI,dimy_TI])
cm = np.fliplr(cm1)
import matplotlib.pyplot as plt
#plt.imshow(cm)
#plt.colorbar()
#plt.show()
out1 = np.dstack((out, cm))

#plt.imshow(out, cmap = "gray")
#plt.show()
#plt.imshow(out_flip1, cmap = "gray")
#plt.show()
#plt.imshow(out_flip2, cmap = "gray")
#plt.show()

import argparse
# Training settings
parser = argparse.ArgumentParser(description='geostat')
parser.add_argument('--dimx_TI', type=int, default=out.shape[0], help='dimensionTI x')
parser.add_argument('--dimy_TI', type=int, default=out.shape[1], help='dimensionTI y')
parser.add_argument('--dimx_Re', type=int, default=101, help='dimensionRe x')
parser.add_argument('--dimy_Re', type=int, default=101, help='dimensionRe y')
parser.add_argument('--dimz', type=int, default=1, help='dimension z')
parser.add_argument('--pat', type=int, default=21, help='patch size')
parser.add_argument('--patz', type=int, default=1, help='patch size z')
parser.add_argument('--innerPatch', type=int, default=15, help='inner patch size')
parser.add_argument('--innerPatchz', type=int, default=1, help='inner patch size z')
parser.add_argument('--mr', type=int, default=3, help='multi resolution')
parser.add_argument('--m', type=int, default=1, help='number of patterns to be skipped')
parser.add_argument('--flip', type=int, default=0, help='whether to flip the patches')
parser.add_argument('--addnew', type=int, default=3200, help='how many new patches needed')

opt = parser.parse_args()

print(opt)


numRe = 5
pat = 21
innerpatch = 15
# 0: use original patches
# 1: use only new patches
# 2: use both original and new patches
UsingNewPatches = 2
m = 1
mr = 3
addnew =3200
flip = 1

realizations = np.zeros((opt.dimx_Re, opt.dimy_Re, opt.dimz, numRe))
coherenceMap = np.zeros((opt.dimx_Re, opt.dimy_Re, opt.dimz, numRe))

emptyGrid = np.zeros((opt.dimx_Re, opt.dimy_Re))


from dcgan_geostat import DCGAN_geostat
for i in range(numRe):
    opt.pat = pat
    opt.innerPatch = innerpatch
    opt.m  = m
    opt.mr = mr
    opt.flip = flip
    opt.addnew = addnew
    realizations[:,:,:,i], coherenceMap[:,:,:,i] = DCGAN_geostat(out1, opt, emptyGrid, UsingNewPatches)
    
#import pickle as pkl
## Save training generator samples
#with open('realizations_original.pkl', 'wb') as f:
#    pkl.dump(realizations, f)
plt.imshow(np.squeeze(realizations[:,:,:,1]))
plt.show()

import matplotlib
cM = np.squeeze(coherenceMap[:,:,:,0])
masked_array = np.ma.masked_where(cM == -2, cM)
cmap = matplotlib.cm.spring  # Can be any colormap that you want after the cm
cmap.set_bad(color='white')

plt.imshow(masked_array, cmap=cmap)
plt.show()


import scipy.io as sio
#sio.savemat('realizations_101_301_21_15_1_2_gan.mat', {
#        'realizations': realizations}) 

#                     _patchsize_inner_mr_m_original
#sio.savemat('realizations_101_301_21_15_1_2_original.mat', {
#        'realizations': realizations}) 
#sio.savemat('realizations_gan_3200new.mat', {
#        'realizations': realizations}) 
sio.savemat('realizations_original_flip.mat', {
        'realizations': realizations}) 
#sio.savemat('realizations_original_3200new.mat', {
#       'realizations': realizations}) 

#sio.savemat('comap_101_301_21_15_1_2_original.mat', {
#        'comap': coherenceMap})

#sio.savemat('realizations_101_301_21_15_1_1_new.mat', {
#        'realizations': realizations}) 

#sio.savemat('comap_101_301_21_15_1_1_new.mat', {
#        'comap': coherenceMap})

#sio.savemat('realizations_101_301_21_15_1_2_new_original.mat', {
#        'realizations': realizations}) 

#sio.savemat('comap_101_301_21_15_1_2_new_original.mat', {
#        'comap': coherenceMap})


    
    
    
    