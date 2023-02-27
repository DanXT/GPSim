# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 08:20:47 2019

@author: xtan
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 21:14:53 2019

@author: xtan
"""
import torch
from imresize_matlab import imresize, graythresh
#import cv2 as cv
import matplotlib.pyplot as plt
import math
import numpy as np
from extractPatterns import extractPatterns, ind2sub, findClosestPattern_Non, existNonFrozenNodes, pastePattern, getDataEvent, getPatternShape,pastePattern2
import random
#from generatePatches import gan4patches
import pickle as pkl


def DCGAN_geostat(out,opt, grid, usingNew):
    out1 = out
    
    # work on training image
    mr = opt.mr
    out_c = out[:,:,0]
    out = imresize(out_c, scalar_scale=1/mr)
    level = graythresh(out) 
    BW = out>level
    out = BW
    
    
    # work on realizations
    aaa = imresize(grid, scalar_scale=1/mr)
    
    opt.dimx = aaa.shape[0]
    opt.dimy = aaa.shape[1] 
    opt.szRealizationx = opt.dimx + (opt.pat -1)
    opt.szRealizationy = opt.dimy + (opt.pat -1)
    opt.szRealizationz = opt.dimz
    realization = 0.5*np.ones([opt.szRealizationx,opt.szRealizationy,opt.szRealizationz])
    
    
    for m1 in range(mr,0,-1):
        # work on training image
        out = imresize(out_c,scalar_scale=1/m1)
        
        level = graythresh(out)
        BW = out>level
        out = BW
        
        aaa = imresize(grid, scalar_scale=1/m1)
        opt.dimx = aaa.shape[0]
        opt.dimy = aaa.shape[1]
        
        if m1 != mr:
            # work on realization
            realization = imresize(realization, scalar_scale=(m1+1)/m1)
            realization = +realization
            level = graythresh(realization)
            BW = realization>level
            realization = BW
            
            opt.szRealizationx  = opt.dimx  + (opt.pat  - 1)
            opt.szRealizationy  = opt.dimy  + (opt.pat  - 1)
            opt.szRealizationz  = opt.dimz
            difdx = (realization.shape[0]  - opt.szRealizationx)/2
            difdy = (realization.shape[1]  - opt.szRealizationy)/2
        
            realization = realization[math.floor(difdx):-math.ceil(difdx),math.floor(difdy):-math.ceil(difdy),:]
#            
#            
        # Store the frozen nodes in each coarse simulation
        frozenRealiz = np.zeros([opt.szRealizationx, opt.szRealizationy, opt.szRealizationz])
#       
        

        
        coherenceMap = 0.5*np.ones([realization.shape[0],realization.shape[1],realization.shape[2]])
        # 
        
        #out2 = np.dstack((out,out1[:,:,1]))
        # work on training image
        
        X = extractPatterns(out,opt.pat,opt.m)
        if opt.flip == 1:
            out_flip1 = np.flip(out, 0)
            out_flip2 = np.flip(out, 1)
            X1 = extractPatterns(out_flip1,opt.pat,opt.m)
            X2 = extractPatterns(out_flip2,opt.pat,opt.m)
            X = np.concatenate((X, X1, X2), axis=0) 

#       Locdb = Locdb + (par.Pat -1)/2;
        
        
        if m1 == 1:
            CMM = extractPatterns(out1[:,:,1],opt.pat,opt.m)

            if usingNew != 0: # if new patches are used
                Y2 = torch.load("newpatches.pt")
                # fake coherence map
                Y_CM = -100*np.ones([Y2.shape[0], Y2.shape[1]])
                
                if usingNew == 2: # use both original and new patches
                    CMM = np.concatenate((CMM,Y_CM), axis=0)
                    Y2 = Y2[0:opt.addnew,:]
                    X = np.concatenate((X, Y2.detach()), axis=0) 
                if usingNew == 1: # use only new patches
                    CMM = Y_CM
                    Y2 = Y2[0:opt.addnew,:]
                    X = Y2.detach().numpy()
                    #import scipy.io as sio

                    #sio.savemat('newpatches3200.mat', {'patches_new': X}) 
                
                
                
            
        
        
        
#        fig = plt.figure(figsize=(25,4))
#        
#        numX = X.shape[0]
#        step0 = np.ceil(numX/20)
#        ii = 0
#
#        for idx in np.arange(0,numX, int(step0)):
#            ii = ii + 1
#            ax = fig.add_subplot(2, 10, ii, xticks = [], yticks =[])
#            ax.imshow(np.reshape(X[idx,:], (opt.pat,opt.pat)), cmap='gray')
        
        #plt.show()
        #######################################################################
        #######################################################################
        ################## USE GAN to GENERATE MORE PATCHES ###################
        #######################################################################
        
        #if #exist the saved files
             #load that file
        #else #apply GAN
    
        #X = gan4patches(X,opt)
        
        
        
        
        
        #######################################################################
        #######################################################################
        #######################################################################
#            
#       # work on realization
        # Define a random path throught the grid nodes
        opt.szCoarseGridx = int(np.fix((opt.dimx  - 1))+1)
        opt.szCoarseGridy = int(np.fix((opt.dimy  - 1))+1)
        opt.szCoarseGridz= int(np.fix((opt.dimz - 1))+1)
        lengthRandomPath = opt.szCoarseGridx*opt.szCoarseGridy*opt.szCoarseGridz
        
        
        random.seed(5+m1)
        randomPath       = np.random.permutation(lengthRandomPath)
        
        

        
        #random.seed(10+m1)
        #randomPatches       = np.random.randint(X.shape[0], size=lengthRandomPath)
        #if m1 == 2:
            # torch.save(randomPath, "randomPath_reso_2.pt")
            #randomPath = torch.load("randomPath_reso_2.pt")
            #torch.save(randomPatches, "randomPatches_reso_2.pt")
            #randomPatches = torch.load("randomPatches_reso_2.pt")
        #if m1 == 1:
            #torch.save(randomPath, "randomPath_reso_1.pt")
            #torch.save(randomPatches, "randomPatches_reso_1.pt")
            #randomPath = torch.load("randomPath_reso_1.pt")
            #randomPatches = torch.load("randomPatches_reso_1.pt")
            
            
        #ix                  = 0
        
        
#        if m1 == 2:
#            torch.save(randomPath, "randomPath_reso_2.pt")
#            torch.save(randomPatches, "randomPatches_reso_2.pt")
#        if m1 == 1:
#            torch.save(randomPath, "randomPath_reso_1.pt")
#            torch.save(randomPatches, "randomPatches_reso_1.pt")
        
        # Change to subscripts
        nodeI, nodeJ, nodeK = np.unravel_index(randomPath,  (opt.szCoarseGridx,opt.szCoarseGridy,opt.szCoarseGridz), order = 'F')
        #[nodeI, nodeJ, nodeK] = ind2sub([opt.szCoarseGridx,opt.szCoarseGridy,opt.szCoarseGridz], randomPath)
        node                  = np.vstack((nodeI,nodeJ,nodeK));
        [wx, wy, wz]          = getPatternShape(node, opt);
#    
        opt.m = opt.m + 1
#            
        # ---------------------------------------------------------------------
        # Perform simulation
        # for each node in the random path Do:
        # ---------------------------------------------------------------------
        # wb = wb + 1;
        for i in range(lengthRandomPath):
                
#               waitbar(i/lengthRandomPath*(1/par.MR)+(1/par.MR)*(wb-1),h,sprintf('stage%d : node:%5d : %3.0f%%',wb,i,100*i/lengthRandomPath));
#        
                if frozenRealiz[wx[i, (opt.pat-1)//2], wy[i, (opt.pat-1)//2], wz[i, (opt.patz-1)//2]] == 1:
                    continue
                
                dataEvent, status = getDataEvent(realization, wx[i,:], wy[i,:], wz[i,:]);
                weightEvent     = np.ones([1,opt.pat**2*opt.dimz]);
                # Check if there is any data conditioning event or not and find the
                # pattern to be pasted on the simulation grid
                random.seed(10)
                if status == 'empty':
                    # use pre-determined numbers 
                    #randIdx    = randomPatches[ix]
                    #ix         = ix + 1
                    randIdx    = np.ceil(np.multiply(X.shape[0], np.random.uniform()))
                    #Pattern    = X[randIdx,:]
                    Pattern    = X[int(randIdx-1),:]
                    if m1 == 1:
                        Pt_CM      = CMM[int(randIdx-1),:]
                elif status == 'some':
                    #dataLoc=[wx[i,1], wy[i,1]]
                    # calculate d_pat and d_loc and d_ns
                    idxNumber = findClosestPattern_Non(dataEvent[0,0:opt.pat**2*opt.dimz], X[:,0:opt.pat**2*opt.dimz])
                
                    Pattern    = X[int(idxNumber),:]
                    if m1 == 1:
                        Pt_CM      = CMM[int(idxNumber),:]
                elif status == 'full':
                    if existNonFrozenNodes(frozenRealiz, wx[i,:], wy[i,:], wz[i,:]):
                            #dataLoc=[wx[i,1], wy[i,1]]
                            # calculate d_pat and d_loc and d_ns
                            idxNumber = findClosestPattern_Non(dataEvent[0,0:opt.pat**2*opt.dimz], X[:,0:opt.pat**2*opt.dimz])
                            
                            Pattern    = X[int(idxNumber),:]
                            if m1 == 1:
                                #Pt_CM      = CMM[int(idxNumber),:]
                                Pt_CM      = CMM[1,:]

                    else:
                        continue
                    
                
                
       
                if m1 == 1:
                    coherenceMap = pastePattern2(Pt_CM, wx[i,:], wy[i,:], wz[i,:], coherenceMap, frozenRealiz, opt)
                # Paste the pattern on simulation grid and updates frozen nodes
                [realization, frozenRealiz] = pastePattern(Pattern, wx[i,:], wy[i,:], wz[i,:], realization, frozenRealiz, opt)
                
                
    # crop the realization to its true dimensions
    limitsx  = (opt.szRealizationx  -opt.dimx )//2
    limitsy  = (opt.szRealizationy  -opt.dimy )//2
    realization = realization[limitsx:limitsx+opt.dimx , limitsy:limitsy+opt.dimy , :]
    if m1 == 1:
        coherenceMap = coherenceMap[limitsx:limitsx+opt.dimx , limitsy:limitsy+opt.dimy , :]
    return realization, coherenceMap