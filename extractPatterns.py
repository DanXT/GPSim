# -*- coding: utf-8 -*-
"""

@author: xtan
"""



import numpy as np
import numpy.matlib as npm

def extractPatterns(out, pat, m):
    Dimx, Dimy = out.shape
    # Number of patterns : disDim*disDim
    
    disDimx  = np.int64(np.ceil((Dimx  - (1+(pat -1)) + 1)/m))
    disDimy  = np.int64(np.ceil((Dimy  - (1+(pat -1)) + 1)/m))
    print('\n\nNext Phase\n------------------------------------------------\n')
    print('Number of Patterns : %d x %d x %d\n', disDimx,disDimy)


    X=np.zeros([disDimx*disDimy,pat**2])
    a=0
    for i in range(disDimx):
        for j in range(disDimy):
            #wx = list(range(m*i, m*i+pat))
            #wy = list(range(m*j, m*j+pat))
            wx1 = m*i
            wx2 = m*i+pat
            wy1 = m*j
            wy2 = m*j+pat
            aaa = out[wx1:wx2,wy1:wy2]
            bbb = np.reshape(aaa, (1,pat**2))
            X[a,:]=bbb
                     
            a=a+1
          
           

    return X


def extractPatterns3D(out, pat, m):
    Dimx, Dimy, Dimz = out.shape
    # Number of patterns : disDim*disDim
    
    disDimx  = np.int64(np.ceil((Dimx  - (1+(pat -1)) + 1)/m))
    disDimy  = np.int64(np.ceil((Dimy  - (1+(pat -1)) + 1)/m))
    print('\n\nNext Phase\n------------------------------------------------\n')
    print('Number of Patterns : {:5d} x {:5d}\n'.format(disDimx,disDimy))


    X=np.zeros([disDimx*disDimy,pat**2*Dimz])
    a=0
    for i in range(disDimx):
        for j in range(disDimy):
            #wx = list(range(m*i, m*i+pat))
            #wy = list(range(m*j, m*j+pat))
            wx1 = m*i
            wx2 = m*i+pat
            wy1 = m*j
            wy2 = m*j+pat
            aaa = out[wx1:wx2,wy1:wy2,:]
            bbb = np.reshape(aaa, (1,pat**2*Dimz))
            X[a,:]=bbb
                     
            a=a+1
          
           

    return X



def ind2sub(array_shape, ind):
    # Gives repeated indices, replicates matlabs ind2sub
    rows = (ind.astype("int32") // array_shape[1])
    cols = (ind.astype("int32") % array_shape[1])
    return (rows, cols)



def getDataEvent(realization, wx, wy, wz):
    pat = len(wx)
    patz = len(wz)
    
    dataEvent = realization[wx[0]:wx[-1]+1,wy[0]:wy[-1]+1,wz[0]:wz[-1]+1]
    dataEvent = dataEvent.flatten()
    
    diff = sum(dataEvent != 0.5)
    
    if diff == 0:
        status = 'empty'
    elif diff == pat**2*patz:
        status = 'full'
    else:
        status = 'some'
        
    dataEvent = np.expand_dims(dataEvent, axis=0)

        
    return dataEvent, status



def findClosestPattern_Non(dataEvent, X):
    index1 = range(X.shape[0])
#    index_hardData= np.where(weightEvent == 0.5)[0]
#    if len(index_hardData) != 0:
#        reducedX_hd = X[:,index_hardData[0]:index_hardData[-1]+1]
#        reducedDataEvent_hd = dataEvent[1,index_hardData[0]:index_hardData[-1]+1]
#        diff1 = 0
#        
#        index1 = np.where(diff1/len(index_hardData) <=0.1)
#        if len(index1)==0:
#            index1=range(X.shape[0])

    index = np.where(dataEvent != 0.5)
    index = np.asarray(index)
    index = np.squeeze(index)
    reducedX = X[index1[0]:index1[-1]+1, index]
    reducedDataEvent = dataEvent[index]
    #weightEvent = weightEvent[1,index]
    
    
    rD = np.expand_dims(reducedDataEvent, axis=0)
    di1 = reducedX - rD
    di1 = np.absolute(di1)
    
    if di1.ndim == 2:
    #if di1.shape[1]!=0:   
        difference1 = np.sum(di1, axis=1)
    else:
        difference1 = di1
    #difference1 = sum(bsxfun(@times,abs(bsxfun(@minus, reducedX, reducedDataEvent)),wieghtEvent), 2);

    #difference3 = pdist2(Locdb(index1,:),dataLoc);

    if sum(difference1)!=0:
    
        d1_normalize=difference1/max(np.absolute(difference1))
    else:
        d1_normalize=0
        


    #d3_normalize=difference3/max(abs(difference3));

    #d=(1-w_ssm)*d1_normalize+w_ssm*d3_normalize;
    d = d1_normalize
    #dummy, idxNum = min(d)

    
    #idxNumber=index1(idxNum)
    idxNumber = np.argmin(d)
    
    
    return idxNumber

def existNonFrozenNodes(frozenRealiz, wx, wy, wz):
    boolOutput = False
    
    frozenTemplate = frozenRealiz[wx[0]:wx[-1]+1,wy[0]:wy[-1]+1,wz[0]:wz[-1]+1]
    
    if np.any(frozenTemplate.flatten()==0):
        boolOutput = True
    
    return boolOutput

def pastePattern(Pattern, wx, wy, wz, realization, frozenRealiz, opt):
    
    DimzAll = opt.dimz
    Pat = opt.pat
    Patz= opt.patz
    innerPatch = opt.innerPatch
    innerPatchz= opt.innerPatchz
    
    tmpp = frozenRealiz[wx[0]:wx[-1]+1, wy[0]:wy[-1]+1, wz]
    
    #frozenTemplate = tmpp.view(1,np.asarray(Pat**2*DimzAll))
    frozenTemplate =  np.reshape(tmpp, (1, Pat**2*DimzAll)) 
    
    
    
    #frozenTemplate      = reshape(frozenRealiz[wx[0]:wx[-1]+1, wy[0]:wy[-1]+1, wz],1,Pat**2*DimzAll)
    realizationTemplate = np.reshape(realization[wx[0]:wx[-1]+1, wy[0]:wy[-1]+1, wz],(1,Pat**2*DimzAll))
    
    Pattern = np.expand_dims(Pattern, axis=0)

    
    Pattern[0,np.squeeze(frozenTemplate == 1)] = realizationTemplate[0,np.squeeze(frozenTemplate == 1)]


    realization[wx[0]:wx[-1]+1, wy[0]:wy[-1]+1, wz] = np.reshape(Pattern, (Pat, Pat, DimzAll))


    middleIdx  = (Pat +1)//2
    middleIdxz = (Patz+1)/2
    boundary   = (innerPatch -1)//2
    boundaryz  = (innerPatchz-1)/2
    wxInner = wx[middleIdx- boundary - 1  : middleIdx +boundary ]
    wyInner = wy[middleIdx- boundary -1  : middleIdx +boundary ]
    wzInner = wz
    frozenRealiz[wxInner[0]:wxInner[-1]+1, wyInner[0]:wyInner[-1]+1, wzInner] = 1

    
    return realization, frozenRealiz

def getPatternShape(node, opt):
    dimAll = opt.dimz
    pat = opt.pat
    node = np.transpose(node)
    
    borders = (pat -1)/2
    
    r = (pat + 1)/2
    i = range(pat)
    
    
    aaa = np.ones([node.shape[0], pat])
    bbb = 1 + (node[:,0] - r) + borders
    
    ccc = np.transpose(aaa)*np.transpose(bbb)
    wx = np.transpose(ccc)
    wx = wx + i
    #wx = bsxfun(@times, ones(size(node,1), Pat),  1 + (node(:,1) -r) + borders);
    #wx = bsxfun(@plus, wx, i);
    
    bbb = 1 + (node[:,1] - r) + borders
    
    ddd = np.transpose(aaa)*np.transpose(bbb)
    wy = np.transpose(ddd)
    wy = wy + i
 
    #wy = bsxfun(@times, ones(size(node,1), Pat),  1 + (node(:,2) -r) + borders);
    #wy = bsxfun(@plus, wy, i);

    wz = range(dimAll)
    wz = np.matlib.repmat(wz,node.shape[0],1);
    
    return wx.astype(int), wy.astype(int), wz.astype(int)


def pastePattern2(Pattern, wx, wy, wz, realization, frozenRealiz, opt):
    
    DimzAll = opt.dimz
    Pat = opt.pat
    Patz= opt.patz
    
    tmpp = frozenRealiz[wx[0]:wx[-1]+1, wy[0]:wy[-1]+1, wz]
    
    #frozenTemplate = tmpp.view(1,np.asarray(Pat**2*DimzAll))
    frozenTemplate =  np.reshape(tmpp, (1, Pat**2*DimzAll)) 
    
    
    
    #frozenTemplate      = reshape(frozenRealiz[wx[0]:wx[-1]+1, wy[0]:wy[-1]+1, wz],1,Pat**2*DimzAll)
    realizationTemplate = np.reshape(realization[wx[0]:wx[-1]+1, wy[0]:wy[-1]+1, wz],(1,Pat**2*DimzAll))
    
    Pattern = np.expand_dims(Pattern, axis=0)

    
    Pattern[0,np.squeeze(frozenTemplate == 1)] = realizationTemplate[0,np.squeeze(frozenTemplate == 1)]


    realization[wx[0]:wx[-1]+1, wy[0]:wy[-1]+1, wz] = np.reshape(Pattern, (Pat, Pat, DimzAll))



    
    return realization