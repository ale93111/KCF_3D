#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 20:06:27 2017

@author: alessandro
"""

import matplotlib.pyplot as plt
import kcftracker3d_numpy
import numpy as np

def drawbox(volume,bbox,width=1):
    res = volume.copy()
    for k in range(width):
        for j in range(width):
            for i in range(width):
                res[bbox[2]                +k,bbox[1]:bbox[1]+bbox[4]+j,bbox[0]:bbox[0]+bbox[3]+i] = 255
                res[bbox[2]+bbox[5]        +k,bbox[1]:bbox[1]+bbox[4]+j,bbox[0]:bbox[0]+bbox[3]+i] = 255
                res[bbox[2]:bbox[2]+bbox[5]+k,bbox[1]:bbox[1]+bbox[4]+j,bbox[0]                +i] = 255
                res[bbox[2]:bbox[2]+bbox[5]+k,bbox[1]:bbox[1]+bbox[4]+j,bbox[0]+bbox[3]        +i] = 255
                res[bbox[2]:bbox[2]+bbox[5]+k,bbox[1]                +j,bbox[0]:bbox[0]+bbox[3]+i] = 255
                res[bbox[2]:bbox[2]+bbox[5]+k,bbox[1]+bbox[4]        +j,bbox[0]:bbox[0]+bbox[3]+i] = 255
                
    return res

path = ''#'/home/alessandro/sohpc/'
#%%
rec001 = np.load(path+"./rec013_ims.npy")
rec002 = np.load(path+"./rec014_ims.npy")
rec003 = np.load(path+"./rec015_ims.npy")
rec004 = np.load(path+"./rec016_ims.npy")
rec005 = np.load(path+"./rec017_ims.npy")
#rec003 = np.load(path+"real/rec003.npy")
#%%
bbox = np.array([288.0, 311.0, 498.0, 40.0, 40.0, 39.0],dtype=np.int64)
#bbox = np.array([288.0, 311.0, 478.0, 40.0, 40.0, 39.0],dtype=np.int64)
#bbox = np.array([258.0, 288.0, 239.0, 40.0, 40.0, 26.0],dtype=np.int64)
#bbox = np.array([180.0, 295.0, 238.0, 29.0, 29.0, 24.0],dtype=np.int64)
#bbox = np.array([180.0, 295.0, 238.0, 29.0, 29.0, 24.0],dtype=np.float64)
#bbox = bbox/2
#bbox = np.array(bbox,dtype=np.int64)

print(bbox)

plt.imshow(rec001[bbox[2]+bbox[5]//2,bbox[1]:bbox[1]+bbox[4],bbox[0]:bbox[0]+bbox[3]])
#%%
tracker = kcftracker3d_numpy.KCFTracker(False, False, False)
#tracker.lambdar = 0.0001   # regularization
tracker.padding = 3.5   # extra area surrounding the target
tracker.output_sigma_factor = 0.025   # bandwidth of gaussian target
tracker.sigma = 0.2
tracker.interp_factor = 0.0
tracker.init(bbox, rec001)
boxlist = [bbox]
#%%
for i in range(3):
    bbox = tracker.update(rec002)
    print(list(map(int,bbox)))

boxlist.append(list(map(int,bbox)))
#%%
bb = list(map(int,bbox))
plt.imshow(rec002[bb[2]+bb[5]//2,bb[1]:bb[1]+bb[4],bb[0]:bb[0]+bb[3]])
#%%
for i in range(3):
    bbox  = tracker.update(rec003)
    print(list(map(int,bbox)))
boxlist.append(list(map(int,bbox)))
#%%
bb = list(map(int,bbox))
plt.imshow(rec003[bb[2]+bb[5]//2,bb[1]:bb[1]+bb[4],bb[0]:bb[0]+bb[3]])
#%%
for i in range(3):
    bbox  = tracker.update(rec004)
    print(list(map(int,bbox)))
boxlist.append(list(map(int,bbox)))
#%%
bb = list(map(int,bbox))
plt.imshow(rec004[bb[2]+bb[5]//2,bb[1]:bb[1]+bb[4],bb[0]:bb[0]+bb[3]])
#%%
for i in range(3):
    bbox  = tracker.update(rec005)
    print(list(map(int,bbox)))
boxlist.append(list(map(int,bbox)))
#%%
bb = list(map(int,bbox))
plt.imshow(rec005[bb[2]+bb[5]//2,bb[1]:bb[1]+bb[4],bb[0]:bb[0]+bb[3]])
#%%
#output = np.zeros(tuple([3] + list(rec001.shape)),dtype=np.uint8)
#%%
rec001 = drawbox(rec001,boxlist[0],3)
#%%
rec002 = drawbox(rec002,boxlist[1],3)
#%%
rec003 = drawbox(rec003,boxlist[2],3)
#%%
rec004 = drawbox(rec004,boxlist[3],3)
#%%
rec005 = drawbox(rec005,boxlist[4],3)
#%%
plt.imshow(rec001[boxlist[0][2]+boxlist[0][5]//2])
#%%
plt.imshow(rec002[boxlist[1][2]+boxlist[1][5]//2])
#%%
plt.imshow(rec003[boxlist[2][2]+boxlist[2][5]//2])
#%%
plt.imshow(rec004[boxlist[3][2]+boxlist[3][5]//2])
#%%
plt.imshow(rec005[boxlist[4][2]+boxlist[4][5]//2])

#%%
plt.imsave("./out/z_rec001.png",rec001[boxlist[0][2]+boxlist[0][5]//2])
plt.imsave("./out/z_rec002.png",rec002[boxlist[1][2]+boxlist[1][5]//2])
plt.imsave("./out/z_rec003.png",rec003[boxlist[2][2]+boxlist[2][5]//2])
plt.imsave("./out/z_rec004.png",rec004[boxlist[3][2]+boxlist[3][5]//2])
plt.imsave("./out/z_rec005.png",rec005[boxlist[4][2]+boxlist[4][5]//2])
#%%
plt.imshow(rec001.T[boxlist[0][0]+boxlist[0][3]//2].T)
#%%
plt.imshow(rec002.T[boxlist[1][0]+boxlist[1][3]//2].T)
#%%
plt.imshow(rec003.T[boxlist[2][0]+boxlist[2][3]//2].T)
#%%
plt.imshow(rec004.T[boxlist[3][0]+boxlist[3][3]//2].T)
#%%
plt.imshow(rec005.T[boxlist[4][0]+boxlist[4][3]//2].T)

#%%
plt.imsave("./out/x_rec001.png",rec001.T[boxlist[0][0]+boxlist[0][3]//2].T)
plt.imsave("./out/x_rec002.png",rec002.T[boxlist[1][0]+boxlist[1][3]//2].T)
plt.imsave("./out/x_rec003.png",rec003.T[boxlist[2][0]+boxlist[2][3]//2].T)
plt.imsave("./out/x_rec004.png",rec004.T[boxlist[3][0]+boxlist[3][3]//2].T)
plt.imsave("./out/x_rec005.png",rec005.T[boxlist[4][0]+boxlist[4][3]//2].T)
#%%
plt.imshow(np.transpose(rec001,axes=(1,2,0))[boxlist[0][1]+boxlist[0][4]//2].T)
#%%
plt.imshow(np.transpose(rec002,axes=(1,2,0))[boxlist[1][1]+boxlist[1][4]//2].T)
#%%
plt.imshow(np.transpose(rec003,axes=(1,2,0))[boxlist[2][1]+boxlist[2][4]//2].T)
#%%
plt.imshow(np.transpose(rec004,axes=(1,2,0))[boxlist[3][1]+boxlist[3][4]//2].T)
#%%
plt.imshow(np.transpose(rec005,axes=(1,2,0))[boxlist[4][1]+boxlist[4][4]//2].T)

#%%
plt.imsave("./out/y_rec001.png",np.transpose(rec001,axes=(1,2,0))[boxlist[0][1]+boxlist[0][4]//2].T)
plt.imsave("./out/y_rec002.png",np.transpose(rec002,axes=(1,2,0))[boxlist[1][1]+boxlist[1][4]//2].T)
plt.imsave("./out/y_rec003.png",np.transpose(rec003,axes=(1,2,0))[boxlist[2][1]+boxlist[2][4]//2].T)
plt.imsave("./out/y_rec004.png",np.transpose(rec004,axes=(1,2,0))[boxlist[3][1]+boxlist[3][4]//2].T)
plt.imsave("./out/y_rec005.png",np.transpose(rec005,axes=(1,2,0))[boxlist[4][1]+boxlist[4][4]//2].T)