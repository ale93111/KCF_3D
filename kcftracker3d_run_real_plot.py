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

path = '/home/alessandro/sohpc/'
#%%
rec001 = np.load(path+"real/rec001.npy")
rec002 = np.load(path+"real/rec002.npy")
rec003 = np.load(path+"real/rec003.npy")
#%%
bbox = np.array([180.0, 295.0, 238.0, 29.0, 29.0, 24.0],dtype=np.int64)
#bbox = np.array([180.0, 295.0, 238.0, 29.0, 29.0, 24.0],dtype=np.float64)
#bbox = bbox/2
#bbox = np.array(bbox,dtype=np.int64)

print(bbox)
#%%
plt.imshow(rec001[bbox[2]+bbox[5]//2,bbox[1]:bbox[1]+bbox[4],bbox[0]:bbox[0]+bbox[3]])
#%%
tracker = kcftracker3d_numpy.KCFTracker(False, False, False)
#tracker.lambdar = 0.0001   # regularization
tracker.padding = 4.5   # extra area surrounding the target
tracker.output_sigma_factor = 0.125   # bandwidth of gaussian target
tracker.sigma = 0.1
tracker.interp_factor = 0.0
tracker.init(bbox, rec001)
boxlist = [bbox]
#%%
for i in range(3):
    bbox = tracker.update(rec002)
    print(list(map(int,bbox)))
#%%
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
#output = np.zeros(tuple([3] + list(rec001.shape)),dtype=np.uint8)
#%%
rec001 = drawbox(rec001,boxlist[0],3)
#%%
rec002 = drawbox(rec002,boxlist[1],3)
#%%
rec003 = drawbox(rec003,boxlist[2],3)
#%%
plt.imshow(rec001[boxlist[0][2]+boxlist[0][5]//2])
#%%
plt.imshow(rec002[boxlist[1][2]+boxlist[1][5]//2])
#%%
plt.imshow(rec003[boxlist[2][2]+boxlist[2][5]//2])
#%%
plt.imsave("./rec001_2.png",rec001[boxlist[0][2]+boxlist[0][5]//2])
plt.imsave("./rec002_2.png",rec002[boxlist[1][2]+boxlist[1][5]//2])
plt.imsave("./rec003_2.png",rec003[boxlist[2][2]+boxlist[2][5]//2])
#%%

plt.imshow(rec001.T[boxlist[0][0]+boxlist[0][3]//2].T)
#%%
plt.imshow(rec002.T[boxlist[1][0]+boxlist[1][3]//2].T)
#%%
plt.imshow(rec003.T[boxlist[2][0]+boxlist[2][3]//2].T)
#%%
plt.imsave("./rec001_0.png",rec001.T[boxlist[0][0]+boxlist[0][3]//2].T)
plt.imsave("./rec002_0.png",rec002.T[boxlist[1][0]+boxlist[1][3]//2].T)
plt.imsave("./rec003_0.png",rec003.T[boxlist[2][0]+boxlist[2][3]//2].T)
#%%
plt.imshow(np.transpose(rec001,axes=(1,2,0))[boxlist[0][1]+boxlist[0][4]//2].T)
#%%
plt.imshow(np.transpose(rec002,axes=(1,2,0))[boxlist[1][1]+boxlist[1][4]//2].T)
#%%
plt.imshow(np.transpose(rec003,axes=(1,2,0))[boxlist[2][1]+boxlist[2][4]//2].T)
#%%
plt.imsave("./rec001_1.png",np.transpose(rec001,axes=(1,2,0))[boxlist[0][1]+boxlist[0][4]//2].T)
plt.imsave("./rec002_1.png",np.transpose(rec002,axes=(1,2,0))[boxlist[1][1]+boxlist[1][4]//2].T)
plt.imsave("./rec003_1.png",np.transpose(rec003,axes=(1,2,0))[boxlist[2][1]+boxlist[2][4]//2].T)