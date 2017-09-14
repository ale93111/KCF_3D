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

filenames = ["./rec013_ims.npy",
             "./rec014_ims.npy",
             "./rec015_ims.npy",
             "./rec016_ims.npy",
             "./rec017_ims.npy"]

#%%
rec001 = np.load(filenames[0])

#%%
bbox = np.array([288.0, 311.0, 498.0, 40.0, 40.0, 38.0],dtype=np.int64)
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
tracker.padding = 2.5   # extra area surrounding the target
tracker.output_sigma_factor = 0.125   # bandwidth of gaussian target
tracker.sigma = 0.05
#tracker.interp_factor = 0.0
tracker.init(bbox, rec001)
boxlist = [bbox]
#%%
draw_xyz = False
n_iter = 5
for j,filename in enumerate(filenames):
    print("Loading data...")
    current_rec = np.load(filename)
    print("Tracking frame "+str(j).zfill(2)+"...")
    
    if j:
        for i in range(n_iter):
            bbox = tracker.update(current_rec)
            print(bbox)
    
    bb = list(map(int,bbox))
    print(bb)
    boxlist.append(bb)
    
    plt.imshow(current_rec[bb[2]+bb[5]//2,bb[1]:bb[1]+bb[4],bb[0]:bb[0]+bb[3]])
    plt.show()
    
    if(draw_xyz):
        frame = drawbox(current_rec,bb,3)
    
        plt.imsave("./out/z_rec"+str(j).zfill(2)+".png",frame[bb[2]+bb[5]//2])
        plt.imsave("./out/x_rec"+str(j).zfill(2)+".png",frame.T[bb[2]+bb[5]//2].T)
        plt.imsave("./out/y_rec"+str(j).zfill(2)+".png",np.transpose(frame,axes=(1,2,0))[bb[2]+bb[5]//2].T)
    
    #del current_rec
#%%