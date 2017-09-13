#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 20:06:27 2017

@author: alessandro
"""

import kcftracker3d_numpy
from scipy import ndimage
from scipy import signal
import numpy as np
import gputools as gp
import matplotlib.pyplot as plt

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

def make_sphere(radius = 5):
    r2 = np.arange(-radius, radius+1)**2
    dist2 = r2[:, None, None] + r2[:, None] + r2
    volume = (dist2 <= radius**2).astype(np.int)
    return volume
#%%
rad = 8
of = 16

shape = (64,64,64*2)
data = np.zeros(shape)
sphere = make_sphere(rad)
data[of:of+sphere.shape[0],of:of+sphere.shape[0],of:of+sphere.shape[0]] = sphere
data = (data+0.1)/1.1
noise = gp.perlin3(size=shape[::-1], scale=(10.,10.,10.))
data = 0.75*data + 0.25*data*noise
data = np.array(255*data,dtype=np.uint8)

bbox = np.array([of, of, of, 2*rad, 2*rad, 2*rad],dtype=np.int64)
print(bbox)
#%%
old_data = data.copy()
tracker = kcftracker3d_numpy.KCFTracker(False, False, False)
#tracker.padding = 2.5   # extra area surrounding the target
tracker.output_sigma_factor = 0.125   # bandwidth of gaussian target
#tracker.interp_factor = 0.075
tracker.sigma = 0.2
        
tracker.init(bbox, old_data)
#%%
for i in range(0,20,1):
    print(i)
    data = ndimage.shift(old_data,(signal.square(2*np.pi*0.05*i),0,1),order=1)
    print("data generated...")
    
    bbox = tracker.update(data)
    
    frame = drawbox(data,list(map(int,bbox)))
    plt.imshow(np.transpose(frame,axes=(1,2,0))[24])
    plt.show()
    #print("saving frame..")
    #frame = drawbox(data,bbox_new,width=1)
    #np.save(path+"test/frame"+str(i),frame)

    old_data = data
    
    print(list(map(int,bbox)))
