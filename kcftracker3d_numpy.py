#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 21:50:09 2017

@author: alessandro
"""

from scipy.ndimage.interpolation import zoom
import numpy as np 

# ffttools
def fftd(img):    
    return np.fft.fftn(img)

def ifftd(img):    
    return np.fft.ifftn(img)

def rearrange(img):
    return np.fft.fftshift(img, axes=(0,1,2))

# recttools
def x2(rect):
    return rect[0] + rect[3]

def y2(rect):
    return rect[1] + rect[4]

def z2(rect):
    return rect[2] + rect[5]

def limit(rect, limit):
    if(rect[0]+rect[3] > limit[0]+limit[3]):
        rect[3] = limit[0]+limit[3]-rect[0]
    if(rect[1]+rect[4] > limit[1]+limit[4]):
        rect[4] = limit[1]+limit[4]-rect[1]
    if(rect[2]+rect[5] > limit[2]+limit[5]):
        rect[5] = limit[2]+limit[5]-rect[2]
        
    if(rect[0] < limit[0]):
        rect[3] -= (limit[0]-rect[0])
        rect[0] = limit[0]
    if(rect[1] < limit[1]):
        rect[4] -= (limit[1]-rect[1])
        rect[1] = limit[1]
    if(rect[2] < limit[2]):
        rect[5] -= (limit[2]-rect[2])
        rect[2] = limit[2]
    
    if(rect[3] < 0):
        rect[3] = 0
    if(rect[4] < 0):
        rect[4] = 0
    if(rect[5] < 0):
        rect[5] = 0
    return rect

def getBorder(original, limited):
    res = [0,0,0,0,0,0]
    res[0] = limited[0] - original[0]
    res[1] = limited[1] - original[1]
    res[2] = limited[2] - original[2]
    res[3] = x2(original) - x2(limited)
    res[4] = y2(original) - y2(limited)
    res[5] = z2(original) - z2(limited)
    assert(np.all(np.array(res) >= 0))
    return res

def subwindow(img, window, mode='edge'): #mode='reflect'
    cutWindow = [x for x in window]
    limit(cutWindow, [0,0,0,img.shape[2],img.shape[1],img.shape[0]])   # modify cutWindow
    assert(cutWindow[3]>0 and cutWindow[4]>0 and cutWindow[5]>0)
    border = getBorder(window, cutWindow)
    res = img[cutWindow[2]:cutWindow[2]+cutWindow[5], 
              cutWindow[1]:cutWindow[1]+cutWindow[4], 
              cutWindow[0]:cutWindow[0]+cutWindow[3]]

    if(border != [0,0,0,0,0,0]):
        #res = cv2.copyMakeBorder(res, border[1], border[3], border[0], border[2], borderType)
        res = np.pad(res,[(border[2],border[5]),
                          (border[1],border[4]),
                          (border[0],border[3])],mode='edge')
    return res

# KCF tracker
class KCFTracker:
    def __init__(self, hog=False, fixed_window=True, multiscale=False):
        self.lambdar = 0.0001   # regularization
        self.padding = 2.5   # extra area surrounding the target
        self.output_sigma_factor = 0.625   # bandwidth of gaussian target

        self.interp_factor = 0.075
        self.sigma = 0.4
        self.cell_size = 1

        if(multiscale):
            self.template_size = 96   # template size
            self.scale_step = 1.05   # scale step for multi-scale estimation
            self.scale_weight = 0.96   # to downweight detection scores of other scales for added stability
        elif(fixed_window):
            self.template_size = 96
            self.scale_step = 1
        else:
            self.template_size = 1
            self.scale_step = 1

        self._tmpl_sz = [0,0,0]  
        self._roi = [0.,0.,0.,0.,0.,0.] 
        self.size_patch = [0,0,0]  
        self._scale = 1.   
        self._alphaf = None  
        self._prob = None  
        self._tmpl = None  
        self.hann = None  

    def subPixelPeak(self, left, center, right):
        divisor = 2*center - right - left   #float
        return (0 if abs(divisor)<1e-3 else 0.5*(right-left)/float(divisor))

    def createHanningMats(self):
        hann3t, hann2t, hann1t = np.ogrid[0:self.size_patch[0],
                                          0:self.size_patch[1],
                                          0:self.size_patch[2]]

        hann1t = 0.5 * (1 - np.cos(2*np.pi*hann1t/(self.size_patch[2]-1.)))
        hann2t = 0.5 * (1 - np.cos(2*np.pi*hann2t/(self.size_patch[1]-1.)))
        hann3t = 0.5 * (1 - np.cos(2*np.pi*hann3t/(self.size_patch[0]-1.)))
        hann3d = hann3t * hann2t * hann1t

        self.hann = hann3d
        self.hann = self.hann.astype(np.float32)

    def createGaussianPeak(self, sizez, sizey, sizex):
        szh, syh, sxh = sizez/2., sizey/2., sizex/2.
        output_sigma = np.sqrt(sizex*sizey*sizez) / self.padding * self.output_sigma_factor
        mult = -0.5 / (output_sigma*output_sigma)
        z, y, x = np.ogrid[0:sizez, 0:sizey, 0:sizex]
        z, y, x = (z-szh)**2, (y-syh)**2, (x-sxh)**2
        res = np.exp(mult * (z+y+x))
        return res

    def gaussianCorrelation(self, x1, x2):
        c = fftd(x1)*np.conj(fftd(x2))
        c = ifftd(c)
        c = np.real(c)
        c = rearrange(c)

        d = (np.sum(x1*x1) + np.sum(x2*x2) - 2.0*c) / float(self.size_patch[0]*self.size_patch[1])#*self.size_patch[2])

        d = d * (d>=0)
        d = np.exp(-d / (self.sigma*self.sigma))

        return d

    def getFeatures(self, image, inithann, scale_adjust=1.0):
        extracted_roi = [0,0,0,0,0,0]   #[int,int,int,int]
        cx = self._roi[0] + self._roi[3]/2.  #float
        cy = self._roi[1] + self._roi[4]/2.  #float
        cz = self._roi[2] + self._roi[5]/2.  #float

        if(inithann):
            padded_w = self._roi[3] * self.padding
            padded_h = self._roi[4] * self.padding
            padded_d = self._roi[5] * self.padding

            if(self.template_size > 1):
                #if(padded_w >= padded_h):
                #    self._scale = padded_w / float(self.template_size)
                #else:
                #    self._scale = padded_h / float(self.template_size)
                self._scale = min(padded_w,
                                  padded_h,
                                  padded_d) / float(self.template_size)
                self._tmpl_sz[0] = int(padded_w / self._scale)
                self._tmpl_sz[1] = int(padded_h / self._scale)
                self._tmpl_sz[2] = int(padded_d / self._scale)
            else:
                self._tmpl_sz[0] = int(padded_w)
                self._tmpl_sz[1] = int(padded_h)
                self._tmpl_sz[2] = int(padded_d)
                self._scale = 1.

            self._tmpl_sz[0] = int(self._tmpl_sz[0])
            self._tmpl_sz[1] = int(self._tmpl_sz[1])
            self._tmpl_sz[2] = int(self._tmpl_sz[2])

        extracted_roi[3] = int(scale_adjust * self._scale * self._tmpl_sz[0])
        extracted_roi[4] = int(scale_adjust * self._scale * self._tmpl_sz[1])
        extracted_roi[5] = int(scale_adjust * self._scale * self._tmpl_sz[2])
        extracted_roi[0] = int(cx - extracted_roi[3]/2.)
        extracted_roi[1] = int(cy - extracted_roi[4]/2.)
        extracted_roi[2] = int(cz - extracted_roi[5]/2.)

        z = subwindow(image, extracted_roi)
        if(z.shape[2]!=self._tmpl_sz[0] or z.shape[1]!=self._tmpl_sz[1] or z.shape[0]!=self._tmpl_sz[2]):
            #z = cv2.resize(z, tuple(self._tmpl_sz))
            z = zoom(z,(self._tmpl_sz[0]/z.shape[2],
                        self._tmpl_sz[1]/z.shape[1],
                        self._tmpl_sz[2]/z.shape[0]))
            self._tmpl_sz[0] = z.shape[2]
            self._tmpl_sz[1] = z.shape[1]
            self._tmpl_sz[2] = z.shape[0]
            print("zoooom")

        FeaturesMap = z   #(size_patch[0], size_patch[1]) #np.int8  #0~255
        FeaturesMap = FeaturesMap.astype(np.float32) / 255.0 - 0.5
        self.size_patch = [z.shape[0], z.shape[1], z.shape[2]]

        if(inithann):
            self.createHanningMats()  # createHanningMats need size_patch

        FeaturesMap = self.hann * FeaturesMap
        return FeaturesMap

    def detect(self, z, x):
        k = self.gaussianCorrelation(x, z)
        res = np.real(ifftd(self._alphaf*fftd(k)))

        pv = np.max(res)
        pi = list(np.unravel_index(res.argmax(), res.shape))[::-1]
        p = [float(pi[0]), float(pi[1]), float(pi[2])]

        if(pi[0]>0 and pi[0]<res.shape[2]-1):
            p[0] += self.subPixelPeak(res[pi[2]  ,pi[1]  ,pi[0]-1], pv, res[pi[2]  ,pi[1]  ,pi[0]+1])
        if(pi[1]>0 and pi[1]<res.shape[1]-1):
            p[1] += self.subPixelPeak(res[pi[2]  ,pi[1]-1,pi[0]  ], pv, res[pi[2]  ,pi[1]+1,pi[0]  ])
        if(pi[2]>0 and pi[2]<res.shape[0]-1):
            p[2] += self.subPixelPeak(res[pi[2]-1,pi[1]  ,pi[0]  ], pv, res[pi[2]+1,pi[1]  ,pi[0]  ])

        p[0] -= res.shape[2] / 2.
        p[1] -= res.shape[1] / 2.
        p[2] -= res.shape[0] / 2.

        return p, pv

    def train(self, x, train_interp_factor):
        k = self.gaussianCorrelation(x, x)
        alphaf = fftd(self._prob)/( fftd(k) + self.lambdar)

        self._tmpl = (1-train_interp_factor)*self._tmpl + train_interp_factor*x
        self._alphaf = (1-train_interp_factor)*self._alphaf + train_interp_factor*alphaf


    def init(self, roi, image):
        self._roi = list(map(float, roi))
        assert(roi[3]>0 and roi[4]>0 and roi[5]>0)
        self._tmpl = self.getFeatures(image, 1)
        self._prob = self.createGaussianPeak(self.size_patch[0], self.size_patch[1], self.size_patch[2])
        self._alphaf = np.zeros((self.size_patch[0], self.size_patch[1], self.size_patch[2]), np.complex128)
        self.train(self._tmpl, 1.0)

    def update(self, image):
        if(self._roi[0]+self._roi[3] <= 0):  self._roi[0] = -self._roi[3] + 1
        if(self._roi[1]+self._roi[4] <= 0):  self._roi[1] = -self._roi[4] + 1
        if(self._roi[2]+self._roi[5] <= 0):  self._roi[2] = -self._roi[5] + 1
        if(self._roi[0] >= image.shape[2]-1):  self._roi[0] = image.shape[2] - 2
        if(self._roi[1] >= image.shape[1]-1):  self._roi[1] = image.shape[1] - 2
        if(self._roi[2] >= image.shape[0]-1):  self._roi[2] = image.shape[0] - 2

        cx = self._roi[0] + self._roi[3]/2.
        cy = self._roi[1] + self._roi[4]/2.
        cz = self._roi[2] + self._roi[5]/2.

        loc, peak_value = self.detect(self._tmpl, self.getFeatures(image, 0, 1.0))

        if(self.scale_step != 1):
            # Test at a smaller _scale
            new_loc1, new_peak_value1 = self.detect(self._tmpl, self.getFeatures(image, 0, 1.0/self.scale_step))
            # Test at a bigger _scale
            new_loc2, new_peak_value2 = self.detect(self._tmpl, self.getFeatures(image, 0, self.scale_step))

            if(self.scale_weight*new_peak_value1 > peak_value and new_peak_value1>new_peak_value2):
                loc = new_loc1
                peak_value = new_peak_value1
                self._scale /= self.scale_step
                self._roi[3] /= self.scale_step
                self._roi[4] /= self.scale_step
                self._roi[5] /= self.scale_step
            elif(self.scale_weight*new_peak_value2 > peak_value):
                loc = new_loc2
                peak_value = new_peak_value2
                self._scale *= self.scale_step
                self._roi[3] *= self.scale_step
                self._roi[4] *= self.scale_step
                self._roi[5] *= self.scale_step
        
        self._roi[0] = cx - self._roi[3]/2.0 + loc[0]*self.cell_size*self._scale
        self._roi[1] = cy - self._roi[4]/2.0 + loc[1]*self.cell_size*self._scale
        self._roi[2] = cz - self._roi[5]/2.0 + loc[2]*self.cell_size*self._scale
        
        if(self._roi[0] >= image.shape[2]-1):  self._roi[0] = image.shape[2] - 1
        if(self._roi[1] >= image.shape[1]-1):  self._roi[1] = image.shape[1] - 1
        if(self._roi[2] >= image.shape[0]-1):  self._roi[2] = image.shape[0] - 1
        if(self._roi[0]+self._roi[3] <= 0):  self._roi[0] = -self._roi[3] + 2
        if(self._roi[1]+self._roi[4] <= 0):  self._roi[1] = -self._roi[4] + 2
        if(self._roi[2]+self._roi[5] <= 0):  self._roi[2] = -self._roi[5] + 2
        assert(self._roi[3]>0 and self._roi[4]>0 and self._roi[5]>0)

        x = self.getFeatures(image, 0, 1.0)
        self.train(x, self.interp_factor)

        return self._roi
