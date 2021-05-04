# -*- coding: utf-8 -*-
"""
Created on Tue May  4 23:14:50 2021

@author: Allan
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

thresh=0.5

def get_hsv_mask(img):
    thresh_low=np.array([0, 50,0],dtype=np.uint8)
    thresh_up=np.array([150, 150,255],dtype=np.uint8)
    img_hsv=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    mask_hsv=cv2.inRange(img_hsv,thresh_low,thresh_up)
    
    mask_hsv[mask_hsv<128]=0
    mask_hsv[mask_hsv>=128]=1
    
    return mask_hsv.astype(float)

def get_rgb_mask(img):
    thresh_low=np.array([45,52,108],dtype=np.uint8)
    thresh_up=np.array([255,255,255],dtype=np.uint8)
    
    mask_a=cv2.inRange(img,thresh_low,thresh_up)
    mask_b=255*((img[:,:,2]-img[:,:,1])/20)
    mask_c=255*(((np.max(img,axis=2))-(np.min(img,axis=2)))/20)
    mask_d=np.bitwise_and(np.uint64(mask_a),np.uint64(mask_b))
    mask_rgb=np.bitwise_and(np.uint64(mask_c),np.uint64(mask_d))
    
    mask_rgb[mask_rgb<128]=0
    mask_rgb[mask_rgb>=128]=1
    
    return mask_rgb.astype(float)

def get_ycrcb_mask(img):
    thresh_low=np.array([90,100,130],dtype=np.uint8)
    thresh_up=np.array([230,120,180],dtype=np.uint8)
    
    img_ycrcb=cv2.cvtColor(img,cv2.COLOR_RGB2YCR_CB)
    mask_ycrcb=cv2.inRange(img_ycrcb,thresh_low,thresh_up)
    
    mask_ycrcb[mask_ycrcb<128]=0
    mask_ycrcb[mask_ycrcb>=128]=1
    
    return mask_ycrcb.astype(float)

def morph(mask):
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
    
    return mask

def cut_mask(mask):
    kernel=np.ones([50,50],np.float32)/(50*50)
    filt=cv2.filter2D(mask,-1,kernel)
    filt[filt!=0]=255
    
    free=np.array(cv2.bitwise_not(filt),dtype=np.uint8)
    
    grab_mask=np.zeros(mask.shape,dtype=np.uint8)
    grab_mask[:,:]=2
    grab_mask[mask==255]=1
    grab_mask[free==255]=0
    
    if np.unique(grab_mask).tolist==[0,1]:
        backgnd=np.zeros((1,65),np.float64)
        foregnd=np.zeros((1,65),np.float64)
        
        if mask.size!=0:
            mask,backgnd,foregnd=cv2.grabCut(mask,grab_mask,None,backgnd,foregnd,5,cv2.GC_INIT_WITH_MASK)
            mask=np.where((mask==2)|(mask==0),0,1).astype(np.uint8)
    return mask

def process(img):
    mask_hsv=get_hsv_mask(img)
    mask_rgb=get_rgb_mask(img)
    mask_ycrcb = get_ycrcb_mask(img)

    no_masks=3.0

    mask=(mask_hsv+mask_rgb+mask_ycrcb)/no_masks

    mask[mask<thresh]=0.0
    mask[mask>=thresh]=255.0

    mask=mask.astype(np.uint8)

    mask=morph(mask) #To remove noisy pixels

    mask=cut_mask(mask)
    
    return mask

