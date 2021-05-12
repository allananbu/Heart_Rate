# -*- coding: utf-8 -*-
"""
Created on Tue May  4 07:12:20 2021

@author: Allan
"""

import numpy as np
import matplotlib.pyplot as plt
import dlib
import cv2
import os
import time

import skin_detection



pathin='videoplayback.mp4'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    
r=[]
g=[]
b=[]
    
nseg=12 #Frequency analysis
    
left_increase_ratio = 0.05 #5%
top_increase_ratio = 0.25 #5%
    
count=0
cam=cv2.VideoCapture(pathin)
fps=cam.get(cv2.CAP_PROP_FPS)
start =0
end=2500

frame_count=0
i=start

start_time=time.time()
H = np.zeros([2500])

while count<end:
    ret,frame=cam.read()
    
    if ret==True:
        count+=1
    else:
        break    
    
    h=frame.shape[0]
    w=frame.shape[1]
    
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    rect=detector(gray,0)
    
    if len(rect)==0:
        continue
    if len(rect)>0:
        rects=rect[0]
        
        left,right,top,bottom=rects.left(),rects.right(),rects.top(),rects.bottom()
        
        width=abs(right-left)
        height=abs(bottom-top)
        
        face_left=int(left - (left_increase_ratio/2)*width)
        face_top=int(top-(top_increase_ratio)*height)
        face_right=right
        face_bottom=bottom
        
        face = frame[face_top:face_bottom,face_left:face_right]
        
        mask=skin_detection.process(face)
        
        mask_face=cv2.bitwise_and(face,face,mask=mask)
        no_pixels=np.sum(mask>0)
       
        #channel mean
        mean_r=(np.sum(mask_face[:,:,2]))/no_pixels
        mean_g=(np.sum(mask_face[:,:,1]))/no_pixels
        mean_b=(np.sum(mask_face[:,:,0]))/no_pixels
        
        if frame_count==0:
            mean_rgb=np.array([mean_r,mean_b,mean_g])
        else:
            mean_rgb=np.vstack((mean_rgb,np.array([mean_r,mean_b,mean_g])))
        frame_count+=1
        

    
    #plot signal
#    if frame_count>0:
#        f=np.arange(0,mean_rgb.shape[0])
#        plt.plot(f, mean_rgb[:,0] , 'r', f,  mean_rgb[:,1], 'g', f,  mean_rgb[:,2], 'b')
#        plt.show()       
    
    l = int(fps * 1.6)
    print("Window Length : ",l)

    
    
    if frame_count>=l:
        

        for t in range(0, (mean_rgb.shape[0]-l)):
        #t = 0
        # Step 1: Spatial averaging
            C = mean_rgb[t:t+l-1,:].T
    
            #Step 2 : Temporal normalization
        mean_color = np.mean(C, axis=1)
        #print("Mean color", mean_color)
        
        diag_mean_color = np.diag(mean_color)
        #print("Diagonal",diag_mean_color)
        
        diag_mean_color_inv = np.linalg.inv(diag_mean_color)
        #print("Inverse",diag_mean_color_inv)
            
        Cn = np.matmul(diag_mean_color_inv,C)
    
            #Step 3: 
        projection_matrix = np.array([[0,1,-1],[-2,1,1]])
        S = np.matmul(projection_matrix,Cn)
    
        #2D signal to 1D signal
        std = np.array([1,np.std(S[0,:])/np.std(S[1,:])])
    
        P = np.matmul(std,S)
    
    #Step 5: Overlap-Adding
        H[t:t+l-1] = H[t:t+l-1] +  (P-np.mean(P))/np.std(P)
#    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
#    cv2.imshow('RealSense', mask_face)
#    key=cv2.waitKey(1)
#    if key & 0xFF == ord('q') or key == 27:
#        cv2.destroyAllWindows()
#        break
cam.release()
end_time=time.time()-start_time

