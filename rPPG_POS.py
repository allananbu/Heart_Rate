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

start_time=time.time()

pathin='videoplayback.mp4'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    

    
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
       

#    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
#    cv2.imshow('RealSense', frame)
#    key=cv2.waitKey(1)
#    if key & 0xFF == ord('q') or key == 27:
#        cv2.destroyAllWindows()
#        break
end_time=time.time()-start_time

