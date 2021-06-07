# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 22:12:54 2021

@author: Allan
"""


import pyrealsense2 as rs
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import time
import dlib
import scipy.io


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
#rs.config.enable_device_from_file(config, "C:/Users/Allan/Desktop/JRF/Realsense/Python/test3.bag", repeat_playback=False)
pipeline = rs.pipeline()
profile=pipeline.start(config)
device = profile.get_device()
depth_sensor = device.query_sensors()[0]
set_laser = 0
depth_sensor.set_option(rs.option.laser_power, set_laser)
count=0
fps=30

H=np.zeros([1800])
diff=np.zeros([1651])

images=[]
i = 0

mean_all=[]
frame_all=[]

#Load filter Coeffcients
a1=scipy.io.loadmat('filt_coefficient.mat')
a=a1['h']
a=np.transpose(a)
a=a[:,0]

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        frame_no=frames.get_frame_number() #Get frame number
        frame_time=frames.get_timestamp() #Get timestamp
        frame_all.append(frame_no)
        color_frame = frames.get_color_frame()
        
        if frame_no<20:
            continue
        if not color_frame:
            continue
        if frame_no>=1800:
            break

        # Convert images to numpy arrays
        
        frame = np.asanyarray(color_frame.get_data())
        #Convert color to gray for classifier
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        faces=detector(gray,0)
        
        
    
        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()
            
        landmark=predictor(gray,face)
    
        x1=landmark.part(17).x
        y1=landmark.part(17).y-40
        x2=landmark.part(14).x
        y2=landmark.part(14).y
        x3=landmark.part(36).x
        y3=landmark.part(36).y-15
        x4=landmark.part(39).x
        y4=landmark.part(39).y+15
        x5=landmark.part(42).x
        y5=landmark.part(42).y-20
        x6=landmark.part(45).x
        y6=landmark.part(45).y+15
        
        
        frame[y3:y4,x3:x4]=0
        frame[y5:y6,x5:x6]=0
#    cv2.circle(frame,(x5,y5),4, (255, 0, 0), -1)
#    cv2.circle(frame,(x6,y6),4, (255, 0, 0), -1)
#    cv2.circle(frame,(x3,y3),4, (255, 0, 0), -1)
#    cv2.circle(frame,(x4,y4),4, (255, 0, 0), -1)
#    cv2.rectangle(frame,(x1,y1),(x2,y2),(255, 0, 0),2)
    
        roi=frame[y1:y2,x1:x2]
        no_pixel=np.sum(roi>0)
    
        r=np.sum(roi[:,:,2])/no_pixel
        g=np.sum(roi[:,:,1])/no_pixel
        b=np.sum(roi[:,:,0])/no_pixel
    
        if count==0:
            mean_rgb = np.array([r,g,b])
        else:
            mean_rgb = np.vstack((mean_rgb,np.array([r,g,b])))
    
        count+=1
        l=int(fps*1.6)
    
#    H=np.zeros(mean_rgb.shape[0])
    
    
        if count>l:
            for t in range(0,mean_rgb.shape[0]-l+1):
                #Spatial averaging
                C=mean_rgb[t:t+l-1,:].T
    
    #Temporal normalization
            mean_color = np.mean(C, axis=1)
    
            diag_mean_color = np.diag(mean_color)
    
            diag_mean_color_inv = np.linalg.inv(diag_mean_color)
    
            Cn = np.matmul(diag_mean_color_inv,C)
    
            #Separate specular & pulse components
            projection_matrix = np.array([[0,1,-1],[-2,1,1]])
            S = np.matmul(projection_matrix,Cn)

        
    #tuning & conversion to 1-D signal
            std = np.array([1,np.std(S[0,:])/np.std(S[1,:])])
        
#        P = S[0,:]+(std*S[1,:])
        
            P = np.matmul(std,S)
            
                            #Filtering for window
#            try:
#                y=a*P            
#            except:
#                continue
    
    #Step 5: Overlap-Adding
            H[t:t+l-1] = H[t:t+l-1] +  (P-np.mean(P))/np.std(P)
#        
#        plt.plot(d,H,'c')
#        plt.pause(0.0001)
    
        cv2.namedWindow('window', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('window', frame)
        key=cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break        
        
        print("frame number",frame_no)
except RuntimeError:
    print("There are no more frames left in the .bag file!")

finally:

    pipeline.stop()

pulse=[sum(H[i:i+6])/6 for i in range(len(H)-6+1)]

tot_time=np.linspace(0,int(count/fps),num=len(pulse))
plt.plot(pulse)