# -*- coding: utf-8 -*-
"""
Created on Sat May 15 00:25:13 2021

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

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
rs.config.enable_device_from_file(config, "Record_light_12.bag", repeat_playback=False)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

count=0
fps=30

H=np.zeros([1800])
H_exp=np.zeros([1800])

images=[]
inter_beat=[]
temp_beat=0
i = 0
diff=[]
j=0
k=0
temp=np.zeros([2,1])
alpha=0.2


mean_all=[]
frame_all=[]
# Start streaming
profile=pipeline.start(config)
playback = profile.get_device().as_playback()
playback.set_real_time(False)

a1=scipy.io.loadmat('filt_coefficient.mat')
a=a1['h']
a=np.transpose(a)
a=a[:,0]

try:
    while True:
        time.sleep(0.001)
         # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        frame_no=frames.get_frame_number() #Get frame number
        frame_time=frames.get_timestamp() #Get timestamp
        frame_all.append(frame_time)
#        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        if frame_no<20:
            continue
        if not color_frame:
            continue

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
        l=int(fps*1.25)
    
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

        
            #Tuning & conversion to 1-D signal
            std = np.array([1,np.std(S[0,:])/np.std(S[1,:])])
        
#        P = S[0,:]+(std*S[1,:])
        
            P = np.matmul(std,S)
                #Filtering for window
            try:
                y=a*P            
            except:
                continue
            
    
    #Step 5: Overlap-Adding
            H[t:t+l-1] = H[t:t+l-1] +  (y-np.mean(y))/np.std(y)
#        d=np.arange(0,H.shape[0])
            #Peak detection - IBI
        if k>=1:
            #Single Exponential smoothing
            H_exp[j]=alpha*H[j-1]+(1-alpha)*H_exp[j-1]
            #Peak detection
            con=np.sign((H_exp[j]-H_exp[j-1]))
            
            temp[1]=con
            

            if temp[0]!=temp[1]:
                if temp[1]==-1:
#                    peak=k-1
                    inter_beat.append(k-1)
            #Heart rate every 10 seconds
            if not (j-l)%300 and count>=l:
                heart_rate=6*(len(inter_beat)-temp_beat)
                temp_beat=len(inter_beat)
                print("heart rate",heart_rate)
            j+=1
        k+=1
        temp[0]=temp[1]
#        plt.plot(d,H,'c')
#        plt.pause(0.0001)
    
        cv2.namedWindow('window', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('window', frame)
        key=cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break        
        
#        print("frame number",frame_no)
except RuntimeError:
    print("There are no more frames left in the .bag file!")

finally:

    # Stop streaming
    pipeline.stop()
pulse=[sum(H[i:i+6])/6 for i in range(len(H)-6+1)]

tot_time=np.linspace(0,int(count/fps),num=len(H_exp))
inter_beat=np.array(inter_beat)
plt.plot(tot_time,H_exp)
plt.plot(tot_time[inter_beat],H_exp[inter_beat],'x')