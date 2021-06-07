# -*- coding: utf-8 -*-
"""
Created on Fri May  7 19:51:22 2021

@author: Allan
"""

import numpy as np
import matplotlib.pyplot as plt
import dlib
import cv2
import time
import scipy.io


pathin='cv_camera_sensor_stream_handler.avi'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

a1=scipy.io.loadmat('filt_coefficient.mat')
a=a1['h']
a=np.transpose(a)
a=a[:,0]

cam=cv2.VideoCapture(pathin)
fps=cam.get(cv2.CAP_PROP_FPS)

start =0
end=2500

count=0
i=0
j=0

diff=np.zeros([1651])
H=np.zeros([1651])
inter_beat=[]


start_time=time.time()
while count<end:
    ret,frame=cam.read()
    
    if count==len(H):
        break

    
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
        y5=landmark.part(42).y-15
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
    l=int(fps*1.5)
    
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
        try:
            y=a*P            
        except:
            continue
    
    # Overlap-Adding
        H[t:t+l-1] = H[t:t+l-1] +  (y-np.mean(y))/np.std(y)    
        
               
    #Moving average
#        new_avg=[sum(avg_pulse[i:i+2])/2 for i in range(len(avg_pulse)-2+1)]
    
    #Peak detection - IBI
    
        con=np.sign(float(H[count-2]-H[count-3]))
        
        diff[i]=con
        
        if diff[i]!=diff[i-1]:
            if diff[i]==-1:
                peak=count-1
                inter_beat.append(peak)
        i+=1

#        d=np.arange(0,H.shape[0])
#        
#        plt.plot(d,H,'c')
#        plt.pause(0.0001)
    
    cv2.namedWindow('window', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('window', frame)
    key=cv2.waitKey(1)
    if key & 0xFF == ord('q') or key == 27:
        cv2.destroyAllWindows()
        break
cam.release()
end_time=time.time()-start_time
pulse=[sum(H[i:i+6])/6 for i in range(len(H)-6+1)]

tot_time=np.linspace(0,int(count/fps),num=len(pulse))
plt.plot(pulse)