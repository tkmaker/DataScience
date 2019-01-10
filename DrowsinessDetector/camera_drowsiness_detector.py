#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 23:51:02 2018

@author: trushk
"""

import face_recognition
import cv2
from scipy.spatial import distance as dist
import subprocess

resize_ratio = 0.12
#picture resize value affects this ratio threshold
ratio_threshold = 3

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

#eye aspect ratio - smaller ratio indicates a possible closed eye
def eye_aspect_ratio(eye):
    
    ratio = (dist.euclidean(eye[1],eye[5]) + dist.euclidean(eye[2],eye[4])) \
            /  2 *(dist.euclidean(eye[0],eye[3]))
    
    #print (ratio)
    return ratio


blink_counter = 0

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Resize image and convert to grayscale
    image = cv2.resize(rgb_frame,(0,0),fx=resize_ratio,fy=resize_ratio)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    
    try:
        #Extract facial landmarks
        landmarks = face_recognition.face_landmarks(gray)[0]

        #Extract left and right Eye Aspect Ratios (EAR)
        l_ear = eye_aspect_ratio(landmarks['left_eye'])
        r_ear = eye_aspect_ratio(landmarks['right_eye'])
        
        #Average Left and Right EAR
        ear = (l_ear + r_ear) / 2
        
         
        #Debug - Uncomment to see EAR
        #print (ear)
        
        #If EAR is above threshold we consider it as an open eye state
        if ear >= ratio_threshold:
            eye_status = 'Open'
        else: 
            eye_status = 'Closed'
            #Increment blink counter if eyes are closed
            blink_counter +=1
            #Play system sound when eyes are closed
            subprocess.call(['/usr/bin/canberra-gtk-play','--id','bell'])
     
        cv2.putText(frame, "Eye Status: {} Blink Counter: {}".format(eye_status,blink_counter), (10, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    except:
        next
        #print ("No face detected")
        
    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()