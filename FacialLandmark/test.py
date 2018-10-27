# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 14:40:15 2018

@author:Trushant Kalyanpur
"""

from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import pickle

IMAGE_HEIGHT = 96
IMAGE_WIDTH = 96


#Load model
model = load_model('resnet.h5')


def show_predictions(X, Y,Y_true):
    
   
    
    #copy image - we dont want to alter the original
    img = np.copy(X)
    
    ###Predicted labels
    
    #Iterate through keypts and skip by 2  since they are stored in x,y pairs
    for i in range(0,Y.shape[0],2):
        #If y coord is part of image and x coord is part of image - change pixel color
        if 0 < Y[i+1] < IMAGE_HEIGHT and 0 < Y[i] < IMAGE_WIDTH:
            img[int(Y[i+1]),int(Y[i]),0] = 255
    
    ###Ground truth
    #Iterate through keypts and skip by 2  since they are stored in x,y pairs
    for i in range(0,Y_true.shape[0],2):
        #If y coord is part of image and x coord is part of image - change pixel color
        if 0 < Y_true[i+1] < IMAGE_HEIGHT and 0 < Y_true[i] < IMAGE_WIDTH:
            img[int(Y_true[i+1]),int(Y_true[i]),0] = 3
            
    #Show image with keypoints
    plt.imshow(img[:,:,0],cmap='gray')

    
  
with open('X_test.pickle','rb') as f:
    X_test = pickle.load(f)
    
#Baseline images are 1 channel. Convert to 3 channel 
x_test = np.array([ X_test[:,:,:,0],X_test[:,:,:,0],X_test[:,:,:,0]])

#Move axes to match (batches,width,height,channels)
x_test = np.swapaxes(x_test,0,1)
x_test = np.swapaxes(x_test,1,2)
x_test = np.swapaxes(x_test,2,3)


with open('X.pickle','rb') as f:
    X = pickle.load(f)
   
with open('y.pickle','rb') as f:
    y = pickle.load(f)
    
#Baseline images are 1 channel. Convert to 3 channel 
x = np.array([ X[:,:,:,0],X[:,:,:,0],X[:,:,:,0]])

#Move axes to match (batches,width,height,channels)
x = np.swapaxes(x,0,1)
x = np.swapaxes(x,1,2)
x = np.swapaxes(x,2,3)


#Select image index here
img_indx = 33

#Get prediction and dsiaply results
y_pred = model.predict(x[img_indx:(img_indx+1)])
y_true = y[img_indx:(img_indx+1)]

show_predictions(x[img_indx], y_pred[0],y_true[0])


