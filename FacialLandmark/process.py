# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 14:06:08 2018

@author: Trushant Kalyanpur
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import csv



IMAGE_WIDTH = 96
IMAGE_HEIGHT = 96





def load_dataset(file):
    
    X = []
    y = []
    
    
    #Parse input CSV
    with open(file) as csvfile:
        
        #Create CSV reader for lines in file
        csv_reader = csv.DictReader(csvfile)
        
        #Iterate through all rows 
        for row in csv_reader:
            
            y_keypts = []
            
            #Handle the image creation. It is stored as a string in the last column called "Image"
            
            #Create a baseline img for every iteration
            img = np.zeros((IMAGE_HEIGHT,IMAGE_WIDTH,1), dtype=np.float)
            
            #Iterate through image column and split pixel values
            for i, val in enumerate(row["Image"].split(" ")):
                #store as 3D array (x,y,channel) - Only 1 channel here
                img[i//IMAGE_WIDTH,i%IMAGE_WIDTH,0] = val
            
            has_keypts = True
            
            for col in row:
                if (col == "Image"):
                    continue
                elif (row[col] == ""):
                    has_keypts = False
                else : 
                    y_keypts.append(np.float(row[col]))
            
            
            if has_keypts:
               X.append(img)
               y.append(y_keypts)
            
                
         
        
    return np.array(X), np.array(y)

def show_image(X, Y):
    
    #Show image
    plt.imshow(X[:,:,0],cmap='gray')
    plt.show()
    
    #copy image - we dont want to alter the original
    img = np.copy(X)
    
    #Iterate through keypts and skip by 2  since they are stored in x,y pairs
    for i in range(0,Y.shape[0],2):
        #If y coord is part of image and x coord is part of image - change pixel color
        if 0 < Y[i+1] < IMAGE_HEIGHT and 0 < Y[i] < IMAGE_WIDTH:
            img[int(Y[i+1]),int(Y[i]),0] = 255
    
    #Show image with keypoints
    plt.imshow(img[:,:,0],cmap='gray')
    

#Get image and keypoints
X,y= load_dataset('training.csv')



#save data 
with open('X.pickle','wb') as f:
    pickle.dump(X,f)

with open('y.pickle','wb') as f:
    pickle.dump(y,f)    
    

show_image(X[100],y[100])