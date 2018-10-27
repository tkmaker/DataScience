# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 20:23:32 2018

@author: Trushant Kalyanpur
"""

import pickle
import time
from sklearn.metrics import roc_auc_score,roc_curve,auc
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Dense,Dropout,Conv2D,Activation,MaxPool2D,BatchNormalization,Flatten,Conv2DTranspose,Reshape,Input
from sklearn.model_selection import KFold
from keras.callbacks import EarlyStopping,ReduceLROnPlateau
from keras.optimizers import Adam,SGD,RMSprop
from keras.models import Sequential
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras import regularizers
from sklearn.model_selection import train_test_split


from keras.preprocessing.image import ImageDataGenerator


with open('X.pickle','rb') as xfp:
    X = pickle.load(xfp)

with open('y.pickle','rb') as yfp:
    y = pickle.load(yfp)    



def createResnetModel ():
    
    #baseline Resnet50 model
    base_model = ResNet50(include_top=False, input_tensor=Input(shape=(96,96,3)), pooling='avg')
    base_model.trainable = False
    
    
    #Top model added after Resnet
    dropout  = 0.5
    kernel_init  = 'he_uniform'
    use_bn = False
    
    #Dense layer
    top_model = Sequential()
    top_model.add(Dense(512,  input_shape=(2048,),kernel_initializer=kernel_init))
    if use_bn:
        top_model.add(BatchNormalization())
    top_model.add(Activation('relu'))
    
    #Dense layer    
    top_model.add(Dense(256, kernel_initializer=kernel_init))
    if use_bn:
        top_model.add(BatchNormalization())
    top_model.add(Activation('relu'))

    #Dropout    
    top_model.add(Dropout(dropout))
    
    #Dense layer
    top_model.add(Dense(128, kernel_regularizer=regularizers.l2(0.01),kernel_initializer=kernel_init))
    if use_bn:
        top_model.add(BatchNormalization())
    top_model.add(Activation('relu'))
    
    #Dropout    
    top_model.add(Dropout(dropout))
    
    #Dense layer
    top_model.add(Dense(96, kernel_regularizer=regularizers.l2(0.01),kernel_initializer=kernel_init))
    if use_bn:
        top_model.add(BatchNormalization())
    top_model.add(Activation('relu'))
    
    #Dropout    
    top_model.add(Dropout(dropout))
    
    top_model.add(Dense(48, kernel_regularizer=regularizers.l2(0.01),kernel_initializer=kernel_init))
    if use_bn:
        top_model.add(BatchNormalization())
    top_model.add(Activation('relu'))
    
    
    top_model.add(Dense(30))
    
    top_model.compile(loss='mse', optimizer=Adam(0.001), metrics=['mae'])
    
    #Combine models
    final_model = Sequential([base_model, top_model])
    final_model.compile(loss='mse', optimizer=Adam(0.001), metrics=['mae'])
    final_model.summary()

    return final_model



    
    
    
def trainModel (datagen, model, X, y):
       
        print ("Training model \n")
        
        
        X_train,X_val,y_train,y_val = train_test_split(X,y,test_size = 0.2, shuffle=True)
      

        #Use early stopping to avoid overfit
        earlystop = EarlyStopping(patience=early_stop_epochs,monitor='val_loss')
        reduce_lr =  ReduceLROnPlateau(monitor='loss', factor=0.1, \
                                       verbose=verbose,  mode='min')
        
        callbacks_list = [earlystop,reduce_lr]
        
        #track model training time
        start_time = time.time()

        generator = datagen.flow(X_train, y_train, batch_size = batch_size)

        # Fitting the model to the training set
        model_info =  model.fit_generator(
                generator,
                steps_per_epoch=batch_size,
                epochs=epochs,
                shuffle=True,
                verbose=verbose,
                validation_data = (X_val, y_val),
                callbacks = callbacks_list)
            
        # plot model history
        plot_model_history(model_info,'mean_absolute_error','val_mean_absolute_error')
   
        #Evaluate the model
        scores = model.evaluate(X_val,y_val)
        print("\n%s: %.2f\n" % (model.metrics_names[1],scores[1]))
 
         

        end_time = time.time()
        train_time = (end_time - start_time)/60
  
        print("Model train time = %f minutes" % train_time)
        
        
def trainModelKfold (datagen, model, X, y):
       
        print ("Training model using K-Fold cross val\n")
        
     
        
        #K-Fold Cross validation
        kf = KFold(n_splits=5,shuffle=True, random_state=1) 

        #Store cross val scores
        cv_scores = []

        #Use early stopping to avoid overfit
        earlystop = EarlyStopping(patience=early_stop_epochs,monitor='val_loss')
        reduce_lr =  ReduceLROnPlateau(monitor='loss', factor=0.1, \
                                       verbose=verbose,  mode='min')
        
        callbacks_list = [earlystop,reduce_lr]
        
        #track model training time
        start_time = time.time()

        #Do k-fold cross val across dataset and store scores
        for train_index, val_index in kf.split(X):
            
            
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            generator = datagen.flow(X_train, y_train, batch_size = batch_size)

            # Fitting the model to the training set
            model_info =  model.fit_generator(
                generator,
                steps_per_epoch=batch_size,
                epochs=epochs,
                shuffle=True,
                verbose=verbose,
                validation_data = (X_val, y_val),
                callbacks = callbacks_list)
            
            # plot model history
            plot_model_history(model_info,'mean_absolute_error','val_mean_absolute_error')
   
            #Evaluate the model
            scores = model.evaluate(X_val,y_val)
            print("\n%s: %.2f%%\n" % (model.metrics_names[1],scores[1]*100))
            cv_scores.append(scores[1]*100)
 
         

        end_time = time.time()
        train_time = (end_time - start_time)/60
  
        print("Model train time = %f minutes" % train_time)
        print("Overall Accuracy = %.2f%% (+/- %.2f%%)" % (np.mean(cv_scores), np.std(cv_scores)))

    


def plot_model_history(model_history,train_metric,val_metric):
        
        fig, axs = plt.subplots(1,2,figsize=(15,5))
        
        # summarize history for accuracy
        axs[0].plot(range(1,len(model_history.history[train_metric])+1),model_history.history[train_metric])
        axs[0].plot(range(1,len(model_history.history[val_metric])+1),model_history.history[val_metric])
        axs[0].set_title(train_metric)
        axs[0].set_ylabel(train_metric)
        axs[0].set_xlabel('Epoch')
        axs[0].set_xticks(np.arange(1,len(model_history.history[train_metric])+1),len(model_history.history[train_metric])/10)
        axs[0].legend(['train', 'val'], loc='best')
        
        # summarize history for loss
        axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
        axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
        axs[1].set_title('Model Loss')
        axs[1].set_ylabel('Loss')
        axs[1].set_xlabel('Epoch')
        axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
        axs[1].legend(['train', 'val'], loc='best')
        plt.show()



#Data augmentation    
train_datagen = ImageDataGenerator(
        #rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        #shear_range=0.2,
        #zoom_range=0.2,
        #horizontal_flip=True,
        #fill_mode='nearest'
        )

#Model hyper params
epochs  = 50
batch_size = 32
early_stop_epochs = 25
verbose = True

#Baseline images are 1 channel. Convert to 3 channel 
x = np.array([ X[:,:,:,0],X[:,:,:,0],X[:,:,:,0]])

#Move axes to match (batches,width,height,channels)
x = np.swapaxes(x,0,1)
x = np.swapaxes(x,1,2)
x = np.swapaxes(x,2,3)


model = createResnetModel()
trainModel(train_datagen,model,x,y)

#Save model                   
model.save('resnet.h5')