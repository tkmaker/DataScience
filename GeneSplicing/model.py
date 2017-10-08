"""
Created on Mon Aug  7 19:16:42 2017

@author: takalyan
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import re

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

dataset = pd.read_csv('data.csv')


#Take a look at the data 
dataset.describe()

#3178 Unique donors and 3092 unique sequences

#Create Duplicate column with boolean values for all duplicate donors
dataset['Duplicate'] = dataset.duplicated('Donor')

#index 54 and 559 are duplicate donors strings-
#confirmed they have same sequence so we can skip duplicate donors

#Filter out duplicate donors
data = dataset.loc[dataset['Duplicate'] == False]
data= data.reset_index(drop=True)

#Create target variable with encoded class
y = pd.DataFrame(data['Class'])
cols_to_transform = y.dtypes[y.dtypes=="object"].index
y = pd.get_dummies(y,columns = cols_to_transform )


X = pd.DataFrame()

#Loop throgh all rows
#  - Convert char to unicode
for i in range (0,data.shape[0]):
    
    #Extract sequence
    seq = data.loc[i,'Sequence']
    
    #Remove whitespace
    seq = re.sub("\s+","", seq)
    
    #Conver char to unicode
    for j in range (0,len(seq)):
        X.loc[i,j] = ord(seq[j])
    
    
from sklearn.model_selection import train_test_split
X_train_pre,X_val_pre,y_train,y_val =  train_test_split(X,y,test_size=0.2)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()


X_train_pre=  sc.fit_transform(X_train_pre.values.astype(int))
X_val_pre=  sc.fit_transform(X_val_pre.values.astype(int))

y_train = y_train.values.astype(int)
y_val = y_val.values.astype(int)


# reshape input to be 3D for LSTM[samples, timesteps, features]
X_train = X_train_pre.reshape((X_train_pre.shape[0], 1, X_train_pre.shape[1]))
X_val =  X_val_pre.reshape((X_val_pre.shape[0], 1, X_val_pre.shape[1]))

#
# fix random seed for reproducibility
np.random.seed(7)
epochs = 10

# design network
model = Sequential()
model.add(LSTM(1000, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(3, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())
model.fit(X_train, y_train, nb_epoch=epochs, batch_size=64)
y_val_pred = model.predict(X_val)

# Final evaluation of the model
scores = model.evaluate(X_val, y_val, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

