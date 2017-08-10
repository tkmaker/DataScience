# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 19:16:42 2017

@author: takalyan
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import re


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



#Function to calculate letter frequency in a word
def letterCount(word):
    return {c: word.count(c) for c in word}


X = pd.DataFrame()

#Loop throgh all rows
#  - Convert char to unicode
#  - Calculate letter frequencies to create new features
for i in range (0,data.shape[0]):
    
    #Extract sequence
    seq = data.loc[i,'Sequence']
    
    #Remove whitespace
    seq = re.sub("\s+","", seq)
    
    #Conver char to unicode
    for j in range (0,len(seq)):
        X.loc[i,j] = ord(seq[j])
    
    #Count frequency of alphabets
    cnt = letterCount(seq)
    #Feature Engineering
    for key in (cnt):
        X.loc[i,key] = cnt[key]
    
#Replace missing values of frequency counts with 0
X.fillna(0, inplace=True)



#Build the model
from sklearn.model_selection import KFold,cross_val_score
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier


def create_model():
	# create model
	model = Sequential()
	model.add(Dense(68, input_dim=68, activation='relu'))
	model.add(Dense(32, input_dim=68, activation='relu'))
	model.add(Dense(3, activation='sigmoid'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


#model = KerasClassifier(build_fn=create_model(), epochs=200, batch_size=8, verbose=0)

model = create_model()


kf = KFold(n_splits=10) # Define the split - into 2 folds 

for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

X_train = np.array(X_train)
y_train = np.array(y_train)

model.fit(X_train,y_train,batch_size = 8, nb_epoch = 100)

y_pred= model.predict(np.array(X_test))
y_test
#kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
#results = cross_val_score(model, X_train, y_train, cv=kfold)


#np.average(results)

