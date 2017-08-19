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


from nltk.probability import FreqDist
genome_list = re.sub("\s+","",dataset['Sequence'].str.cat())
dist = FreqDist(genome_list)
dist.plot()

from sklearn.preprocessing import LabelBinarizer, LabelEncoder

#Create target variable with encoded class
y = pd.DataFrame(dataset['Class'])
lbl = LabelEncoder()
y= lbl.fit_transform(np.array(y))

#Function to calculate letter frequency in a word
def letterCount(word):
    return {c: word.count(c) for c in word}


X = pd.DataFrame()

#Loop throgh all rows
#  - Convert char to unicode
#  - Calculate letter frequencies to create new features
for i in range (0,dataset.shape[0]):
    
    #Extract sequence
    seq = dataset.loc[i,'Sequence']
    
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
import xgboost as xgb


#Define model
model = xgb.XGBClassifier(max_depth= 10,   min_child_weight= 3,\
                         subsample= 0.5, \
                         objective= 'multi:softprob', silent= 1)


#Split train and test 
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=45)

#Train the model
model.fit(X_train,y_train,eval_metric='merror')

#Check against test sample
seed = 5
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
scores = cross_val_score(model, X_test, y_test, cv=kfold)

print ("Average Accuracy = ",np.mean(scores))
