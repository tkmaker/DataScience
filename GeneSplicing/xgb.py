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

#Display index of duplicates
print (dataset[dataset['Duplicate'] == True].index.tolist())

#Check one duplicate value
dup_donor = dataset.iloc[86]['Donor']

print (dataset[dataset['Donor'] == dup_donor].index.tolist())

#index 84 and 86 are duplicate donors strings

print (re.sub("\s+","",dataset.iloc[84]['Sequence']), re.sub("\s+","",dataset.iloc[86]['Sequence']))

#confirmed they have same sequence so we can skip duplicate donors

#Filter out duplicate donors
data = dataset.loc[dataset['Duplicate'] == False]
data= data.reset_index(drop=True)

from nltk.probability import FreqDist
genome_list = re.sub("\s+","",data['Sequence'].str.cat())
dist = FreqDist(genome_list)
dist.plot()

#Create target variable with encoded class
y = pd.DataFrame(data['Class'])
y.loc[:,'Class'] = y['Class'].replace({'EI': '0'}, regex=True)
y.loc[:,'Class'] = y['Class'].replace({'IE': '1'}, regex=True)
y.loc[:,'Class'] = y['Class'].replace({'N': '2'}, regex=True)
y = y.iloc[:,-1]

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
import xgboost as xgb


# train model
model = xgb.XGBClassifier(max_depth= 10,   min_child_weight= 3,\
                         subsample= 0.5, \
                         objective= 'multi:softprob', silent= 1)

seed = 5
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

results = cross_val_score(model, X, y, cv=kfold)


print ("Average Accuracy = ",np.average(results))

model.fit(X,y)
seed = 5
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

scores = cross_val_score(model, X.iloc[0:2861,:], y[0:2861,], cv=kfold)


print ("Average Accuracy = ",np.mean(scores))

pred = model.predict(X.iloc[2861:,:])
res = pd.DataFrame()
res['Predictions'] = pred
res['Actual'] = y[2861:,]
res['Correct'] = res['Predictions']==res['Actual']
res.describe()

