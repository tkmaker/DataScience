# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 19:59:43 2018

@author: Trushant Kalyanpur
"""


import pandas as pd
import numpy as np
from kerasSentModel import kerasSentiment
from sklearn.metrics import confusion_matrix,accuracy_score


train_samples = 500000
train_samples_end = train_samples + 5000

data_df =  pd.read_csv('train.csv')
test_df = data_df.iloc[train_samples:train_samples_end:]

#replace empty cells with Nan
test_df.review.replace('', np.nan, inplace=True)
#drop null rows
test_df.dropna(inplace=True)

#Instantiate model
kerasModel = kerasSentiment()
kerasModel.loadModel('kerasModelLSTM')


#assign train dataframe and define which columns have the data
kerasModel.text_df = test_df
kerasModel.text_colname = "review"
kerasModel.num_classes = 3
kerasModel.sentiment_colname = "sentiment"


#Get padded docs
padded_docs = kerasModel.encodeTestText()

#Get predictions based on padded docs
predicted_sentiment = kerasModel.predict(padded_docs)
#ground truth
actual_sentiment = test_df.sentiment

#Model evaluation
#confusion matrix
cm = confusion_matrix(actual_sentiment,predicted_sentiment)
#Accuracy
acc = accuracy_score(actual_sentiment,predicted_sentiment) * 100

print ("Accuracy on test set = %d\n"%acc)



