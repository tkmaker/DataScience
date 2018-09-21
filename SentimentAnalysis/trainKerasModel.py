# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 19:59:43 2018

@author: Trushant Kalyanpur
"""


import pandas as pd
import numpy as np
from kerasSentModel import kerasSentiment





data_df = pd.read_csv( "labelledTrainData.tsv", header=0, 
 delimiter="\t", quoting=3 )


train_samples = 15000
train_df = data_df.iloc[0:train_samples,:]

#replace empty cells with Nan
train_df.review.replace('', np.nan, inplace=True)
#drop null rows
train_df.dropna(inplace=True)

#Instantiate model
kerasModel = kerasSentiment()

train_df['review'] = train_df.review.apply(lambda x : kerasModel.cleanup_review(x))

#assign train dataframe and define which columns have the data
kerasModel.text_df = train_df
kerasModel.text_colname = "review"
kerasModel.num_classes = 2
kerasModel.sentiment_colname = "sentiment"
#kerasModel.max_sentence_length = 100

#Get padded docs
padded_docs, labels= kerasModel.encodeTrainText()

#optional
#Use pre-Trained glove model
#kerasModel.glove_path = "glove6b_100d_Dict.pickle"
#embedding_matrix = kerasModel.getPretrainedEmbedding(embed_type="glove")


#Create the model
#kerasModel.createModel(embedding_matrix=embedding_matrix,useTrainedEmbedding=True)
kerasModel.createModel()

#Model hyperparameters
kerasModel.lstm_units = 10
kerasModel.early_stop_epochs = 10
kerasModel.epochs = 5
kerasModel.batch_size = 128
kerasModel.verbose = True

#Train the model
kerasModel.trainModelKfold(padded_docs,labels)
#Save model 
kerasModel.saveModel('kerasModelLSTM')

