# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 11:13:34 2018

@author: Trushant Kalyanpur
"""

from numpy import zeros
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense,Flatten,Embedding,LSTM
from keras.callbacks import EarlyStopping
from keras.models import load_model
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelBinarizer


import time
from sklearn.model_selection import KFold

class kerasSentiment (object):
    
    def __init__ (self) :
        #model verbosity
        self.verbose = False
        self.text_df = pd.DataFrame ()
        #Column name of dataframe containing text
        self.text_colname = ""
        #Column name of dataframe containing sentiment
        self.sentiment_colname = ""
        #Number of classes to predict
        self.num_classes = 1
        #Keras tokenizer for embedding
        self.tokenizer = Tokenizer()
        #Label binarizer to encode target variables
        self.labelBinarizer = LabelBinarizer()
        #Path to pre-trained glove embedding
        self.glove_path = ""
        #PAth to pre-trained word2vec embedding
        self.word2vec_path = ""
        #Embedding dimenssions 
        self.embeddingDim = 100
        #Vocabulary size computed internally
        self.vocab_size = 0
        #Max sentence length computed internally or can be overridden 
        self.max_sentence_length  = 0
        #The Keras Model
        self.model = Sequential()
        #LSTM memory units
        self.lstm_units = 50
        #LSTM dropout
        self.lstm_dropout = 0.2
        #Early stopping epochs 
        self.early_stop_epochs = 50
        #Number of K-Folds
        self.num_folds = 5
        #Model epochs
        self.epochs = 1000
        #Batch size
        self.batch_size = 64
           
    def  cleanup_review (self,text,remove_stop_words=False,stemmer=False,lemma=False):
        # Function to clean text in a document and return a sentence for sentiment analysis
        #
        # Remove HTML
        text = BeautifulSoup(text,"lxml").get_text()
        #  
        # Remove non-letters
        text = re.sub("[^a-zA-Z]"," ", text)
        #
        # Convert words to lower case 
        words= text.lower()
    
        #Optionally lemmatize words (false by default)
        if lemma:
            lemmatizer = WordNetLemmatizer()
            words = lemmatizer.lemmatize(words)
        
        #tokenize words
        words = words.split()
    
      

        # Optionally remove stop words (false by default)
        if remove_stop_words:
            stops = set(stopwords.words("english"))
            words = [w for w in words if not w in stops]
    

        #Optionally stem words (false by default)
        if stemmer:
            porter = PorterStemmer()
            words = [porter.stem(word) for word in words]

   
        #Recombine words into a sentence
        sentence = "".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in words]).strip()
    
        return(sentence)

    def encodeTrainText (self) :
        
        print ("Encoding and padding text..\n")
        
        
        #Find max sentence length in text
        sentence_length = self.text_df[self.text_colname].apply(lambda x: len(x.split()))
        
        #If user doesnt specify max sentence length, find the max length in doc
        if (self.max_sentence_length == 0) :
            self.max_sentence_length = max(sentence_length)
        
        print ("Max sentence length in doc:%i\n"%self.max_sentence_length)
        
        #Convert dataframe to list of text and labels 
        docs =[]
        
        for index,row in self.text_df.iterrows():
            docs.append(row[self.text_colname])
    
                  
       
        labels = self.labelBinarizer.fit_transform(self.text_df[self.sentiment_colname])
               
            
        #prepare Keras tokenizer
        self.tokenizer.fit_on_texts(docs)

        #Get size of vocabulary        
        self.vocab_size = len(self.tokenizer.word_index) + 1
        
        print ("Encoding text..\n")
        #Convert to integer encoded values
        encoded_docs = self.tokenizer.texts_to_sequences(docs)


        print ("Padding text..\n")
        # pad documents to a max length of words
        padded_docs = pad_sequences(encoded_docs, maxlen=self.max_sentence_length, padding='post')

        
        return padded_docs,labels
    
    def encodeTestText (self):
        
         #Convert dataframe to list of text and labels 
        docs =[]
        
        for index,row in self.text_df.iterrows():
            docs.append(row[self.text_colname])
    
        
        print ("Encoding text..\n")
        
        #Convert to integer encoded values
        encoded_docs = self.tokenizer.texts_to_sequences(docs)


        print ("Padding text..\n")
        # pad documents to a max length of words
        padded_docs = pad_sequences(encoded_docs, maxlen=self.max_sentence_length, padding='post')

        return padded_docs
        
    def getPretrainedEmbedding (self, embed_type = "glove"):
       
        print ("Getting pre-trained word embeddings using %s vectors\n"%embed_type)
        
        if (embed_type=="glove"):
            with open (self.glove_path,'rb') as f:
                embedding_dict = pickle.load(f)
        elif (embed_type=="word2vec"):
            with open (self.word2vec_path,'rb') as f:
                embedding_dict = pickle.load(f)
                
        # create a weight matrix for words in training docs
        embedding_matrix = zeros((self.vocab_size, self.embeddingDim))
        
        #iterate through all words in current doc and get pre-trained embeddings
        for word, i in self.tokenizer.word_index.items():
            embedding_vector = embedding_dict.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
                
        return embedding_matrix

   
    
    def createModel (self, embedding_matrix=[],useTrainedEmbedding=False)  :
        
        print ("Creating Keras model..\n")
        
        # define model
        #self.model = Sequential()
        
        #If using preTrained embeddings like Glove or Word2Vec, use embeddings as weight
        #Else learn weights
        if useTrainedEmbedding:
            e = Embedding(self.vocab_size, self.embeddingDim, weights=[embedding_matrix], \
              input_length=self.max_sentence_length, trainable=False)
        else :
            e = Embedding(self.vocab_size, self.embeddingDim,  \
              input_length=self.max_sentence_length, trainable=True)
            
        self.model.add(e)
        #self.model.add(Flatten())
        self.model.add(LSTM(units=self.lstm_units,dropout=self.lstm_dropout,\
                            recurrent_dropout=self.lstm_dropout))

        #Output layer number of units depends on the number of classes
        if (self.num_classes > 2 or self.num_classes == 1) :
            output_dim = self.num_classes
        elif self.num_classes ==2  :
            output_dim = self.num_classes - 1 
        else :
            print ("Invalid number of classes = ",self.num_classes)
            exit
            
        
        #Use softmax for multiclass and sigmoid for binary classification
        if (self.num_classes > 2) :
            self.model.add(Dense(output_dim, activation='softmax'))
        else :
            self.model.add(Dense(output_dim, activation='sigmoid'))
        
        # compile the model
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
        # summarize the model
        print(self.model.summary())
        
    
    
 

    def trainModel (self,padded_docs, labels):
        
        print ("Training model..\n")
        
        #convert to array for Keras classification
        labels_arr = np.array(labels)
        
        # fit the model
        self.model.fit(padded_docs, labels_arr, epochs=50, verbose=self.verbose)
        
        # evaluate the model
        loss, accuracy = self.model.evaluate(padded_docs, labels_arr, verbose=self.verbose)
        print('Accuracy: %f' % (accuracy*100))
        

    def trainModelKfold (self, padded_docs, labels):
       
        print ("Training model using K-Fold cross val..\n")
        
        #Convert list to numpy array
        labels = np.array(labels)
        
        #K-Fold Cross validation
        kf = KFold(n_splits=self.num_folds,shuffle=True, random_state=1) # Define the split 

        #Store cross val scores
        cv_scores = []

        #Use early stopping to avoid overfit
        earlystop = EarlyStopping(patience=self.early_stop_epochs,monitor='val_loss')
        callbacks_list = [earlystop]
        
        #track model training time
        start_time = time.time()

        #Do k-fold cross val across dataset and store scores
        for train_index, test_index in kf.split(padded_docs):
            
            
            X_train, X_test = padded_docs[train_index], padded_docs[test_index]
            y_train, y_test = labels[train_index], labels[test_index]

            # Fitting the model to the training set
            model_info = self.model.fit(np.array(X_train), np.array(y_train), validation_split=0.2, \
                          verbose = self.verbose, batch_size = self.batch_size, epochs = self.epochs,\
                          callbacks=callbacks_list)

            # plot model history
            self.plot_model_history(model_info)
   
            #Evaluate the model
            scores = self.model.evaluate(X_test,y_test)
            print("\n%s: %.2f%%\n" % (self.model.metrics_names[1],scores[1]*100))
            cv_scores.append(scores[1]*100)
 
         

        end_time = time.time()
        train_time = (end_time - start_time)/60
  
        print("Model train time = %f minutes" % train_time)
        print("Overall Accuracy = %.2f%% (+/- %.2f%%)" % (np.mean(cv_scores), np.std(cv_scores)))

    

    def predict(self,padded_docs):
        
        print ("Predicting..\n" )
        pred = self.model.predict(padded_docs)
        
        #Get back original classes from encoded values
        prediction = self.labelBinarizer.inverse_transform(pred)
        
        return prediction
    
    def plot_model_history(self,model_history):
        fig, axs = plt.subplots(1,2,figsize=(15,5))
        # summarize history for accuracy
        axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
        axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
        axs[0].set_title('Model Accuracy')
        axs[0].set_ylabel('Accuracy')
        axs[0].set_xlabel('Epoch')
        axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
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


    def saveModel (self,path):
        
        print ("Saving model..\n")
        
        model_path = path + "_model.h5"
        self.model.save(model_path)  
       
        tokenizer_path = path + "_tokenizer.pickle"
        with open (tokenizer_path,'wb') as f:
            pickle.dump(self.tokenizer,f)
    
        lb_path = path + "_labelBinarizer.pickle"
        with open (lb_path, 'wb') as f:
            pickle.dump(self.labelBinarizer,f)
            
        params_csv = path + '_params.csv'
        out_df = pd.DataFrame({'max_sentence_length':self.max_sentence_length,'vocab_size':self.vocab_size},\
                              index = [0])
        out_df.to_csv (params_csv,index=False)
    
    def loadModel (self,path):

        print ("Loading model..\n")
        
        model_path = path + "_model.h5"
        self.model = load_model(model_path)

                   
        tokenizer_path = path + "_tokenizer.pickle"
        with open (tokenizer_path,'rb') as f:
            self.tokenizer = pickle.load(f)
        
        lb_path = path + "_labelBinarizer.pickle"
        with open (lb_path, 'rb') as f:
            self.labelBinarizer= pickle.load(f)
            
        params_csv = path + '_params.csv'
        in_df = pd.read_csv(params_csv)
        
        self.vocab_size = np.int(in_df.vocab_size)
        self.max_sentence_length = np.int(in_df.max_sentence_length)
        
        
        
    
    
  
