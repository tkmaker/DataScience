# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 23:02:39 2018

@author: takalyan
"""

import pandas as pd

# Import various modules for string cleaning
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from afinn import Afinn
from textblob import TextBlob


import textacy
import spacy
import numpy as np

data_df = pd.read_csv('../data/amazon_echo_10pgs.csv')
nlp = spacy.load('en_core_web_sm')



def cleanup_review( review,stop_words=False,stemmer=False,lemma=False):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # Remove HTML
    words = BeautifulSoup(review,"lxml").get_text()


    #Remove special chars in this scrape output for reviews
    #reviews are contained in ['...']
    words = re.sub("(\[\')|(\'\])","", words) 

    # Remove non-letters. Keep ' and .
    words = re.sub("[^a-zA-Z.']"," ", words) 
    
      
    if (re.match('[.]$',words) ) : 
        #print (words)
        words = words
    else: 
        words = words + "."
    #
    
    #return
    nomarkup  = words

    # Convert words to lower case and split them
    #words= words.lower()
    
    if lemma:
        lemmatizer = WordNetLemmatizer()
        words = lemmatizer.lemmatize(words)
        
    #tokenize
    words = words.split()
    #
    # .remove punctuation from each word
    #table = str.maketrans('', '', string.punctuation)
    #words = [w.translate(table) for w in words]

    # Optionally remove stop words (false by default)
    if stop_words:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    

    #stemming of words
    if stemmer:
        porter = PorterStemmer()
        words = [porter.stem(word) for word in words]

   

    clean_review = "".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in words]).strip()
    
 
          
    return(clean_review,nomarkup)

i=0
j=0

def get_vader_sentiment(text,threshold=0.1,verbose=False):
    
        # analyze the sentiment for review
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    
   
    
    # get aggregate scores and final sentiment
    agg_score = scores['compound']
    
    #final_sentiment:
    # 1 'positive' if agg_score > 0
    # 0 neutral 
    #-1 negative 
            
    if (agg_score >= threshold):
        sentiment = 1
    elif (agg_score < threshold and agg_score >= 0) :
        sentiment = 0
    else :
        sentiment = -1
        
    if (verbose and i <= word_print_count) :
        print(text)
        print ("Vader Sentiment:")
        print(scores,"\nDetected sentiment:%s\n"%sentiment)
   
    return sentiment




def get_afinn_sentiment(text,verbose=False):
    
    afinn = Afinn()

    afinn_score = afinn.score(text)
    
    
    #Score is negative for negative reviews and positive for positive reviews
    if (afinn_score > 0):
        sentiment = 1
    elif (afinn_score == 0) :
        sentiment = 0
    else : 
        sentiment = -1
    
    if (verbose and j <= word_print_count):
            print ("Afinn sentiment:")
            print (afinn_score,"\n","Detectec sentiment:%i\n"%sentiment)
            
    return sentiment

def get_textblob_sentiment (text,verbose=False):
    
    textblob_score = TextBlob(text).sentiment
    
  
    if (textblob_score.polarity > 0):
        sentiment = 1
    elif (textblob_score.polarity == 0) :
        sentiment = 0
    else: 
        sentiment = -1
    
    if (verbose and j <= word_print_count):
            print (text)
            print ("Textblob sentiment:")
            print (textblob_score.polarity,"\n","Detected sentiment:%i\n"%sentiment)
    
    
    return sentiment

from nltk.corpus import sentiwordnet as swn
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer


tag_map = {'NN':'n','RB':'r','VB':'v','VBZ':'v','JJ':'a'}

def get_sentiWordNet_sentiment (text,verbose=False):
    
    
    text = re.sub("\'","",text)
    
    #lemmatozer to prevent sentiwordnet fails
    lemmatizer = WordNetLemmatizer()
    text = lemmatizer.lemmatize(text)
    
    #tokens = word_tokenize(text) # Generate list of tokens
    words = word_tokenize(text)
    
    #porter = PorterStemmer()
    #stemmed = [porter.stem(word) for word in words]
    
    #get wordNet POS       
    tokens_pos = pos_tag(words) 
    
    pos_score = 0
    neg_score = 0
    neut_score = 0
    
    for word,pos in tokens_pos:
        #skip redundant POS
        if (re.search('[^\.|PRP|PRP\$|CC]',pos)):
            print (word,pos)
            #create sentiwordnet string of format "word.POS.01"
            swn_string =  word + "." + tag_map[pos] + ".01"
            print (swn_string)
            pos_score = pos_score + swn.senti_synset(swn_string).pos_score()
            neg_score = neg_score + swn.senti_synset(swn_string).neg_score()
            neut_score = neut_score + swn.senti_synset(swn_string).obj_score()
        
    print (pos_score,neg_score,neut_score)
    
    if (pos_score > neg_score and pos_score > neut_score):
        sentiment = 1
    elif (neg_score > pos_score and neg_score > neut_score):
        sentiment = -1
    elif (neut_score > pos_score and neut_score > neg_score):
        sentiment = 0
    
    if (verbose and j <= word_print_count):
            print (text)
            print ("SentiwordNet sentiment:\n")
            print ("Detected sentiment:%i\n"%sentiment)
            
    return sentiment
        
reviews = ""



#Iterate through dataframe of reviews.
#Each row is a review
#Create one large corpus of all reviews
for index,row  in data_df.iterrows():
    clean_review,nomarkup_review = cleanup_review(row.review_body)
    reviews = reviews + clean_review 

out = {'review':[reviews]}
out_df = pd.DataFrame(data=out)
out_df.to_csv('../data/clean_review.csv')

#Create spacy doc of reviews
#doc = textacy.Doc(reviews,lang=u'en_core_web_sm')
spacy_doc = nlp(reviews)

pos_reviews = ""
neg_reviews = ""
neut_reviews = ""
word_print_count = 500

#Iterate through sentences in doc and get sentiment
for sentence in spacy_doc.sents :
    
    #extract sentence string for doc sent
    sentence_str = sentence.string.strip()
    
    #Add period to end of each sentence if not present
    #we will be appending all sentences 
    if (re.search('[^\.]$',sentence_str)) :
        sentence_str = sentence_str + "."
        
    #get sentiment using different lexical algorithms
    #vader_sentiment = get_vader_sentiment(sentence_str,0.1,True)
    #afinn_sentiment = get_afinn_sentiment(sentence_str,True)
    textblob_sentiment = get_textblob_sentiment(sentence_str,False)
    #sentiWord_sentiment  = get_sentiWordNet_sentiment(sentence_str,True)
    #increment counters for printing
    i=i+1
    j=j+1
    
    
    #assign sentences to positive/negative/neutral word groups
    if (textblob_sentiment == 1) :
        pos_reviews = pos_reviews + sentence_str
    elif (textblob_sentiment == 0) :
        neut_reviews = neut_reviews + sentence_str
    else :
        neg_reviews = neg_reviews + sentence_str
    
    #output review groups to csv via a dataframe
    out = {'pos':[pos_reviews],'neg':[neg_reviews],'neut':[neut_reviews]}
    out_df = pd.DataFrame(data=out)
    out_df.to_csv('../data/review_groups.csv')
        
    