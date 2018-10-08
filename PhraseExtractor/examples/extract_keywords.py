# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 13:34:16 2018

@author: Trushant Kalyanpur
"""

import spacy
from spacy.matcher import Matcher
import pandas as pd
from phraseExtractor import phraseExtractor

data_df = pd.read_csv('../data/clean_review.csv')
reviews = data_df.review[0]


nlp = spacy.load('en_core_web_sm')
#create spacy doc of reviews
reviews_doc = nlp(reviews)
#initialize matcher
matcher = Matcher(nlp.vocab)


### INITIALIZE ALL THE PATTERNS ###
allPatterns = {}

#create list of patterns to search using spacy Matcher
pattern1 = [{'POS': 'VERB','OP':'+'},{'POS': 'PART','OP':'+'}, \
            {'POS' : 'VERB','OP':'+'}, {'POS' : 'NOUN','OP':'+'}]

allPatterns.__setitem__("VPaVN", pattern1)


#pattern2 = [{'POS': 'PRON','OP':'+'}, {'POS' : 'VERB','OP':'+'},{'POS': 'PROPN','OP':'+'}]
#matcher.add('Keywords2', None, pattern2)

pattern2 =  [{'POS': 'ADJ','OP':'+'},{'POS': 'ADJ','OP':'+'},\
             {'POS' : 'NOUN','OP':'+'}]
allPatterns.__setitem__("AdjAdjN", pattern2)
 
pattern3 = [{'POS':'ADJ','OPJ':'+'},{'POS':'PART','OP':'+'},\
            {'POS' : 'VERB','OP':'+'},{'POS':'PART','OP':'*'}]
allPatterns.__setitem__("AdjPaVPa", pattern3)
 
 
pattern4 = [{'POS':'ADJ','OPJ':'+'},{'POS':'NOUN','OP':'+'},\
            {'POS' : 'ADP','OP':'+'},{'POS':'VERB','OP':'+'},{'POS':'NOUN','OP':'+'}]
allPatterns.__setitem__("AdjNAdpVN", pattern4)
 
#pattern6 = [{'DEP':'nsubj','OPJ':'+'},{'DEP':'ROOT','OP':'+'},{'DEP':'dobj','OP':'+'}]
#matcher.add('Keywords6', None, pattern6)

pattern5 = [{'POS': 'ADV','OP':'+'},{'POS': 'VERB','OP':'+'},\
            {'POS' : 'NOUN','OP':'+'}]
            #,{'IS_ASCII':True,'OP':'+'},{'IS_ASCII':True,'OP':'+'}]
allPatterns.__setitem__("AdvVN", pattern5)

pattern6 = [{'DEP': 'amod','OP':'+'},{'DEP': 'compound','OP':'+'},\
            {'POS' : 'NOUN','OP':'+'}]
allPatterns.__setitem__("AmCN", pattern6)

pattern7 = [{'DEP': 'aux','OP':'+'},{'DEP': 'neg','OP':'+'},\
            {'DEP' : 'ROOT','OP':'+'},{'DEP' : 'dobj','OP':'+'}]
allPatterns.__setitem__("AuxNegRoot", pattern7)

########################
#Define model and params
#########################

#Initialize model
phraseExtractor = phraseExtractor()

#Define printing verbosity
phraseExtractor.verbose = False

#choose how  many words to print per class
phraseExtractor.phrase_print_threshold =20 

#Choose similarity scorer  and threshold
phraseExtractor.sim_scorer = "jaccard"
phraseExtractor.sim_threshold = 0.85

#Choose sentiment scorer and threshold
phraseExtractor.sent_scorer = "keras"
phraseExtractor.sent_model_path = "kerasLSTM/kerasModelLSTM"
phraseExtractor.sent_threshold = 0

#Assign spacy doc and vocab to model
phraseExtractor.doc = reviews_doc
phraseExtractor.nlp = nlp 

#Attach current matcher to model
phraseExtractor.matcher = matcher
phraseExtractor.allPatterns = allPatterns

#get phrases 
phraseExtractor.summarize_reviews() 


#Debug only
#access extractor elements
phrase_matcher = phraseExtractor.phrase_matcher
similar_phrases = phraseExtractor.similar_phrases
positive_df = phraseExtractor.positive_df
negative_df = phraseExtractor.negative_df
counter = phraseExtractor.counter








