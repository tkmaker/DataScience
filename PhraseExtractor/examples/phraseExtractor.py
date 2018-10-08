# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 17:26:33 2018

@author: takalyan
"""


import pandas as pd
import numpy as np
import re

import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer


from collections import Counter
import textacy.keyterms
import spacy

#from textblob import Tex
from afinn import Afinn
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from kerasLSTM.kerasSentModel import kerasSentiment

data_df = pd.read_csv('../data/clean_review.csv')



#extract positive,negative,neutral reviews as string
reviews = data_df.review[0]






class phraseExtractor (object):
    
    def __init__ (self):
        self.verbose = False
        #Similarity scorer and threshold
        self.sim_scorer = "jaccard"
        self.sim_threshold = 0.5
        #Sentiment scorer and optinal threshold, model path
        self.sent_scorer = "textblob"
        self.sent_threshold = 0
        self.sent_model_path = ""
        #pre-trained keras sentiment model
        self.kerasModel= kerasSentiment()
        
        #Spacy doc
        self.doc = []
        self.nlp = [] 
        #Spacy matcher
        self.matcher = []
        #Matcher patterns
        self.allPatterns = {}
        #Dict to store matching phrases (key) and the matcher that matched (value)
        self.phrase_matcher = dict()
        #Dict to store keywords (key and similar phrases (value)
        self.similar_phrases = dict()
        #Collections counter to track scores
        self.counter = Counter()
        #defines how many phrases to prevent for each class
        self.phrase_print_threshold = 10
        #Dataframes for positive and negative keywords with scores
        self.positive_df = pd.DataFrame()
        self.negative_df = pd.DataFrame()
        
        
               
  
    # run all patterns against the doc
    def getKeyPhraseSpans(self):
        
        spans = []
        
        #Iterate through patterns dict
        for (id, pattern) in self.allPatterns.items():
            
            #add pattern and id to spacy matcher 
            self.matcher.add(id, None, pattern)
            
            #get list of matches  and append to spans array
            matches = self.matcher(self.doc)
            for match_id, start, end in matches:
                spans.append((start, end))

                string_id = self.nlp.vocab.strings[match_id]  # get string representation
                span = self.doc[start:end]  # the matched span
                self.phrase_matcher[span.text] = string_id

        # remove all subsumed spans
        spans = [item for item in spans if not self.subsumed(spans, item)]
        return spans

    # check if item is subsumed (fully covered) by another span in spans
    def subsumed(self,spans, item):
        for span in spans:
            if (span==item):
                continue
            if (span[0]<=item[0] and span[1]>=item[1]):
                return True
            return False

    #Function takes in spacy doc and returns counter with phrases and frequency
    def extract_matching_phrases (self):
    
        print ("Extacting matching phrases in doc..\n")
        
        for (start, end) in self.getKeyPhraseSpans():
            span = self.doc[start:end]
            #increment counter of current phrase
            self.counter[span.text] += 1
        
        
             
        if (self.verbose):
            print ("Spacy Matcher frequency counts:\n")
            #Find most common phrases from the counter
            for word, count in self.counter.most_common(100):
                if (count > 1) :
                    print("{0}: {1}".format(word, count))

   
                    
    #Funtion takes in a Collections counter as an input and finds all phrases that are similar
    #Returns a dict of phrases as keys and all similar phrases as values
    def find_similar_phrases(self):
    
        print ("Finding similar phrases..\n")
        
        #Convert collection counter to dict
        phrases_dict = dict(self.counter)
     
    
        #Iterate through phrases and find similar ones
        for phrase1,count1 in phrases_dict.items():
           
            #Clear list of similar items before each phrase loop
            similar_phrase_list = []
        
            #For each phrase above iterate through all other phrases
            #Find similar phrases and add matches to a list
            for phrase2,count2 in phrases_dict.items():
              
                text1 = self.cleanup_phrase(phrase1)
                text2 = self.cleanup_phrase(phrase2)
                
                sim_score = self.get_similarity_score(text1,text2)

                #Append to list only if score above threshold
                if (sim_score > self.sim_threshold) :
                    similar_phrase_list.append(phrase2)
            
            #Assign list of similar phrases to phrase we are iterating for
            self.similar_phrases[phrase1] = similar_phrase_list


    def get_sentiment (self, text, word_print_count=0):
        
        #Cleanup phrase before sentiment extraction
        text = self.cleanup_phrase(text)
        
        if self.sent_scorer == "afinn" : 
            sentiment = self.get_afinn_sentiment(text,word_print_count)
        elif self.sent_scorer == "vader" :
            sentiment = self.get_vader_sentiment(text,self.sent_threshold,word_print_count)
        elif self.sent_scorer == "textblob" : 
            sentiment = self.get_textblob_sentiment(text,word_print_count)
        elif self.sent_scorer == "keras":
            sentiment = self.get_keras_sentiment(text)
            
        return sentiment 

    def get_similarity_score (self,text1,text2):
          
        if (self.sim_scorer == "hamming") :
            sim_score = textacy.similarity.hamming(text1, text2)
        elif (self.sim_scorer == "levenshtein"):
            sim_score = textacy.similarity.levenshtein(text1, text2)
        elif (self.sim_scorer == "jaccard"):
            sim_score = textacy.similarity.jaccard(text1, text2)
        elif (self.sim_scorer == "jaro_winkler"):
            sim_score = textacy.similarity.jaro_winkler(text1, text2)
        else:
            sim_score = textacy.similarity.hamming(text1, text2)

        return sim_score
    
    def get_afinn_sentiment(self,text,word_print_count=0):
    
        afinn = Afinn()

        afinn_score = afinn.score(text)
    
    
        #Score is negative for negative reviews and positive for positive reviews
        if (afinn_score > 0):
            sentiment = 1
        elif (afinn_score == 0) :
            sentiment = 0
        else : 
            sentiment = -1
    
        j=0
        if (self.verbose and j <= word_print_count):
            print ("Afinn sentiment:")
            print (afinn_score,"\n","Detectec sentiment:%i\n"%sentiment)
            j+=1     
        return sentiment

    def get_vader_sentiment(self,text,word_print_count=0):
    
        # analyze the sentiment for review
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(text)
    
   
    
        # get aggregate scores and final sentiment
        agg_score = scores['compound']
    
        #final_sentiment:
        # 1 'positive' if agg_score > 0
        # 0 neutral 
        #-1 negative 
            
        if (agg_score >= self.sent_threshold):
            sentiment = 1
        elif (agg_score < self.sent_threshold and agg_score >= 0) :
            sentiment = 0
        else :
            sentiment = -1

        j = 0
    
        if (self.verbose and j <= word_print_count) :
            print(text)
            print ("Vader Sentiment:")
            print(scores,"\nDetected sentiment:%s\n"%sentiment)
            j + j+ 1
        return sentiment

    def get_textblob_sentiment (self,text,word_print_count=0):
        
        textblob_score = TextBlob(text).sentiment
    
  
        if (textblob_score.polarity > 0):
            sentiment = 1
        elif (textblob_score.polarity == 0) :
            sentiment = 0
        else: 
            sentiment = -1
    
        j = 0 
        if (self.verbose and j <= word_print_count):
            print (text)
            print ("Textblob sentiment:")
            print (textblob_score.polarity,"\n","Detected sentiment:%i\n"%sentiment)
            j += 1
    
        return sentiment

    
    def get_keras_sentiment (self,text):
        
        #Conver text to dataframe
        text_df = pd.DataFrame({'text':text},index=[0])
        
     
        self.kerasModel.text_df = text_df
        self.kerasModel.text_colname = 'text'
        
        #Get padded and encoded text
        padded_docs = self.kerasModel.encodeTestText()
        
        #Predict sentiment
        sentiment = self.kerasModel.predict(padded_docs)
        
        return sentiment
        
    def cleanup_phrase (self,phrase,lemma=True,stemmer=True,remove_stop_words=True):
    
        words = phrase.lower()

        # Remove non-letters. Keep ' and .
        words = re.sub("[^a-zA-Z]"," ", words) 
    
      
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
        if remove_stop_words:
            #stops = set(stopwords.words("english"))
            stops = ["the","it","to","and","is","so","as","but"]
            words = [w for w in words if not w in stops]
    

        #stemming of words
        if stemmer:
            porter = PorterStemmer()
            words = [porter.stem(word) for word in words]

   

        clean_phrase = "".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in words]).strip()
        # if (re.match(".+easy.+set",clean_phrase)):
        #    print (clean_phrase)
        return clean_phrase

    #iterates though dict of phrase and its list of similar phrases
    #returns positive and negative dataframe with columns [phrase], [aggregate score]
    def process_phrase_scores (self):
    
        print ("Processing phrase scores and aggregating..\n")

        pos_dict = dict()
        neg_dict = dict()
    
        for phrase, similar_phrases in self.similar_phrases.items():
        
        #if (phrase == 'very disappointing'):
         #   verbose = 1
        #else : 
         #   verbose = 0
            if (self.verbose):
                print ("----------------------------\n")
                print ("Processing base phrase %s\n"%phrase)
            
            #baseline score is of this phrase
            score = self.counter[phrase]
        
            #init sentiment counter
            senti_score = 0
        
            #Get sentiment of base phrase
            base_sentiment = self.get_sentiment(phrase)
        
            senti_score += base_sentiment
            
            #Now iterate through all similar phrases and add scores
            for sim_phrase in similar_phrases:
            
                #get sentiment for similar phrase
                current_sentiment = self.get_sentiment(sim_phrase)
            
                #add or subtract sentiment score 
                senti_score += current_sentiment
            
                #If current sentiment is neutral and :
                #      overall sentiment score is positive - subtract 1 to make mroe neg
                #      overal sentiment score is negative - add 1 to make more positive
                #if (current_sentiment == 0):
                #   if (senti_score < 0 ) :
                #      senti_score += 1
                # elif (senti_score > 0) :
                    #    senti_score -= 1
                    
                if (self.verbose):
                    print ("Similar phrase %s\n"%sim_phrase)
                    print ("Base sentiment = %i Current sentiment = %i" % \
                       (base_sentiment,current_sentiment))
                    print ("Senti Score = %d\n" % senti_score)
            
                #Only add scores if current sentiment matches base sentiment
                if (current_sentiment == base_sentiment):
                    score = score + self.counter[sim_phrase]
            
            #Add to positive or negative dict based on overall sentiment score
            if (senti_score > 0) :
                pos_dict[phrase] = score
            elif (senti_score < 0) :
                neg_dict[phrase] = score                
            
            #Create dataframe for positive and negative reviews.
            self.positive_df =  pd.DataFrame(list(pos_dict.items()),columns=["Phrase","Score"])
            self.negative_df =  pd.DataFrame(list(neg_dict.items()),columns=["Phrase","Score"])
            #Sort by descending scores
            self.positive_df =  self.positive_df.sort_values(by="Score",ascending=False)
            self.negative_df =  self.negative_df.sort_values(by="Score",ascending=False)
        
    


    #takes positive and negative dataframes of the format [phrase] [aggregate score]
    # prints keywords with score higher than threshold as long as they are not similar
    def return_keywords (self,phrase_scores_df,target_type):
    
    
        print ("Printing most used keywords..\n")


        #Dict to keep track of similar phrases to avoid printing duplicates
        similiar_phrase_tracker = dict()
        #track printed scores
        i = 0
        #Iterate through aggregate scores and only print unique phrases
        for index,row in phrase_scores_df.iterrows():

            if self.verbose:
                print ("processing:",row["Phrase"])
                
            #check to make sure current phrase is not added to list of similar phrases parsed before
            if row["Phrase"] not in similiar_phrase_tracker:
            
                #Display top n phrases
                if (i <= self.phrase_print_threshold) : 
                
                    #check if printing positive keywords and sentiment is positive
                    if (target_type == "positive" and self.get_sentiment(row["Phrase"])>0) :
                        print (row["Phrase"],":",row["Score"],self.phrase_matcher[row["Phrase"]])
                        i += 1
                    #check if printing negative keywords and sentiment is negative
                    elif (target_type == "negative" and self.get_sentiment(row["Phrase"])<0) :
                        print (row["Phrase"],":",row["Score"],self.phrase_matcher[row["Phrase"]])
                        i += 1
                    
                    #iterate through similar phrases for this phrase and add to tracker to avoid duplications
                    for sim_phrase in self.similar_phrases[row["Phrase"]] :
                        if self.verbose:
                            print ("adding %s to sim_phrase_dict\n"%sim_phrase)
                        similiar_phrase_tracker[sim_phrase] = 1
                    
                   

    #Top level summarizer that calls all other functions
    def summarize_reviews (self):
    
        #Load pre-trained model if we are using one
        if (self.sent_scorer == "keras"):
               self.kerasModel.loadModel(self.sent_model_path)
               
        #Extract matching phrases
        self.extract_matching_phrases()
        
        #Find phrases similar to matches
        self.find_similar_phrases()
    
        #Aggregate socres
        self.process_phrase_scores()
    
        #print keywords ommitting duplicates
        print ("Positive phrases\n")
        self.return_keywords(self.positive_df,"positive")
        print ("\nNegative phrases\n")
        self.return_keywords(self.negative_df,"negative")
        

