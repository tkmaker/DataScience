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
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
import textacy
import spacy


data_df = pd.read_csv('../data/amazon_echo_10pgs.csv')
nlp = spacy.load('en_core_web_sm')




def cleanup_review( review,stop_words=False,stemmer=False,lemma=False):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # Remove HTML
    words = BeautifulSoup(review,"lxml").get_text()

    words = re.sub("'",". ", words) 
    

    #  
    # Remove non-letters and , and .
    words = re.sub("[^a-zA-Z,.]"," ", words) 
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
    
    #Create spacy doc
    #doc = nlp(clean_review)
    
    
    #for token in doc:
     #   print(token.text, token.pos_, token.tag_, token.dep_)
    #for chunk in doc.noun_chunks:
     #   if (re.match('[a-zA-Z]+\s+[a-zA-Z]+',chunk.text)):
      #      print (chunk.text)
          
    return(clean_review,nomarkup)



pos_reviews = ""
neg_reviews = ""

#Iterate through dataframe of reviews.
#Each row is a review
for index,row  in data_df.iterrows():
    #Check for positive or negative review
    clean_review,nomarkup_review = cleanup_review(row.body)

    #Create overall group of positive and negative reviews
    if (re.search('(4|5).0 out of.*',row.rating)) : 
        pos_reviews = pos_reviews + " " + clean_review
        pos_reviews =  re.sub('\.\s*\.','.',pos_reviews)
    else : 
        neg_reviews = neg_reviews + " " + clean_review
        neg_reviews =  re.sub('\.\s*\.','.',neg_reviews)


#Create textacy doc of pos and negative reviews
doc_pos = textacy.Doc(pos_reviews,lang=u'en_core_web_sm')
doc_neg = textacy.Doc(neg_reviews,lang=u'en_core_web_sm')

#textacy textranking
print ("Textrank keywords in positive reviews:\n")
print (textacy.keyterms.textrank(doc_pos,normalize='lemma',n_keyterms=10))

print ("\nTextrank keywords in negative reviews:\n")
print (textacy.keyterms.textrank(doc_neg,normalize='lemma',n_keyterms=10))



#Textacy bag of terms with count weighting
bot = doc_pos.to_bag_of_terms(ngrams=2, named_entities=True, \
                          weighting='count',as_strings=True)

print ("\nTextacy bag of terms:\n")
print(sorted(bot.items(), key=lambda x: x[1], reverse=True)[:15])


#textacy pattern matching
pattern = r'<VERB>?<ADV>*<VERB>+'
matching_lists = textacy.extract.pos_regex_matches(doc_pos, pattern)

#controls display count
print_count = 20
i=0

print ("\nTextacy POS matching:\n")
for list in matching_lists:
    #Filter out single words
    if re.match('[a-zA-Z]*\s+[a-zA-Z]',list.text):
        print(list.text)
        i +=1
    if (i > print_count) :
        break
    