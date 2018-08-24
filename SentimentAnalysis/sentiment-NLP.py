
# coding: utf-8

# In[9]:


import pandas as pd
# Import various modules for string cleaning
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer

# Read data from files 
train_df = pd.read_csv( "labeledTrainData.tsv", header=0, 
 delimiter="\t", quoting=3 )
train_df.head()


# In[ ]:


#The column sentiment indicated whether it is a positive (1) or a negative (0) review
#Even though this is a labeled dataset, we will predict the sentiment using an unsupervised approach


# In[43]:


#Lets look at one review
print (train_df.review[1])


# In[44]:


def cleanup_review ( review,stop_words=False,stemmer=False,lemma=False):
    # Function to clean text in a document and return a sentence for sentiment analysis
    #
    # Remove HTML
    review_text = BeautifulSoup(review,"lxml").get_text()
    #  
    # Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    #
    # Convert words to lower case 
    words= review_text.lower()
    
    #Optionally lemmatize words (false by default)
    if lemma:
        lemmatizer = WordNetLemmatizer()
        words = lemmatizer.lemmatize(words)
        
    #tokenize words
    words = words.split()
    
    # .remove punctuation from each word
    table = str.maketrans('', '', string.punctuation)
    words = [w.translate(table) for w in words]

    # Optionally remove stop words (false by default)
    if stop_words:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    

    #Optionally stem words (false by default)
    if stemmer:
        porter = PorterStemmer()
        words = [porter.stem(word) for word in words]

   
    #Recombine words into a sentence
    sentence = "".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in words]).strip()
    
    return(sentence)


# In[45]:


#NLTK VADER for sentiment detection
def analyze_sentiment_vader_lexicon(review, 
                                    threshold=0.1):
    
    # analyze the sentiment for review
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(review)
    
    
    # get aggregate scores and final sentiment
    agg_score = scores['compound']
    final_sentiment = 'positive' if agg_score >= threshold  else 'negative'
    return final_sentiment


# In[46]:


reviews = train_df.review[0:5]

sentiment = []

for review in reviews:
    clean_review = cleanup_review(review,stop_words=True,                                      stemmer=True,lemma=True)
    sentiment.append(analyze_sentiment_vader_lexicon(clean_review,threshold=0.5))
    


# In[47]:


print ("Actual=%s,  Predicted=%s" % (train_df.sentiment[1],sentiment[1]))


# In[48]:


#Predicted value matches actual. Further optimization can be done to adjust the threshold to improve accuracy

