# Key Phrase Extractor

This code goes over an example of how to extract key positive and negative phrases from reviews. The example is based on extracting phrases based on 100 pages of Amazon Echo reviews.  

Python notebook describing the usage is in https://github.com/tkmaker/DataScience/blob/master/PhraseExtractor/examples/GetKeyPhrases.ipynb

The high level flow of key phrase extraction is as follows:

1) Convert the reviews into a Spacy doc and use user defined Spacy Matcher patterns to extract phrases
2) Group similar phrases from the extracted phrases using user defined similarity scorers
3) Aggregate scores for similar phrases and also split phrases into positive and negative phrases based on user defined sentiment scores. Return a list of ranked positive and negative phrases 
4) Display key positive and negative phrases along with the aggregate scores and the pattern that matched
