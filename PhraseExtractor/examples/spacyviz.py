import spacy
from spacy import displacy

nlp =  spacy.load('en_core_web_sm')

doc = nlp(u'She has been demoted by my Google home mini. ')

displacy.serve(doc, style='dep')