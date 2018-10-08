import spacy
from spacy import displacy
from spacy.matcher import Matcher

### INITIALIZE ALL THE PATTERNS ###
allPatterns = {}

# optional subject + modifiers
adjNoun = [
    {"DEP": "amod", "OP":"?"},
    {'DEP': 'nsubj', "OP":"?"},
    {'DEP': 'nsubjpass', "OP":"?"}
]

# verb with its optional modifiers
advVerb = [
    {'DEP': 'aux', 'OP':'?'},
    {'DEP': 'auxpass', 'OP':'?'},
    {'DEP': 'neg', 'OP':'?'},
    {'DEP': 'advmod', 'OP':'?'},
    {'POS': 'VERB'},
    {'DEP': 'advmod', 'OP':'?'}
]

# optional pre-modifiers of object
preModObj = [
    {"DEP": "det", "OP":"?"},
    {"DEP": "amod", "OP" : "?"},
    {"DEP": "poss", "OP":"?"},
    {"DEP": "compound", "OP" : "?"},
    {"DEP": "compound", "OP" : "?"}
]

# optional post-modifiers of object
postModObj = [
    {"DEP": "amod", "OP":"?"},
    {"DEP": "advmod", "OP" : "?"}
]

# direct object with its modifiers
adjDObj = []
adjDObj.extend(preModObj)
adjDObj.append({"DEP": "dobj"})
adjDObj.extend(postModObj)

# prepositional object with its modifiers
adjPObj = []
adjPObj.extend(preModObj)
adjPObj.append({"DEP":"pobj"})
adjPObj.extend(postModObj)

# prep-complement
pComp = [
    {"DEP": "acomp", "OP":"?"},
    {"DEP": "prep"},
    {"DEP":"pcomp"}
]

#oneMatchOpt = [{'LEMMA': '', 'OP' : '?'}]
#allMatch = [{'IS_ASCII': True, 'OP' : '*'}]
#allMatchNoNoun = [{'IS_ASCII': True, 'OP' : '*'}, {'POS': 'NOUN', 'OP' : '!'}]
allMatchNoVerb = [{'IS_ASCII': True, 'OP' : '*'}, {'POS': 'VERB', 'OP' : '!'}]

# I like this smaller version
# Basic SVO
pattern0 = []
pattern0.extend(adjNoun)
pattern0.extend(advVerb)
pattern0.extend(adjDObj)
allPatterns.__setitem__("(S)VO", pattern0)

# I cannot voice displeasure with product directly
# Basic SVPO
pattern1 = []
pattern1.extend(adjNoun)
pattern1.extend(advVerb)
pattern1.extend(allMatchNoVerb)
pattern1.extend(adjPObj)
allPatterns.__setitem__("(S)VPO", pattern1)

# Alexa is pretty dumb but the package overall is good
# Handle A-Comp
pattern2 = []
pattern2.extend(adjNoun)
pattern2.extend(advVerb)
pattern2.append({"DEP": "acomp"})
allPatterns.__setitem__("(S)VC", pattern2)

# She is handy without taking up much room.
# Handle P-Comp
pattern3 = []
pattern3.extend(advVerb)
pattern3.extend(pComp)
pattern3.extend(allMatchNoVerb) # longer gap between pcomp and dobj
pattern3.extend(adjDObj)
allPatterns.__setitem__("VPCO", pattern3)

# job is to turn off light
# Handle X-Comp
pattern4 = []
pattern4.extend(adjNoun)
pattern4.extend(advVerb)
pattern4.extend([{}, {"DEP": "xcomp"}])
pattern4.extend(allMatchNoVerb) # longer gap between xcomp and dobj
pattern4.extend(adjDObj)
allPatterns.__setitem__("SVXO", pattern4)

### FINISHED INITIALIZATION OF ALL SPANS ###


# run all patterns against the doc
def getKeyPhraseSpans(doc):
    spans = []
    for (id, pattern) in allPatterns.items():
        matcher = Matcher(nlp.vocab)
        matcher.add(id, None, pattern)
        matches = matcher(doc)
        for match_id, start, end in matches:
            spans.append((start, end))
            #string_id = nlp.vocab.strings[match_id]  # get string representation
            #span = doc[start:end]  # the matched span
            #print(string_id, start, end, span.text)

    # remove all subsumed spans
    spans = [item for item in spans if not subsumed(spans, item)]
    return spans

# check if item is subsumed (fully covered) by another span in spans
def subsumed(spans, item):
    for span in spans:
        if (span==item):
            continue
        if (span[0]<=item[0] and span[1]>=item[1]):
            return True
    return False

nlp =  spacy.load('en_core_web_sm')

doc = nlp(u'She has been demoted by my google home mini and I like this smaller version. '
          u'I cannot voice with product directly. '
          u'She is handy without taking up much room.'
          u'Alexa is pretty dumb but the package overall is good. '
          u'Her job is to turn off the light. ')

for (start, end) in getKeyPhraseSpans(doc):
    span = doc[start:end]
    print(span.text)