import pandas as pd
pd.options.display.max_columns = 50

import re

import nltk
from nltk.corpus import stopwords # Import the stop word list
#nltk.download()  # Download text data sets, including stop words

from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.tokenize import TreebankWordTokenizer
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *

from textblob import TextBlob


def text_and_title(text):
    """Add the title information to the beginning of the text"""
    try:
        x = text['Title']
        title = x + '. '
    except:
        title = 'NO_TITLE'
    try:
        result = title + ' ' + text['Text']
    except:
        result = title
    return (result)

def add_double_title(text):
    """Add the title information to the beginning of the text (and double it, called 'weighting up')"""
    try:
        x = text['Title']
        title = x + '. ' + x + '. '
    except:
        title = 'NO_TITLE'
    try:
        result = title + ' ' + text['Text']
    except:
        result = title
    return (result)
	
	
def letters_only(text, field):
    """Replace all non-alphanumeric characters with a space"""
    try:
        x = re.sub("[^a-zA-Z0-9]",    # The pattern to search for
                   " ",               # The pattern to replace it with
                   text[field] )      # The text to search
    except:
        return ('byte_code_error_ignore_this_ record')
    return (x.lower())

def remove_stop_words(text, field, stopwords_set):
    """Remove stop words from the review text"""
    words = [w for w in text[field].split() if not w in stopwords_set]
    return( " ".join( words ))

def get_wordnet_pos(treebank_tag):
    """This is a helper function to translate part of speech (POS) for us in the make_lemmas function.
    nltk uses pos_tag to determine the POS of a word that is not compatible with the wordnet_lemmatizer."""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def make_lemmas(text, field, stopwords_set):
    """Toeknizes all words in the review and then tags them with the part of speech (POS) they belong to
    as a tuple. Each tuple (word, pos) is then lemmatized before stop words are removed and the list is
    joined back into a single item/doc"""
    x = word_tokenize(text[field])
    #x = word_tokenize(x)
    x = nltk.pos_tag(x)
    doc = []
    for word, part in x:
        doc.append(wordnet_lemmatizer.lemmatize(word, pos=get_wordnet_pos(part)))
    words = [w for w in doc if not w.lower() in stopwords_set]
    x = ( " ".join( words ))
    return(x)

def create_neg_stops():
    """Combine the original list of stop words the negative suffix. For example: 'his', 'they', and 'me' become
    'his_neg', 'they_neg', and 'me_neg'."""
    orig_stops = stopwords.words("english")
    neg_stops = []
    for i in orig_stops:
        neg_stops.append(i+'_neg')
    orig_stops.extend(neg_stops)
    return(orig_stops)

def remove_negated_stop_words(text, field, neg_stops):
    """Remove all instances of stop words that have the '_neg' suffix."""
    # Make the text lowercase
    x = text[field]
    x = x.lower()
    
    stopwords_set = set(neg_stops)
    
    # List comprehension that splits the review into words and removes negative stop words
    words = [w for w in text[field].split() if not w in stopwords_set]
    return( " ".join( words ))

#/////////////////////ACCOUNT FOR NEGATION SENTIMENT///////////////////////////////////
def negatize(text, field):
    """Use the NLTK library's mark_negation to find negative words (like 'not' and 'nor') and append
    the '_neg' suffix to all words following the first negative word until it encounters a period or comma.
    Example: 'I don't like eating pizza, I love eating pizza' 
    becomes 'I don't like_neg eating_neg pizza_neg, I love pizza' """
    x = text[field]
    
    # The TextBlob class provides an easy way to split the reviews into sentences
    x = TextBlob(x)
    
    piece = []
    for sentence in x.sentences:
        
        # Split sentence on commas to ID phrases that need to be negated (if required)
        part = re.split(', ',str(sentence))
        for i in part:
            piece.append(mark_negation(i.split()))
    
    # Combine all terms/phrases back to one doc
    total = []
    for terms in piece:
        total.append(" ".join(terms))
    review = ''
    for phrase in total:
        review += phrase + ' '
    
    # mark_negation adds the _NEG suffix after the period, this catches those and fixes it
    review = review.replace("._NEG","_NEG.")
    review = review.lower()
    
    # return the entire entire review except for the last character which is always a space
    return (review[:-1])
	
	
from time import time
t0 = time()
t1 = time()
#/////////////////////READ THE DATA////////////////////////////////////////////////////
print ('Reading the data...')
wipes = pd.read_csv("home_products.csv", header=0, encoding="ISO-8859-1" )
print("Finished In:     %0.3fs." % (time()-t1))

#/////////////////////ADD THE TITLE TO THE TEXT////////////////////////////////////////
t1 = time()
print ('Adding titles to text...')
wipes['text_and_title'] = wipes.apply(lambda text: text_and_title(text), axis=1)
wipes['double_title'] = wipes.apply(lambda text: add_double_title(text), axis=1)
print("Finished In:     %0.3fs." % (time()-t1))

#/////////////////////REMOVE NON-ALPHANUMERICS AND PUNCTUATION/////////////////////////
t1 = time()
print ('Removing Non-Alphanumerics...')
wipes['text_and_title_no_stops'] = wipes.apply(lambda text: letters_only(text, 'text_and_title'), axis=1)
wipes['double_title_no_stops'] = wipes.apply(lambda text: letters_only(text, 'double_title'), axis=1)
print("Finished In:     %0.3fs." % (time()-t1))

#/////////////////////LOAD THE STOPWORDS PROVIDED BY NLTK//////////////////////////////
stopwords_set = set(stopwords.words("english"))

#/////////////////////REMOVE STOP WORDS AND PUNCTUATION////////////////////////////////
t1 = time()
print ('Removing Stop Words...')
wipes['text_and_title_no_stops'] = wipes.apply(lambda text: remove_stop_words(text, 'text_and_title_no_stops', stopwords_set), axis=1)
wipes['double_title_no_stops'] = wipes.apply(lambda text: remove_stop_words(text, 'double_title_no_stops', stopwords_set), axis=1)
print("Finished In:     %0.3fs." % (time()-t1))

#/////////////////////NEGATE TEXT AND TITLES///////////////////////////////////////////
t1 = time()
print ('Tagging Negative Text...')
wipes['text_and_title_negation'] = wipes.apply(lambda text: negatize(text, 'text_and_title'), axis=1)
wipes['double_title_negation'] = wipes.apply(lambda text: negatize(text, 'double_title'), axis=1)
print("Finished In:     %0.3fs." % (time()-t1))

#/////////////////////REMOVE NEGATIVE STOPS////////////////////////////////////////////
t1 = time()
print ('Removing Negative Stop Words...')
wipes['text_and_title_negation_no_stops'] = wipes.apply(lambda text: remove_negated_stop_words(text, 'text_and_title_negation', create_neg_stops()), axis=1)
wipes['double_title_negation_no_stops'] = wipes.apply(lambda text: remove_negated_stop_words(text, 'double_title_negation', create_neg_stops()), axis=1)
print("Finished In:     %0.3fs." % (time()-t1))

#/////////////////////LEMMATIZE THE TEXT REVIEWS///////////////////////////////////////
t1 = time()
print ('Lemmatizing The Text...')
wipes['lemma_text_title_no_stops'] = wipes.apply(lambda text: make_lemmas(text, 'text_and_title_no_stops', stopwords_set), axis=1)
wipes['lemma_double_title_no_stops'] = wipes.apply(lambda text: make_lemmas(text, 'double_title_no_stops', stopwords_set), axis=1)
print("Finished In:     %0.3fs." % (time()-t1))

#/////////////////////WRITE THE DATA///////////////////////////////////////////////////
wipes.to_csv('home_products_additional_features.csv', index=False)

print("FINISHED: \nTime Elapsed:    %0.3fs." % (time() - t0))
