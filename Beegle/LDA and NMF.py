from __future__ import print_function
from time import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

# Read in the data
reviews = pd.read_csv('home_products_additional_features.csv', header=0, encoding="ISO-8859-1" )

def run_nmf(nmf_features, nmf_topics, nmf_top_words, nmf_data_samples, nmf_max_df, nmf_min_df, nmf_alpha, nmf_l1_ratio):
    print("Extracting tf-idf features for NMF...")
    tfidf_vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df,
                                       max_features=n_features,
                                       stop_words='english')

    tfidf = tfidf_vectorizer.fit_transform(nmf_data_samples)

    nmf = NMF(n_components=n_topics, random_state=1,
              alpha=alpha, l1_ratio=l1_ratio).fit(tfidf)

    print("\nTopics in NMF model:")
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    print_top_words(nmf, tfidf_feature_names, n_top_words)
	
def run_lda(lda_features, lda_topics, lda_top_words, lda_data_samples, lda_max_df, lda_min_df, lda_max_iter, lda_learning_offset):
    tf_vectorizer = CountVectorizer(max_df=lda_max_df, min_df=lda_min_df,
                                    max_features=lda_features,
                                    stop_words='english')
    
    tf = tf_vectorizer.fit_transform(lda_data_samples)
    
    lda = LatentDirichletAllocation(n_topics=lda_topics, max_iter=lda_max_iter,
                                    learning_method='online',
                                    learning_offset=lda_learning_offset,
                                    random_state=0)
    
    lda.fit(tf)
    
    print("\nTopics in LDA model:")
    tf_feature_names = tf_vectorizer.get_feature_names()
    print_top_words(lda, tf_feature_names, lda_top_words)    
	
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()
	
#////////////////////////SET NMF PARAMETERS AND RUN MODEL/////////////////////////////

# Uncomment one of the lines (and only one line) that begins with nmf_data_samples to change the data set the model runs on

#nmf_data_samples = list(reviews['Text'])
#nmf_data_samples = list(reviews['Title'])
#nmf_data_samples = list(reviews['text_and_title'])
#nmf_data_samples = list(reviews['double_title'])
#nmf_data_samples = list(reviews['text_and_title_no_stops'])
nmf_data_samples = list(reviews['double_title_no_stops'])
#nmf_data_samples = list(reviews['text_and_title_negation'])
#nmf_data_samples = list(reviews['double_title_negation'])
#nmf_data_samples = list(reviews['text_and_title_negation_no_stops'])
#nmf_data_samples = list(reviews['double_title_negation_no_stops'])
#nmf_data_samples = list(reviews['lemma_text_title_no_stops'])
#nmf_data_samples = list(reviews['lemma_double_title_no_stops'])

nmf_features = 10000         # Size of the vocabulary
nmf_topics = 10              # Number of topics
nmf_top_words = 20           # Words to include in the topic
nmf_max_df=0.95              # Ignore terms that have a doc frequency (percent or int) strictly higher than the given threshold
nmf_min_df=2                 # Ignore terms that have a doc frequency (percent or int) strictly lower than the given threshold
nmf_alpha=.1                 # Constant that multiplies the regularization terms. Set to zero for no regularization.
nmf_l1_ratio=.5              # Regularization mixing parameter.  0 <= l1_ratio <= 1

run_nmf(nmf_features, nmf_topics, nmf_top_words, nmf_data_samples, nmf_max_df, nmf_min_df, nmf_alpha, nmf_l1_ratio)


#////////////////////////SET LDA PARAMETERS AND RUN MODEL/////////////////////////////

# Uncomment one of the lines (and only one line) that begins with lda_data_samples to change the data set the model runs on

#lda_data_samples = list(reviews['Text'])
#lda_data_samples = list(reviews['Title'])
#lda_data_samples = list(reviews['text_and_title'])
#lda_data_samples = list(reviews['double_title'])
#lda_data_samples = list(reviews['text_and_title_no_stops'])
lda_data_samples = list(reviews['double_title_no_stops'])
#lda_data_samples = list(reviews['text_and_title_negation'])
#lda_data_samples = list(reviews['double_title_negation'])
#lda_data_samples = list(reviews['text_and_title_negation_no_stops'])
#lda_data_samples = list(reviews['double_title_negation_no_stops'])
#lda_data_samples = list(reviews['lemma_text_title_no_stops'])
#lda_data_samples = list(reviews['lemma_double_title_no_stops'])

lda_features = 10000         # Size of the vocabulary 
lda_topics = 10              # Number of topics
lda_top_words = 20           # Words to include in the topic
lda_max_df= 0.95             # Ignore terms that have a doc frequency (percent or int) strictly higher than the given threshold
lda_min_df= 2                # Ignore terms that have a doc frequency (percent or int) strictly lower than the given threshold
lda_max_iter=5               # Number of iterations to compute
lda_learning_offset=50.      # A parameter that downweights early iterations in online learning. Should be > 1

run_lda(lda_features, lda_topics, lda_top_words, lda_data_samples, lda_max_df, lda_min_df, lda_max_iter, lda_learning_offset)