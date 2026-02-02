''' (updated 28 Dec 2025)
This module provides functionality for topic modeling using LDA (latent dirichlet allocation) from Gensim. 
Functionality for computations, mapping back to the train/test data, and visualizations is included.
**Best to import in a venv because of gensim's dependencies**
Refactor into a class, particularly for class attributes
'''

import sys
import os
import numpy as np
import gensim
import nltk
from nltk.corpus import stopwords
import pandas as pd
import re

import gensim.corpora as corpora
#from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import gensim, logging, warnings
#from gensim.utils import lemmatize, simple_preprocess

try:
    import pyLDAvis.gensim
    print("Successfully imported pyLDAvis")
    intertopic_map = True
except:
    print("Could not import pyLDAvis")
    intertopic_map = False

import warnings
warnings.filterwarnings("ignore")



# Download NLTK stopwords
nltk.download('stopwords')
stop_words = stopwords.words('english')

# automate data preparation
def automate_corpus_prep(df, raw_col):
    
    data = df.copy()
    data['corpus_clean'] = data[raw_col].apply(preprocess_text)
    print("preprocessed...")
    
    data['tokens'] = data['corpus_clean'].apply(tokenize)
    print("tokenized...")

    ################### DEV
    ##### add functionality to remove from training/data where the corpus_clean is non-descriptive
    
        # add topic_title column to fill these
#    data['topic_title'] = np.nan
        
        # filter out these rows
    
        # assign topic as 'other'
    
    
    
    corpus_data = data # placeholder - in DEV
################### ^ DEV    

    # skip lemmatization
    
    id2word = corpora.Dictionary(corpus_data['tokens']) # rename id2word as corpus_dictionary
    texts = corpus_data['tokens']
    corpus = [id2word.doc2bow(text) for text in texts]
    print("dictionary and corpus created")
    
    #from gensim.corpora.dictionary import Dictionary # another way to create id2word
    #train_dictionary = Dictionary(list(corpus_data['tokens']))
    
    
    return(data, corpus_data, corpus, id2word) 

def identify_best_model(n_topics_range = (1, 5)):
    
    print("IN DEV")
    print("loop through each num_topics and calculate the highest coherence score.")
    print("output analysis of results")
    
    return(best_model)
    


# preprocess
def preprocess_text(text):
    text = re.sub('\s+', ' ', text)  # Remove extra spaces
    text = re.sub('\S*@\S*\s?', '', text)  # Remove emails
    text = re.sub('\'', '', text)  # Remove apostrophes
    text = re.sub('[^a-zA-Z]', ' ', text)  # Remove non-alphabet characters
    text = text.lower()  # Convert to lowercase
    return text

# tokenize
def tokenize(text):
    tokens = gensim.utils.simple_preprocess(text, deacc=True)
    tokens = [token for token in tokens if token not in stop_words]
    return tokens


# examine topics
def topics_EDA(model, corpus_data, id2word, num_topics, num_words = 10, suppress_top_words = False):
    
    print("="*80)
    print(f"EDA on the LDA model - {num_topics} topics were assigned\n", '='*80)
    
    # get coherence score
    coherence_model_lda = CoherenceModel(model=model, texts=corpus_data['tokens'], dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print(f"Coherence score when assigning {num_topics} topics:")
    print(f'\t{coherence_lda}\n')
    
    topics = model.print_topics(num_words = num_words)
    
    # title for each topic
    topic_names = get_titles(model)
    print("Most probable title for each topic, based on probability distributions for each topic [look into the probability scores from LDA]\n")
    for title_id, values in topic_names.items():
        print(f'\t{title_id}:    {values}')
    #print(f"\t{topic_names}")
    
    if (not suppress_top_words):
        # top words in each topic
        topics = model.print_topics(num_words = num_words)
        print('\n', '='*80, f"\nAnalysis of each topic and the {num_words} most influential words")
        for topic in topics:
            print(f'\t{topic}\n')
        
          

# get title for each topic
def get_titles(model):
          
    topics = model.print_topics(num_words = 10) ####
    
    # get top n-keywords - we probably only want the top 1
    topic_names = {}

    for topic in topics:
        topic_id = topic[0]
        kwd_str = topic[1]
        kwd_list = [s.lstrip().rstrip() for s in kwd_str.split('+')]

        #### can add param to specify more than 1 top words if desired
        top_word = kwd_list[0].split('*')
        top_word_probs = top_word[0] # the probability represents probability that the word belongs to the topic? representativeness vs other words in toipc?
        top_word = top_word[1]

        topic_names[topic_id] = {'title': top_word,
                                 'probability': top_word_probs}

    return(topic_names)
    
    

# map assigned topics to data
def map_topics_to_df(model, df, model_results):
    ######## this needs to account for filtered rows that were assigned "other",
        #### when adding in that functionality
    

    topic_names = get_titles(model)
    #corpus_with_topics = model[corpus]
    
    topic_ids = []
    topic_titles = []

    #for corpus_ind in range(len(corpus_with_topics)):
    for corpus_ind in range(len(model_results)):

        raw_probabilities = [p[1] for p in model_results[corpus_ind][0]]

        index_max = np.argmax(raw_probabilities)
        topic_id = index_max # id corresponds to the index in the list
        topic_title = topic_names[topic_id]['title']

        topic_ids.append(topic_id)
        topic_titles.append(topic_title)

    assigned_df = df.copy()
    assigned_df['topic_title'] = topic_titles
    assigned_df['topic_id'] = topic_ids
          
    return(assigned_df)

# visualize
def vis_intertopic_distance():
        
    pyLDAvis.enable_notebook()
    vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary=lda_model.id2word)
    
    vis
    
    return(vis)

# vis - others - scatterplot of topics
