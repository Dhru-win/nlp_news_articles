# Dependencies 
import pandas as pd

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from gensim.similarities import MatrixSimilarity
from operator import itemgetter
from gensim import corpora
import spacy
import string
from spacy.lang.en.stop_words import STOP_WORDS
import gensim
import operator
import re
from fetch_data import flatten_and_frame


article_df = flatten_and_frame()

# checking for null values
article_df.info()

# getting rid of the trailing categories as they are not necessary. 
article_df = article_df.iloc[:, 0:9]

# basic descriptive statistics of the numeric data
article_df['word_count'].describe()

# dropping all the rows that have a word count of 0 or less
column = article_df.word_count
zero_words = column[column <= 0]
article_df.drop(zero_words.index, inplace=True)
article_df.reset_index(drop=True, inplace=True)

print(article_df.head(5))

# replacing all the null values with an empty string
article_df.fillna('', inplace=True)

# loading spacy
spacy_nlp = spacy.load('en_core_web_sm')
# Punctuations to be removed from corpus

punctuation = string.punctuation
punctuation
# Stop words to be removed from corpus
stop_words = spacy.lang.en.stop_words.STOP_WORDS
stop_words

def spacy_tokenizer(sentence):
    """ 
    This function cleans text and then tokenizes it
    """
    # removing single quotes 
    sentence = re.sub("'", "", sentence)

    # removing string containing numeric values
    sentence = re.sub('\w*\d\w*', '', sentence)

    # removing extra spaces
    sentence = re.sub(' +', ' ', sentence)

    # removing unwanted lines starting with special characters
    sentence = re.sub(r'\n: \'\'.*', '', sentence)
    sentence = re.sub(r'\n!.*','',sentence)
    sentence = re.sub(r'^:\'\'.*','',sentence)

    # removing non-breaking new line characters
    sentence = re.sub(r'\n', ' ', sentence)

    # removing punctuations
    sentence = re.sub(r'[^\w\s]', ' ', sentence)

    # tokenizing sentence
    tokens = spacy_nlp(sentence)

    # lower, strip and lemmatize text
    tokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in tokens]

    # remove stopwords, and exvlude words less than 2 characters
    tokens = [word for word in tokens if word not in stop_words and word not in punctuation and len(word) > 2]

    return tokens


# applying the spacy tokenizer function on the lead paragraph
article_df['preprocessed_lead_para'] = article_df['lead_paragraph'].map(lambda x: spacy_tokenizer(x)).copy()
article_df[['preprocessed_lead_para', 'lead_paragraph']].head()

# Assigning preprocessed data to a variable
lead_para = article_df['preprocessed_lead_para']
# To visualise most frequent occuring words throught the vocabulary

# series = pd.Series(np.concatenate(lead_para)).value_counts()[:100]
# print(series)

# wordcloud = WordCloud(background_color='white').generate_from_frequencies(series)
# plt.figure(figsize=(9,9), facecolor=None)
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')
#plt.show()
# Creating a term dictionary that has all the unique vocabularies(tokens) and index(ID) of each token

def dictionary():
    global dictionary
    dictionary = corpora.Dictionary(lead_para)


dictionary()

# Printing top 50 item from the dictionary of vocabulary
dict_tokens = [[[dictionary[key], dictionary.token2id[dictionary[key]]] for key, value in dictionary.items() if key <= 50]]
dict_tokens
# Representing text as bag of words
corpus = [dictionary.doc2bow(text) for text in lead_para]
# glimps of frequence of each word in a paragraph
word_frequencies = [[(dictionary[id], frequency) for id, frequency in line] for line in corpus]
print(word_frequencies)

# Building model

article_tfidf_model = gensim.models.TfidfModel(corpus, id2word=dictionary)
article_lsi_model = gensim.models.LsiModel(article_tfidf_model[corpus], id2word=dictionary, num_topics=30)

import pickle

with open('article_tfidf_model.txt', 'wb') as f:
    pickle.dump(article_tfidf_model, f)


with open('article_lsi_model.txt', 'wb') as f:
    pickle.dump(article_lsi_model, f)


# for ease of access of model later
gensim.corpora.MmCorpus.serialize('article_tfidf_model_mm', article_tfidf_model[corpus])
gensim.corpora.MmCorpus.serialize('article_lsi_model_mm', article_lsi_model[article_tfidf_model[corpus]])

# Loads the indexed corpus 
article_tfidf_corpus = gensim.corpora.MmCorpus('article_tfidf_model_mm')
article_lsi_corpus = gensim.corpora.MmCorpus('article_lsi_model_mm')

print(article_tfidf_corpus)
print(article_lsi_corpus) #fewer features since dimintionality has been reduced.

def article_index():
    global article_index
    article_index = MatrixSimilarity(article_lsi_corpus, num_features = article_lsi_corpus.num_terms)

article_index()

def search_similar_articles(search_term):
    query_bow = dictionary.doc2bow(spacy_tokenizer(search_term))
    query_tfidf = article_tfidf_model[query_bow]
    query_lsi = article_lsi_model[query_tfidf]

    article_index.num_best = 5

    article_list = article_index[query_lsi]
    article_list.sort(key=itemgetter(1), reverse=True)
    articles = []

    for j, article in enumerate(article_list):
        articles.append(
            {
                'Revelence': round((article[1]*100), 2),
                'Headline': article_df['headline'][article[0]],
                'Lead Paragraph': article_df['lead_paragraph'][article[0]]
            }
        )
        if j == (article_index.num_best-1):
            break
    
    return pd.DataFrame(articles, columns=['Revelence', 'Headline', 'Lead Paragraph'])


print(search_similar_articles('flames'))