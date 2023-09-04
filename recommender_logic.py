# Dependencies 
import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import difflib
from flask import Flask
from fetch_data import flatten_and_frame

article_df = flatten_and_frame()

# checking for null values
article_df.info()
article_df = article_df.iloc[:, 0:9]

# basic descriptive statistics of the numeric data
article_df['word_count'].describe()

# dropping all the rows that have a word count of 0 or less
column = article_df.word_count
zero_words = column[column <= 0]
article_df.drop(zero_words.index, inplace=True)
article_df.reset_index(drop=True, inplace=True)

#print(article_df)

# merging the all three categories in one column
article_df['combined_categories'] = article_df[article_df.columns[-3:]].apply(
    lambda x: ','.join(x.dropna().astype(str)),
    axis=1
)

# replacing all the null values with an empty string
article_df.fillna('', inplace=True)

# We will be computing TF-IDF vectors for each article and then compute the cosine similarity between the vectors to find the similarity between the articles
# The first step will be to tokenize the text and remove the stop words.
tfidf = TfidfVectorizer(stop_words='english')

# Fitting the tfidf on the lead paragraph of the articles to construct the tfidf matrix
tfidf_matrix = tfidf.fit_transform(article_df['lead_paragraph'])

# In this matrix rows represent number of articles and columns represent the number of unique words from all lead paragraphs of articles
print(tfidf_matrix.shape)

# a peak at some words in the tfidf matrix
print(tfidf.get_feature_names_out()[5000:5050])

# computing the cosine similarity matrix.
# using the linear kernel to compute the cosine similarity, since we already have the tfidf vectors
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# each column contains cosine similarity for each articles with all other articles
print(cosine_sim.shape)
print(cosine_sim[0])

# creating a series of indices for the articles
indices = pd.Series(article_df.index, index = article_df['combined_categories'])
def get_recommendation(category, cosine_sim=cosine_sim):

    # Getting close match to the search
    close_match = difflib.get_close_matches(category, article_df['combined_categories'], n=1, cutoff=0.2)

    # Getting the index of article to get the category
    idx = indices[close_match[0]]

    # Getting pairwise similarity score for all the categories 
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sorting the articles based on the similar articles
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Getting 10 most similar articles
    sim_scores = sim_scores[1:11] 

    # Getting the articles 
    article_indices = [i[0] for i in sim_scores]

    # Return headlines of 10 most similar articles
    return article_df['headline'].iloc[article_indices]


print(get_recommendation('fires'))

