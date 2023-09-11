from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from gensim import corpora, similarities
import  gensim
import spacy
import re
import string
import numpy as np
from operator import itemgetter
import recommender_logic  as rl
from recommender_logic import spacy_tokenizer
import pickle
from fetch_data import flatten_and_frame


app = FastAPI()

# vars from recommender_logic.py file
dictionary = rl.dictionary
article_index = rl.article_index

spacy_nlp = spacy.load('en_core_web_sm')


article_tfidf_model = pickle.load(open('article_tfidf_model.txt', 'rb'))

article_lsi_model = pickle.load(open('article_lsi_model.txt', 'rb'))

article_df = flatten_and_frame()

class SearchRequest(BaseModel):
    search_term: str

class SearchResult(BaseModel):
    relevance: float
    headline: str
    lead_paragraph: str


@app.post("/search_similar_articles", response_model=list[SearchResult])
async def search_similar_articles(search_request: SearchRequest):
    try:
        # Tokenize and preprocess the search term
        search_term = search_request.search_term
        query_bow = dictionary.doc2bow(spacy_tokenizer(search_term))
        query_tfidf = article_tfidf_model[query_bow]
        query_lsi = article_lsi_model[query_tfidf]

        # Perform similarity search
        article_index.num_best = 5
        article_list = article_index[query_lsi]
        article_list.sort(key=itemgetter(1), reverse=True)
        articles = []

        for j, article in enumerate(article_list):
            articles.append(
                {
                    'relevance': round((article[1] * 100), 2),
                    'headline': article_df['headline'][article[0]],
                    'lead_paragraph': article_df['lead_paragraph'][article[0]]
                }
            )
            if j == (article_index.num_best - 1):
                break

        return articles

    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)