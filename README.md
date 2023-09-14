# News Articles Recommender

## Goal

To Build a recommender system that suggests articles based on user's topic of interest.

## Processes

### Data collection 
Data was collected from New york times API 
All the articles posted on new york times in the Aug 2023

### Data cleaning and text pre-processing 
Used pandas for basic column filtering 
Used SpaCy for text preprocessing 

### Model building for semantic search
To group similarity of document with each other using SBERT sentence transformer.


### File structure of this repo.

* data - contains data file
* fetch_data.py - to get data and convert it to a dataframe
* recommender_logic.py - logic of the recommender system and search engine
* nytimes.ipynb - whole project in one notebook.