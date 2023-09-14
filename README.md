# News Articles Recommender

## Goal

To Build a recommender system that suggests articles based on user's topic of interest.

## Processes

* Data collection <br>

Data was collected from New york times API <br>
All the articles posted on new york times in the Aug 2023

* Data cleaning and text pre-processing <br>
Used pandas for basic column filtering <br>
Used SpaCy for text preprocessing 

* Model building for semantic search<br>
To group similarity of document with each other using SBERT sentence transformer.


### File structure of this repo.

* data - contains data file
* fetch_data.py - to get data and convert it to a dataframe
* recommender_logic.py - logic of the recommender system and search engine
* nytimes.ipynb - whole project in one notebook.