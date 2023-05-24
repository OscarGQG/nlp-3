#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import gzip

def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)

def get_dataframe(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

songs_oscar = get_dataframe('meta_Digital_Music.json.gz')


# In[2]:


songs_oscar.info()


# In[3]:


songs_oscar


# In[4]:


# Check for missing values
print(songs_oscar.isna().sum())


# In[5]:


# Check the data types of each column
print(songs_oscar.info())


# In[6]:


songs_oscarq = songs_oscar.sample(frac=0.25)


# In[7]:


print(songs_oscarq.info())


# In[8]:


import numpy as np
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

# Select relevant columns and drop any rows with missing values
songs_oscarq = songs_oscarq[['asin', 'title', 'brand', 'description']].dropna()

# Preprocess the text data in the 'description' column
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    # Convert list to string
    text = ' '.join(text)
    # Convert to lowercase
    text = text.lower()
    # Remove punctuations and special characters
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove stop words
    text = ' '.join([word for word in text.split() if word not in stop_words])
    # Perform stemming
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    return text

songs_oscarq['description'] = songs_oscarq['description'].apply(lambda x: preprocess_text(x) if x else '')

# Create TF-IDF vectors for the textual description of every song
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(songs_oscarq['description'])
tfidf_matrix = csr_matrix(tfidf_matrix) # convert to sparse matrix

# Compute pairwise cosine similarity score of every song title
cosine_sim = cosine_similarity(tfidf_matrix)


# In[ ]:





# In[9]:


songs_oscar['title']


# In[16]:


# Define the recommender function
def recommend_song(title):
    # Find the index of the queried song title
    song_index = songs_oscarq[songs_oscarq['title'] == title].index
    if len(song_index) == 0:
        # If the song title is not found, return an error message
        print(f"We don't have recommendations for {title}.")
        return
    
    # Compute the cosine similarities for the queried song
    cosine_scores = list(enumerate(cosine_sim[song_index][0]))
    
    # Sort the cosine scores in descending order
    cosine_scores = sorted(cosine_scores, key=lambda x: x[1], reverse=True)
    
    # Get the top 10 most similar songs
    top_songs = cosine_scores[1:11]
    
    # Print the top 10 most similar song titles
    print(f"\nTop 10 recommended songs for {title}:")
    for i, score in top_songs:
        print(f"{i+1}. {songs_oscar.iloc[i]['title']}")
    
    return

# Prompt the user for input and recommend songs
while True:
    user_input = input("\nEnter a song title (or 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    recommend_song(user_input)


# In[ ]:





# In[ ]:




