#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')


# In[3]:


movies = movies.merge(credits,on='title')


# In[4]:


#coloumns to keep
#genres,id,keywords,title,overview,cast,crew
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[5]:


#creating tags coloumn by joining genres,keyword,crew,cast,overview
movies.isnull().sum()


# In[6]:


#droping the data where overview is not present
movies.dropna(inplace=True)


# In[7]:


movies.isnull().sum()


# In[8]:


#checking for duplicated data
movies.duplicated().sum()


# In[11]:


#changing format of genres coloumn
#creating helper function which gets the genre key word from the dictoniary format and stores it in a list
import ast

def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


# In[12]:


movies['genres'] = movies['genres'].apply(convert)


# In[13]:


movies['keywords'] = movies['keywords'].apply(convert)


# In[14]:


def convert3(obj):
    L=[]
    counter = 0
    for i in ast.literal_eval(obj):
        if counter!=3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L


# In[15]:


movies['cast'] = movies['cast'].apply(convert3)


# In[16]:


def fetch_director(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L


# In[17]:


movies['crew'] = movies['crew'].apply(fetch_director)


# In[18]:


#dataset after preprocessing
movies.head()


# In[19]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())


# In[20]:


movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","")for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","")for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","")for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","")for i in x])


# In[21]:


movies['tag'] = movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']


# In[23]:


new_df = movies[['movie_id','title','tag']]


# In[25]:


new_df['tag'] = new_df['tag'].apply(lambda x:" ".join(x))


# In[26]:


#new dataset 
new_df.head()


# In[27]:


new_df['tag'] = new_df['tag'].apply(lambda x:x.lower())


# In[28]:


new_df.head()


# In[34]:


import nltk


# In[35]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[36]:


def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
        
    return " ".join(y)


# In[37]:


new_df['tag'] = new_df['tag'].apply(stem)


# In[38]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')


# In[39]:


vectors = cv.fit_transform(new_df['tag']).toarray()


# In[40]:


vectors


# In[41]:


from sklearn.metrics.pairwise import cosine_similarity


# In[42]:


#cosine distance for each movie stored in the form of a matrix
similarity = cosine_similarity(vectors)


# In[43]:


def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    
    for i in movies_list:
        print(new_df.iloc[i[0]].title)


# In[44]:


recommend('Batman Begins')


# In[45]:


recommend('Avatar')


# In[ ]:




