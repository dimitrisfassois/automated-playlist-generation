# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 21:59:45 2017

@author: dimit_000
"""

import pandas as pd
import os
import ast
import glob
import numpy as np
from collections import defaultdict
#from util import *

from collections import defaultdict

os.chdir('D:\\Docs\\Stanford - Mining Massive Datasets\\CS229\\Project\\MillionSongSubset')


A = pd.read_csv("A-enhanced-timmed.csv")
B = pd.read_csv("B-enhanced-timmed.csv")
C = pd.read_csv("C-enhanced-timmed.csv")
D = pd.read_csv("D-enhanced-timmed.csv")
E = pd.read_csv("E-enhanced-trimmed.csv")
F = pd.read_csv("F-enhanced-trimmed.csv")
I = pd.read_csv("I-enhanced-trimmed.csv")
K = pd.read_csv("K-enhanced-trimmed.csv")
L = pd.read_csv("L-enhanced-trimmed.csv")

songs = A.append(B).append(C).append(D).append(E).append(F).append(I).append(K)

del A
del B
del C
del D
del E
del F
del I
del K
del L

def song_key(artist, title):
    return artist.lower() + ', ' + title.lower()

# Create song_id column that can be used for looking up the song lyrics
# Create a dict that contains lyrics for every song
songs_lyrics = {}
for index, _ in songs.iterrows():
    song = songs.iloc[index]
    lyrics = ast.literal_eval(song['todo'])
    lyrics = list(lyrics.keys())
    key = song_key(song.loc['artist_name'],song.loc['title'])
    songs_lyrics[key] = lyrics
    print(key)
    songs.loc[index, 'song_artist_title'] = key
    
for index, _ in songs.iterrows():
    song = songs.iloc[index]
    key = song_key(song.loc['artist_name'],song.loc['title'])
    songs.loc[index, 'song_artist_title'] = key
    print(song)


# prints the count of songs we have data on from each playlist


msd_song_titles = {}


for index, _ in songs.iterrows():
    song = songs.iloc[index]
    key = song_key(song['artist_name'],song['title'])
    msd_song_titles[key] = True

playlist_song_titles = {}
for root, dirs, files in os.walk('./playlists'):
    files = glob.glob(os.path.join(root, '*.txt'))
    for f in files:
        playlist_name = f[12:-4]
        playlist_name = playlist_name.encode(encoding='ascii', errors='ignore')
        with open(f, 'r', encoding="utf8") as inFile:
            lines = [line.rstrip('\n') for line in inFile]
            overlap = 0
            for line in lines:
                if line.lower() in msd_song_titles:
                    overlap = overlap + 1
            if overlap > 2:
                print('Playlist: ' + str(f))
                print(overlap)
                playlist_song_titles[playlist_name] = lines
                
song_set = defaultdict(list)           
for playlist in playlist_song_titles.keys():
    for song in playlist_song_titles[playlist]:
        if song.lower() in msd_song_titles.keys():
            song_set[playlist].extend(songs_lyrics[song.lower()])
        

### LDA ####
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim

tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
en_stop = get_stop_words('en')

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()

# list for tokenized documents in loop
texts = []

# loop through document list
for j in song_set.keys():
    
    i = song_set[j]
    i = ' '.join(i)
    # clean and tokenize document string
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop]
    
    # stem tokens
    #stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    
    # add tokens to list
    texts.append(stopped_tokens)

# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)
    
# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]

# generate LDA model
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=3, id2word = dictionary, passes=20)



#columns = song_set.keys()

#df = pd.DataFrame(columns = columns)
#pd.set_option('display.width', 1000)

# Get Per-topic word probability matrix:
K = ldamodel.num_topics
 
topicWordProbMat = ldamodel.print_topics(K)
print (topicWordProbMat) 

for t in texts:
     vec = dictionary.doc2bow(t)
     print (ldamodel[vec])

################### Output LDA results in the csv #################

for song in songs_lyrics.keys():
    song_lyrics = songs_lyrics[song]
    song_lyrics = ' '.join(song_lyrics) 
    song_lyrics = song_lyrics.lower()
    song_lyrics = tokenizer.tokenize(song_lyrics)
    vec = dictionary.doc2bow(song_lyrics)
    songs.loc[index, 'song_artist_title'] = key
    print(song)
    print (ldamodel[vec][0])


# remove stop words from tokens
stopped_tokens = [i for i in tokens if not i in en_stop]

# 40 will be resized later to match number of words in DC
zz = np.zeros(shape=(40,len(song_set.keys())))

last_number=0
DC={}

for x in range (10):
  data = pd.DataFrame({columns[0]:"",
                     columns[1]:"",
                     columns[2]:"",
                     columns[3]:"",
                     columns[4]:"",
                    columns[5]:"",
                    columns[6]:"",
                    columns[7]:"",
                                                                                       
                     
                    },index=[0])
  df=df.append(data,ignore_index=True)  
  
lda_corpus = [max(prob,key=lambda y:y[1])
                for prob in ldamodel[corpus] ]    
for line in topicWordProbMat:
    
    tp, w = line
    probs=w.split("+")
    y=0
    for pr in probs:
               
        a=pr.split("*")
        for index, k in enumerate(lda_corpus):
            if k[0]==tp:
                df.loc[y,song.keys()[index]] = a[1]
        y=y+1

print (df)
print (zz)

import re


columns = song_set.keys()

indeces = []
for topic in ldamodel.print_topics():
    string = topic[1]
    indeces.extend(re.findall(r'"(.*?)"', string))

df = pd.DataFrame(columns = columns, index = indeces )
pd.set_option('display.width', 1000)

for line in topicWordProbMat:
    
    tp, w = line
    probs=w.split("+")
    y=0
    for pr in probs:
               
        a=pr.split("*")
        for index, k in enumerate(lda_corpus):
            if k[0]==tp:
                a[1] = a[1].encode('utf-8')
                df.loc[ str.strip(re.sub(r'"', '', a[1])),song_set.keys()[index]] = a[0]
        y=y+1

df = df.fillna(0)     

import seaborn as sns; sns.set()

df = df[df.columns].astype(float)
df_heatmap = sns.heatmap(df)
fig = df_heatmap.get_figure()
fig.savefig("output.png")

  
import matplotlib.pyplot as plt

zz=np.resize(zz,(len(DC.keys()),zz.shape[1]))

for val, key in enumerate(DC.keys()):
        plt.text(-2.5, val + 0.5, key,
                 horizontalalignment='center',
                 verticalalignment='center'
                 )

plt.imshow(zz, cmap='hot', interpolation='nearest')
plt.show()


