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

os.chdir('C:\\Users\\fade7001\\Documents\\Resources\\CS229\\CS229 Project')
# os.chdir('D:\\Docs\\Stanford - Mining Massive Datasets\\CS229\\Project\\MillionSongSubset')


A = pd.read_csv("A-enhanced-timmed.csv")
B = pd.read_csv("B-enhanced-timmed.csv")
C = pd.read_csv("C-enhanced-timmed.csv")
D = pd.read_csv("D-enhanced-timmed.csv")
E = pd.read_csv("E-enhanced-trimmed.csv")
F = pd.read_csv("F-enhanced-trimmed.csv")
I = pd.read_csv("I-enhanced-trimmed.csv")
K = pd.read_csv("K-enhanced-trimmed.csv")
L = pd.read_csv("L-enhanced-trimmed.csv")
M = pd.read_csv("M-enhanced-trimmed.csv")
N = pd.read_csv("N-enhanced-trimmed.csv")

songs = A.append(B).append(C).append(D).append(E).append(F).append(I).append(K).append(L).append(M).append(N)

del A
del B
del C
del D
del E
del F
del I
del K
del L
del M
del N


# The indices have duplicate values because the songs df was created by concatenating other dfs
songs.index = list(range(songs.shape[0]))


#### Add lyrics data
f = open('mxm_dataset_full.txt', 'r')
song_lyrics = {}
for line in f.readlines():
    if line == '' or line[0] == '#':
        continue
    elif line[0] == '%':
        topwords = line.strip()[1:].split(',')
    else:
        lineparts = line.strip().split(',')
        bag = {}
        id = lineparts[0]
        for wordcnt in lineparts[2:]:
            wordid, cnt = wordcnt.split(':')
            # it's 1-based!
            bag[topwords[int(wordid) - 1]] = True

        song_lyrics[id] = bag
f.close()

def song_key(artist, title):
    return artist.lower() + ', ' + title.lower()


# Add all lyrics to songs df
    # Create song_artist_title column that can be used for looking up the song lyrics
for index, _ in songs.iterrows():
    track_id = songs.loc[index, 'track_id']
    if track_id in song_lyrics:
        songs.loc[index, 'todo'] = str(song_lyrics[track_id])
    key = song_key(songs.loc[index, 'artist_name'],songs.loc[index, 'title'])
    songs.loc[index, 'song_artist_title'] = key
    print(songs.iloc[index])

# Remove duplicated songs
songs = songs.loc[~songs.duplicated('song_artist_title')]
del song_lyrics
songs.index = list(range(songs.shape[0]))

# Create a dict that contains lyrics for every song

songs_lyrics = {}
for index, _ in songs.iterrows():
    song = songs.iloc[index]
    if  isinstance(song['todo'], str):
        lyrics = ast.literal_eval(song['todo'])
        lyrics = list(lyrics.keys())
        songs_lyrics[songs.loc[index, 'song_artist_title']] = lyrics
        print(songs.loc[index, 'song_artist_title'])


# prints the count of songs we have data on from each playlist


msd_song_titles = {}

for index, _ in songs.iterrows():
    key = songs.loc[index, 'song_artist_title']
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
            if overlap > 5:
                print('Playlist: ' + str(f))
                print(overlap)
                playlist_song_titles[playlist_name] = lines
                
song_set = defaultdict(list)           
for playlist in playlist_song_titles.keys():
    for song in playlist_song_titles[playlist]:
        if song.lower() in msd_song_titles.keys():
            song_set[playlist].extend(songs_lyrics[song.lower()])
        
del msd_song_titles

### LDA ####
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
from nltk.corpus import stopwords
tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
en_stop = set(stopwords.words('english'))

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
    probs = np.zeros(3)
    lda_probs = ldamodel[vec]
    for prob in lda_probs:
        probs[prob[0]] = prob[1]
    print(song)
    print (lda_probs)
    print(probs)
    songs.loc[songs['song_artist_title']==song, 'lda_probs_topic_1'] = probs[0]
    songs.loc[songs['song_artist_title']==song, 'lda_probs_topic_2'] = probs[1]
    songs.loc[songs['song_artist_title']==song, 'lda_probs_topic_3'] = probs[2]

songs.to_csv('A_N_lda.csv')
