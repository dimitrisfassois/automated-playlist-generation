# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 20:55:47 2017

@author: dimit_000
"""

import ast
import numpy as np
from hmmlearn import hmm
import os
import pandas as pd
from sklearn.manifold import TSNE

os.chdir("C:\\Users\\fade7001\\Documents\\Resources\\CS229\\CS229 Project")

import sys
import csv
maxInt = sys.maxsize
decrement = True

while decrement:
    # decrease the maxInt value by factor 10 
    # as long as the OverflowError occurs.

    decrement = False
    try:
        csv.field_size_limit(maxInt)
    except OverflowError:
        maxInt = int(maxInt/10)
        decrement = True
        
songs = pd.read_csv("A_N_lda.csv", engine='python')
song = songs.iloc[0]
flat_timbre =np.asarray( ast.literal_eval(song['segments_timbre']))
flat_timbre = np.array([np.array(sublist) for sublist in flat_timbre])
flat_timbre = np.array(flat_timbre)


flat_timbres = []
lengths = []
train_sample_ind = np.random.permutation(songs.shape[0])[0:5000]

# Extract the timbre of every song    
itera = 0
for index in train_sample_ind:
    itera += 1
    print(itera)
    length = 0
    song = songs.iloc[index]
    segments_timbre = ast.literal_eval(song['segments_timbre'])
    timbres = np.average(segments_timbre, axis=1)
    for item in timbres:
        length += 1
        flat_timbres.append(np.array([item]))
    lengths.append(length)
    
flat_timbres = np.array(flat_timbres)
np.savetxt('flat_timbres.txt', flat_timbres)    

# Model the timbre as an HMM
model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=100)
model.fit(flat_timbres, lengths)
Z2 = model.predict(flat_timbres) 

np.average(flat_timbres[Z2==0]) # 4.96
np.average(flat_timbres[Z2==1]) # 13.76
np.average(flat_timbres[Z2==2]) # -5.35

for index, _ in songs.iterrows():
    segments_timbre = ast.literal_eval(songs.loc[index, 'segments_timbre'])
    timbres = np.average(segments_timbre, axis=1)
    song_timbres=[]
    for item in timbres:
        song_timbres.append(np.array([item]))
    song_timbres = np.array(song_timbres)
    hidden_path = model.predict(song_timbres) 
    hidden_path_avg = np.average(hidden_path)
    songs.loc[index, 'hidden_path_avg'] = hidden_path_avg
    print(index)
    print(hidden_path_avg)


### Didn't pursue clustering on audio features!
A = pd.read_csv("A.csv")

all_audio_feautures = []
for index, _ in A.iterrows():
    song = A.iloc[index]
    if isinstance(song['audio_features'], str) and song['audio_features'] != '[None]':
        audio_feautures = ast.literal_eval(song['audio_features'])[0]
        audio_feautures = [[audio_feautures['acousticness'], audio_feautures['danceability'], audio_feautures['energy'], audio_feautures['instrumentalness'], audio_feautures['liveness'], audio_feautures['loudness'], audio_feautures['speechiness'], audio_feautures['valence']]]
        audio_feautures = np.array(audio_feautures)
        all_audio_feautures.extend(audio_feautures)
        print(index)
        
all_audio_feautures = np.array(all_audio_feautures)
audio_feautures_embedded = TSNE(n_components=2).fit_transform(all_audio_feautures)

audio_feautures_embedded
    
