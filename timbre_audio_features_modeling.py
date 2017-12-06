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

os.chdir("D:\\Docs\\Stanford - Mining Massive Datasets\\CS229\\Project\\MillionSongSubset")

output = pd.read_csv("subset.csv")
song = output.iloc[0]
flat_timbre =np.asarray(output['segments_timbre'])
flat_timbre = np.array([np.array(sublist) for sublist in flat_timbre])
flat_timbre = np.array(flat_timbre)

flat_timbres = []
lengths = []
for index, _ in output.iterrows():
    length = 0
    song = output.iloc[index]
    segments_timbre = ast.literal_eval(song['segments_timbre'])
    for sublist in segments_timbre:
        for item in sublist:
            length += 1
            flat_timbres.append(np.array([item]))
    lengths.append(length)
    
flat_timbres = np.array(flat_timbres)    

remodel = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=100)
remodel.fit(flat_timbres, lengths)
Z2 = remodel.predict(flat_timbres) 

np.average(flat_timbres[Z2==0])
np.average(flat_timbres[Z2==1])
np.average(flat_timbres[Z2==2])

A = pd.read_csv("A.csv")

all_audio_feautures = []
for index, _ in A.iterrows():
    song = A.iloc[index]
    if isinstance(song['audio_features'], str) and song['audio_features'] != '[None]':
        audio_feautures = ast.literal_eval(song['audio_features'])[0]
        audio_feautures = [[audio_feautures['acousticness'], audio_feautures['danceability'], audio_feautures['energy'], audio_feautures['instrumentalness'], audio_feautures['liveness'], audio_feautures['loudness'], audio_feautures['speechiness'], audio_feautures['valence']]]
        audio_feautures = np.array(audio_feautures)
        all_audio_feautures.extend(audio_feautures)
        
all_audio_feautures = np.array(all_audio_feautures)
audio_feautures_embedded = TSNE(n_components=2).fit_transform(all_audio_feautures)

audio_feautures_embedded
    
