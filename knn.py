from __future__ import division

import os
import glob
from random import *
import pandas as pd
import json
import ast
import math

from util import *

songs = pd.read_csv("/Users/kade/LocalDocs/MillionSongSubset/subset.csv")

def normalize(val, minVal, maxVal):
    return (float(val) - minVal) / maxVal

def distance(song1, song2):
    if type(song2['audio_features']) is float:
        return float("inf")

    d = 0
    keys = ['sentiment_score', 'popularity']
    for key in keys:
        d += abs(float(song1[key]) - float(song2[key]))

    # excluding loudness for now since it seems to not be normalized
    audio_features = ['acousticness', 'tempo', 'instrumentalness', 'liveness', 'speechiness', 'valence', 'danceability']

    # not sure what I messed up, but there was some single quote double quote weirdness that this fixes
    song1_audio_features = ast.literal_eval(song1['audio_features'])[0]
    song2_audio_features = ast.literal_eval(song2['audio_features'])[0]

    # print song1
    # print song2

    for feature in audio_features:
        # normalize
        if feature == 'tempo':
            d += abs(normalize(song1_audio_features['tempo'], MIN_TEMPO, MAX_TEMPO) - normalize(song1_audio_features['tempo'], MIN_TEMPO, MAX_TEMPO))
        else:
            d += abs(float(song1_audio_features[feature]) - float(song2_audio_features[feature]))

    # TODO fix this when we format the new dates
    # d += abs(normalize(song1['year'], MIN_YEAR, MAX_YEAR) - normalize(song2['year'], MIN_YEAR, MAX_YEAR))

    return d

def knn(k, song):
    distances = []
    i = 0
    for index, _ in songs.iterrows():
        otherSong = songs.iloc[index]
        if song['title'] == otherSong['title'] and song['artist_name'] == otherSong['artist_name']:
            continue
        distances.append( (distance(song, otherSong), i))
        i = i + 1

    distances = sorted(distances)

    neighbors = []
    for j in range(0,k):
        neighbors.append(songs.loc[distances[j][1],'title'])

    return neighbors

n = knn(4, songs.iloc[0])
print n
