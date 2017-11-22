from __future__ import division

import os
import glob
from random import *
import pandas as pd


from util import decode_song

output = pd.read_csv("D:\\Docs\\Stanford - Mining Massive Datasets\\CS229\\Project\\MillionSongSubset\\subset.csv")

minTempo = 45.508
maxTempo = 229.864
minYear = 1959.0
maxYear = 2009.0

def normalize(val, minVal, maxVal):
    return (float(val) - minVal) / maxVal

def distance(song1, song2):
    d = 0
    keys = ['energy', 'sentiment_score', 'danceability', 'popularity']
    for key in keys:
        d += abs(float(song1[key]) - float(song2[key]))
    audio_features = ['acousticness','instrumentalness','liveness','loudness','speechiness', 'valence']
    for feature in audio_features:
        d += abs(float(song1['audio_features'][0][feature]) - float(song2['audio_features'][0][feature]))
        

    #normalize year and tempo
    d += abs(normalize(song1['year'], minYear, maxYear) - normalize(song2['year'], minYear, maxYear))
    d += abs(normalize(song1['tempo'], minTempo, maxTempo) - normalize(song2['tempo'], minTempo, maxTempo))

    return d

#load songs
#songs = []
#for root, dirs, files in os.walk('/Users/kade/LocalDocs/MillionSongSubset2/data'):
#    files = glob.glob(os.path.join(root, '*.txt'))
#    for f in files:
#        with open(f, 'r') as inFile:
#            lines = [line.rstrip('\n') for line in inFile]
#            for line in lines:
#                song = decode_song(line)
#                songs.append(song)


def knn(k, song):
    distances = []
    i = 0
    for index, _ in output.iterrows():
        otherSong = output.iloc[index]
        if song['title'] == otherSong['title'] and song['artist_name'] == otherSong['artist_name']:
            continue
        distances.append( (distance(song, otherSong), i))
        i = i + 1

    distances = sorted(distances)

    neighbors = []
    for j in range(0,k):
        neighbors.append(output.loc[distances[j][1],'title'])

    return neighbors

n = knn(4, output.iloc[0])
print map(lambda x: x['title'], n)
