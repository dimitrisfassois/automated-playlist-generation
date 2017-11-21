import os
import glob
from random import *

from util import *

def distance(song1, song2):
    d = 0
    keys = ['energy', 'sentiment_score', 'danceability']
    for key in keys:
        d += abs(float(song1[key]) - float(song2[key]))

    #normalize year and tempo
    d += abs(normalize(song1['year'], MIN_YEAR, MAX_YEAR) - normalize(song2['year'], MIN_YEAR, MAX_YEAR))
    d += abs(normalize(song1['tempo'], MIN_TEMPO, MAX_TEMPO) - normalize(song2['tempo'], MIN_TEMPO, MAX_TEMPO))

    return d

#load songs
songs = []
for root, dirs, files in os.walk('/Users/kade/LocalDocs/MillionSongSubset2/data'):
    files = glob.glob(os.path.join(root, '*.txt'))
    for f in files:
        with open(f, 'r') as inFile:
            lines = [line.rstrip('\n') for line in inFile]
            for line in lines:
                song = decode_song(line)
                songs.append(song)

def knn(k, song):
    distances = []
    i = 0
    for otherSong in songs:
        if song == otherSong:
            continue
        distances.append( (distance(song, otherSong), i))
        i = i + 1

    distances = sorted(distances)

    neighbors = []
    for j in range(0,k):
        neighbors.append(songs[distances[j][1]])

    return neighbors

n = knn(4, songs[0])
print map(lambda x: x['title'], n)
