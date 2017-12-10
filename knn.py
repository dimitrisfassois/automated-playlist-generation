import pandas as pd
import numpy as np

from util import *

good_songs = []
flat_songs = []

for subset_file in msd:
    songs = pd.read_csv(subset_file)
    for index, _ in songs.iterrows():
        song = songs.iloc[index]
        # some songs have 'NONE' as the audio feature
        if not ast.literal_eval(song['audio_features'])[0]:
            continue
        good_songs.append(song)
        flat_songs.append(np.array(flatten_song(song)))

# returns an array of (song name, distance, flat_song)
def knn(k, song):
    distances = []
    i = 0

    for otherSong in flat_songs:
        distances.append((distance(song, otherSong), i, otherSong))
        i = i + 1

    distances = sorted(distances)

    neighbors = []
    for j in range(0,k):
        sng = str(good_songs[distances[j][1]]['artist_name']) + ', ' + str(good_songs[distances[j][1]]['title'])
        neighbors.append((sng, distances[j][0], distances[j][2]))

    return neighbors
