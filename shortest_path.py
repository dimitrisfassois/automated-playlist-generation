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

print len(good_songs)

# DP shortest path of length k
def shortest_path(k, song1, song2):
    dist = distance(song1, song2)

    #restrict the search to songs that are generally close to our two songs
    neighborhood = []
    for song in flat_songs:
        if distance(song1, song) < dist and distance(song2, song) < dist:
            neighborhood.append(song)

    # TODO dp to get shortest path
    # V = len(neighborhood)
    # shortest_paths = np.
    # for i in range(V)
    #     shortest_paths[i] = []
    #
    # print len(neighborhood)


shortest_path(4, flat_songs[0], flat_songs[1])
