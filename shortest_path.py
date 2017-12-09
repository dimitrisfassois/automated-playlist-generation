import pandas as pd
import numpy as np
import sys

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

    # restrict the search to songs that are generally close to our two songs
    # this is a bounding sphere with the two songs on the edges
    neighborhood = []
    midpoint = (song1 + song2) / 2
    radius = dist / 2.0 + 0.001

    for i, song in enumerate(flat_songs):
        if distance(midpoint, song) <= radius:
            neighborhood.append(song)
    print len(neighborhood)

    # get new indexes of songs in the cut down neighborhood
    song1_n_index = 0
    song2_n_index = 0
    for i, song in enumerate(neighborhood):
        if (song == song1).all():
            song1_n_index = i
        if (song == song2).all():
            song1_n_index = i

    # make distances adjacancy matrix
    V = len(neighborhood)
    distances = np.zeros((V,V))
    for i in range(V):
        for j in range(V):
            # don't let node be connected to itself
            if i == j:
                distances[i][j] = sys.maxint
            else:
                d = distance(neighborhood[i], neighborhood[j])
                distances[i][j] = d


    # shortest_paths[v][e] means shortest path from song1 to v with e+1 edges
    shortest_paths = np.zeros((V,k))
    shortest_paths = shortest_paths + sys.maxint

    for e in range(k):
        for i in range(V):
            if e == 0:
                shortest_paths[i][e] = distances[song1_n_index][i]
            else:
                branches = [sys.maxint] * V
                for j in range(V):
                    branches[j] = shortest_paths[j][e-1] + distances[j][i]
                shortest_paths[i][e] = min(branches)

    # TODO track actual path
    # TODO get indices of songs and map back to titles

    print song1_n_index
    print song2_n_index

    for i in range(k):
        print shortest_paths[song2_n_index][i]

    print 'Actual dist'
    print distances[song1_n_index][song2_n_index]

shortest_path(10, flat_songs[352], flat_songs[8])
