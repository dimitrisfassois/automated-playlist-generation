import pandas as pd
import numpy as np
import sys

from util import *

good_songs = []
flat_songs = []

for subset_file in msd_test:
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

    for song in flat_songs:
        if distance(midpoint, song) <= radius:
            neighborhood.append(song)

    print len(neighborhood)

    # make adjacancy matrix
    V = len(neighborhood)
    distances = np.zeros((V,V))
    for i in range(V):
        for j in range(V/2 + 1):
            d = distance(neighborhood[i], neighborhood[j])
            distances[i][j] = d
            distances[j][i] = d

    shortest_paths = np.zeros((V,V,k))
    shortest_paths = shortest_paths + sys.maxint

    for e in range(k):
        for i in range(V):
            for j in range(V):

                # base case
                if e == 0:
                    shortest_paths[i][j][e] = distances[i][j]
                else:
                    for a in range(V):
                        if i != a and j!= a:
                            p = min(shortest_paths[i][j][e], distances[i][a] + shortest_paths[a][j][e-1])
                            shortest_paths[i][j][e] = p
                            # TODO track actual path

    # TODO get indices of songs and map back to titles
    i = numpy.nonzero(neighborhood == song1)
    j = numpy.nonzero(neighborhood == song2)
    print i[0][0]
    print j[0][0]

    print shortest_paths[0][1][0]
    print shortest_paths[0][1][1]
    print shortest_paths[0][1][2]
    print distances[0][1]

    #TODO keep track of song after conversion to neghborhood and print path

shortest_path(3, flat_songs[0], flat_songs[1])
