import pandas as pd
import numpy as np
import sys

from util import *

# evaluates our algorithm against a particular playlist
playlist_song_titles = {}
with open('./playlists/60s, 70s, 80s Classic Rock.txt', 'r') as inFile:
    lines = [line.rstrip('\n') for line in inFile]
    for line in lines:
        playlist_song_titles[line.lower()] = True

good_songs = []
flat_songs = []
playlist_songs = {}
i = 0

print 'Reading in dataset...'
songs = pd.read_csv("/Users/kade/LocalDocs/A_N_lda_hmm3.csv")
for index, _ in songs.iterrows():
    song = songs.iloc[index]
    # some songs have 'NONE' as the audio feature
    if not ast.literal_eval(song['audio_features'])[0]:
        continue
    # cut unpopular songs in an effort to make playlist more recognizable
    if song['popularity'] < 0.2:
        continue
    good_songs.append(song)
    flat_songs.append(np.array(flatten_song(song)))

# for i, song in enumerate(good_songs):
#     if song['artist_name'] == 'The Smiths':
#         print '==='
#         print i
#         print song['title']

print 'Total songs: ' + str(len(good_songs))

# DP shortest path with k nodes
# Returns list of flat songs on playlist
def shortest_path(k, song1_index, song2_index):
    # algo operates on edges
    k = k + 2

    song1 = flat_songs[song1_index]
    song2 = flat_songs[song2_index]
    dist = distance(song1, song2)

    # restrict the search to songs that are generally close to our two songs
    # this is a bounding sphere (in n-dimensions) with the two songs on the edges
    # it finds the average of the two points, and restricts the graph to songs nearby
    neighborhood = []
    midpoint = (song1 + song2) / 2
    radius = dist / 2.0 + 0.01

    neighborhood_to_good_songs = []
    good_songs_to_neighborhood = [0] * len(flat_songs)

    for i, song in enumerate(flat_songs):
        if distance(midpoint, song) <= radius:
            neighborhood.append(song)
            neighborhood_to_good_songs.append(i)
        good_songs_to_neighborhood[i] = len(neighborhood) - 1

    V = len(neighborhood)
    print 'Songs within bounding area: ' + str(V)

    if (V < k or V > 4000):
        return 0

    # get new indexes of songs in the cut down neighborhood
    song1_n_index = good_songs_to_neighborhood[song1_index]
    song2_n_index = good_songs_to_neighborhood[song2_index]

    print 'Calculating distances...'
    # make distances adjacancy matrix
    distances = np.zeros((V,V))

    # it's symmetrics, so save a little time just iterating over half
    for i in range(V):
        for j in range(i + 1, V):
            d = distance(neighborhood[i], neighborhood[j])
            distances[i][j] = d
            distances[j][i] = d

    # diagonal
    for i in range(V):
        distances[i][i] = sys.maxint

    # shortest_paths[v][e] means shortest path from song1 to v with e+1 edges
    shortest_paths = np.zeros((V,k))
    shortest_paths = shortest_paths + sys.maxint

    # path_indices[v][e] = u means that to get to v, take the shortest path to u then go to v
    path_indices = np.zeros((V,k))
    # prevent cycles
    visited = [set() for _ in range(V)]

    print 'Calculating path...'
    for e in range(k):
        print 'e: ' + str(e)
        for i in range(V):
            if e == 0:
                shortest_paths[i][e] = distances[song1_n_index][i]
                path_indices[i][e] = i
                visited[i].add(song1_n_index)
            else:
                # don't let the source or destination be part of the path
                if e < k-1 and (i == song1_n_index or i == song2_n_index):
                    continue
                # don't let previously visited things be part of the path
                if i in visited[i]:
                    continue

                branches = [sys.maxint] * V

                for j in range(V):
                    if e < k-1 and (j == song1_n_index or j == song2_n_index):
                        continue
                    if j in visited[i]:
                        continue

                    branches[j] = shortest_paths[j][e-1] + distances[i][j]

                best_path = min(branches)
                shortest_paths[i][e] = best_path
                node = branches.index(best_path)
                path_indices[i][e] = node
                visited[i].add(node)

    # construct the path backwards
    path = [song2_n_index]
    for e in reversed(range(1, k)):
        v = int(path_indices[path[-1]][e])
        path.append(v)

    path.append(song1_n_index)

    # the algorithm lets one duplicate node sneak in, so remove it
    path = reduce(lambda l, x: l if x in l else l + [x], path, [])
    path.reverse()
    print 'Shortest path (node indices): ' + str(path)
    print 'Path dist: ' + str(shortest_paths[song2_n_index][k-1])
    print 'Actual dist: ' + str(distances[song1_n_index][song2_n_index])

    plist = []

    print 'The Playlist:'
    for i in path:
        index = neighborhood_to_good_songs[i]
        s = good_songs[index]
        print s['artist_name'] + ', ' + s['title']
        plist.append(neighborhood[i])

    return plist

# finds the indices of the songs and return the path
def shortest_path_between_flat(k, s1, s2):
    i = 0
    j = 0
    for index, song in enumerate(flat_songs):
        if (s1 == song).all():
            i = index
        if (s2 == song).all():
            j = index
    return shortest_path(k, i, j)

# 24 and 308 are The Smiths songs
# shortest_path(10, 24, 308)

########################################## EVALUATION

dist_avg = 0
skips = 0
for playlist_name in best_playlists:
    # evaluates our algorithm against a particular playlist
    playlist_song_titles = {}
    with open(playlist_name, 'r') as inFile:
        lines = [line.rstrip('\n') for line in inFile]
        for line in lines:
            playlist_song_titles[line.lower()] = True

    playlist_songs = {}
    neg_examples = [] # random selection of negative examples
    for i in range(len(good_songs)):
        key = good_songs[i]['song_artist_title']
        if key in playlist_song_titles:
            playlist_songs[key] = np.array(flat_songs[i])

    playlist = playlist_songs.values()
    playlist_len = len(playlist)
    print 'We have ' + str(playlist_len) + ' songs in playlist: ' + str(playlist_name)

    # find path between first and last songs
    predictions = shortest_path_between_flat(playlist_len, playlist[0], playlist[-1])

    if predictions == 0:
        print "THIS PLAYLIST WAS TOO BIG/SMALL. SKIPPING"
        skips = skips + 1
        continue

    count = playlist_len * len(predictions)
    avg = 0
    for s1 in playlist:
        for s2 in predictions:
            avg = avg + distance(s1, s2) / count
    dist_avg = dist_avg + avg
    print 'Avg distance to actual playlist: ' + str(normalize_distance(avg))


denom = float(len(best_playlists) - skips)
print denom
print 'Avg distance to actual playlist: ' + str(normalize_distance(dist_avg / denom))
