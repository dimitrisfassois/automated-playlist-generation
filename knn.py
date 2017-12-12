import pandas as pd
import numpy as np

from util import *

good_songs = []
flat_songs = []

print 'Reading in dataset...'
songs = pd.read_csv("/Users/kade/LocalDocs/A_N_lda.csv")
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

########################################## EVALUATION

within_train_avg = 0
pos_train_avg = 0
pos_test_avg = 0
neg_test_avg = 0

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
        elif i < 1000:
            neg_examples.append(np.array(flat_songs[i]))

    playlist = playlist_songs.values()
    playlist_len = len(playlist)
    divider = playlist_len * 3 / 4
    print 'We have ' + str(playlist_len) + ' songs in playlist: ' + str(playlist_name)

    pos_train = playlist[0:divider] # 3/4 of playlist_songs
    pos_test = playlist[divider:playlist_len] # other 1/4 of playlist_songs
    neg_test = neg_examples[divider:playlist_len] # random songs not in playlist

    centroid = sum(pos_train) / len(pos_train)
    predictions = knn(playlist_len / 4, centroid)

    # avg distance from arbitrary point in X to point in Y
    # calculated by getting distance from every point in X to every point in Y and averaging
    count = len(pos_train) * len(pos_train)
    avg = 0
    for s1 in pos_train:
        for s2 in pos_train:
            avg = avg + distance(s1, s2) / count
    within_train_avg = within_train_avg + avg
    print 'Baseline: Avg distance within train set:          ' + str(normalize_distance(avg))

    count = len(pos_train) * len(predictions)
    avg = 0
    for s1 in pos_train:
        for s2 in predictions:
            avg = avg + distance(s1, s2[2]) / count
    pos_train_avg = pos_train_avg + avg
    print 'Avg distance of prediction to positive train set: ' + str(normalize_distance(avg))

    count = len(pos_test) * len(predictions)
    avg = 0
    for s1 in pos_test:
        for s2 in predictions:
            avg = avg + distance(s1, s2[2]) / count
    pos_test_avg = pos_test_avg + avg
    print 'Avg distance of prediction to positive test set:  ' + str(normalize_distance(avg))

    count = len(neg_test) * len(predictions)
    avg = 0
    for s1 in neg_test:
        for s2 in predictions:
            avg = avg + distance(s1, s2[2]) / count
    neg_test_avg = neg_test_avg + avg
    print 'Avg distance of prediction to negative test set:  ' + str(normalize_distance(avg))

print '================OVERALL AVERAGES================'

denom = float(len(best_playlists))
print denom
print 'Baseline: Avg distance within train set:          ' + str(normalize_distance(within_train_avg / denom))
print 'Avg distance of prediction to positive train set: ' + str(normalize_distance(pos_train_avg / denom))
print 'Avg distance of prediction to positive test set:  ' + str(normalize_distance(pos_test_avg / denom))
print 'Avg distance of prediction to negative test set:  ' + str(normalize_distance(neg_test_avg / denom))
