import os
import glob
import pandas as pd
import numpy as np
from sklearn import linear_model

from util import *
from knn import knn

# evaluates our algorithm against a particular playlist
playlist_song_titles = {}
with open('./playlists/60s, 70s, 80s Classic Rock.txt', 'r') as inFile:
    lines = [line.rstrip('\n') for line in inFile]
    for line in lines:
        playlist_song_titles[line.lower()] = True

playlist_songs = {}
neg_examples = [] # random selection of negative examples
i = 0
for subset_file in msd:
    songs = pd.read_csv(subset_file)
    for index, _ in songs.iterrows():
        song = songs.iloc[index]

        if not ast.literal_eval(song['audio_features'])[0]:
            continue

        key = song_key(song['artist_name'],song['title'])

        if key in playlist_song_titles:
            playlist_songs[key] = np.array(flatten_song(song))
        elif i < 100:
            i = i + 1
            neg_examples.append(np.array(flatten_song(song)))

playlist = playlist_songs.values()
playlist_len = len(playlist)
mid = playlist_len * 3 / 4
print 'We have ' + str(playlist_len) + ' songs'

pos_train = playlist[0:mid] # 3/4 of playlist_songs

pos_test = playlist[mid:playlist_len] # other 1/4 of playlist_songs
neg_test = neg_examples[mid:playlist_len] # random songs not in playlist

centroid = sum(pos_train) / mid
print centroid
predictions = knn(mid, centroid)

# avg distance from arbitrary point in X to point in Y
# calculated by getting distance from every point in X to every point in Y and averaging
count = len(pos_train) + len(predictions)
avg = 0
for s1 in pos_train:
    for s2 in predictions:
        avg = avg + distance(s1, s2[2]) / count

print 'Avg distance to positive train set: ' + str(avg)

count = len(pos_test) + len(predictions)
avg = 0
for s1 in pos_test:
    for s2 in predictions:
        avg = avg + distance(s1, s2[2]) / count

print 'Avg distance to positive test set:  ' + str(avg)

count = len(neg_test) + len(predictions)
avg = 0
for s1 in neg_test:
    for s2 in predictions:
        avg = avg + distance(s1, s2[2]) / count

print 'Avg distance to negative test set:  ' + str(avg)
