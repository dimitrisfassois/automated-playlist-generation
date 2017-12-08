import os
import glob
import pandas as pd
from sklearn import linear_model

from util import *

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
        key = song_key(song['artist_name'],song['title'])

        if key in playlist_song_titles:
            playlist_songs[key] = (flatten_song(song))
        elif i < 100:
            i = i + 1
            neg_examples.append(flatten_song(song))

playlist = playlist_songs.values()
playlist_len = len(playlist)
mid = playlist_len / 2

pos_train = playlist[0:mid] # half of playlist_songs
neg_train = neg_examples[0:mid] # random songs not in playlist

pos_test = playlist[mid:playlist_len] # other half of playlist_songs
neg_test = neg_examples[mid:playlist_len] # random songs not in playlist

x_train = pos_train + neg_train
y_train = [ 1 for x in range(len(pos_train))] + [ 0 for x in range(len(neg_train))]

model = linear_model.LogisticRegression(C=1e5)

model.fit(x_train, y_train)

print 'Positive Test Examples'
print model.predict(pos_test)

print 'Neg Test Examples'
print model.predict(neg_test)
