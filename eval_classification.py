
import os
import glob
import pandas as pd
from sklearn import linear_model
import numpy as np
import ast

from util import *

os.chdir("C:\\Users\\fade7001\\Documents\\Resources\\CS229\\CS229 Project")

songs = pd.read_csv("A_N_lda.csv")

# evaluates our algorithm against a particular playlist
playlist_song_titles = {}
with open('./playlists/60s, 70s, 80s Classic Rock.txt', 'r') as inFile:
    lines = [line.rstrip('\n') for line in inFile]
    for line in lines:
        playlist_song_titles[line.lower()] = True

playlist_songs = {}
neg_examples = [] # random selection of negative examples
i = 0

for index, _ in songs.iterrows():
    song = songs.iloc[index]

    if not ast.literal_eval(song['audio_features'])[0]:
        continue

    key = song['song_artist_title'] 

    if key in playlist_song_titles:
        playlist_songs[key] = (flatten_song(song))
    elif i < 100:
        i = i + 1
        neg_examples.append(flatten_song(song))

playlist = list(playlist_songs.values())
playlist_len = len(playlist)
mid = playlist_len * 3 / 4
print( 'We have ' + str(playlist_len) + ' songs')

pos_train = playlist[0:int(mid)] # 3/4 of playlist_songs
neg_train = neg_examples[0:int(mid)] # random songs not in playlist

pos_test = playlist[int(mid):playlist_len] # other 1/4 of playlist_songs
neg_test = neg_examples[int(mid):playlist_len] # random songs not in playlist

x_train = pos_train + neg_train
y_train = [ 1 for x in range(len(pos_train))] + [ 0 for x in range(len(neg_train))]

x_train = np.array(x_train)
y_train = np.array(y_train)

x_test = pos_test + neg_test
y_test = [ 1 for x in range(len(pos_test))] + [ 0 for x in range(len(neg_test))]

x_test = np.array(x_test)
y_test = np.array(y_test)

model = linear_model.LogisticRegression(C=1e5)

model.fit(x_train, y_train)

train_preds = model.predict(x_train)
print('Train accuracy')
print((train_preds == y_train).sum()/y_train.shape[0])
print(round(float(sum(train_preds)) / float(len(x_train)), 2))

test_preds = model.predict(x_test)
print('Test accuracy')
print((test_preds == y_test).sum()/y_test.shape[0])
