import pandas as pd
import numpy as np

from util import *

good_songs = []
flat_songs = []

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
    print subset_file
    songs = pd.read_csv(subset_file)
    for index, _ in songs.iterrows():
        song = songs.iloc[index]
        # some songs have 'NONE' as the audio feature
        if not ast.literal_eval(song['audio_features'])[0]:
            continue
        good_songs.append(song)
        flat_songs.append(np.array(flatten_song(song)))


        key = song_key(song['artist_name'],song['title'])

        if key in playlist_song_titles:
            playlist_songs[key] = np.array(flatten_song(song))
        elif i < 100:
            i = i + 1
            neg_examples.append(np.array(flatten_song(song)))

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

playlist = playlist_songs.values()
playlist_len = len(playlist)
divider = playlist_len * 3 / 4
print 'We have ' + str(playlist_len) + ' songs'

pos_train = playlist[0:divider] # 3/4 of playlist_songs

pos_test = playlist[divider:playlist_len] # other 1/4 of playlist_songs
neg_test = neg_examples[divider:playlist_len] # random songs not in playlist

centroid = sum(pos_train) / len(pos_train)
print centroid
predictions = knn(playlist_len / 4, centroid)

# avg distance from arbitrary point in X to point in Y
# calculated by getting distance from every point in X to every point in Y and averaging
count = len(pos_train) * len(pos_train)
avg = 0
for s1 in pos_train:
    for s2 in pos_train:
        avg = avg + distance(s1, s2) / count

print 'Baseline: Avg distance within train set:          ' + str(avg)

count = len(pos_train) * len(predictions)
avg = 0
for s1 in pos_train:
    for s2 in predictions:
        avg = avg + distance(s1, s2[2]) / count

print 'Avg distance of prediction to positive train set: ' + str(avg)

count = len(pos_test) * len(predictions)
avg = 0
for s1 in pos_test:
    for s2 in predictions:
        avg = avg + distance(s1, s2[2]) / count

print 'Avg distance of prediction to positive test set:  ' + str(avg)

count = len(neg_test) * len(predictions)
avg = 0
for s1 in neg_test:
    for s2 in predictions:
        avg = avg + distance(s1, s2[2]) / count

print 'Avg distance of prediction to negative test set:  ' + str(avg)
