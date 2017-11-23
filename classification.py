import os
import glob
from sklearn import linear_model
import pandas as pd

from util import *

#load songs
# songs = []
# for root, dirs, files in os.walk('/Users/kade/LocalDocs/MillionSongSubset2/data'):
#     files = glob.glob(os.path.join(root, '*.txt'))
#     for f in files:
#         with open(f, 'r') as inFile:
#             lines = [line.rstrip('\n') for line in inFile]
#             for line in lines:
#                 song = decode_song(line)
#                 song['year'] = normalize(song['year'], MIN_YEAR, MAX_YEAR)
#                 song['tempo'] = normalize(song['tempo'], MIN_TEMPO, MAX_TEMPO)
#                 # remove non-numeric attributes
#                 song.pop('song_id', None)
#                 song.pop('title', None)
#                 song.pop('segments_timbre', None)
#                 song.pop('artist_name', None)
#                 songs.append(flatten_song(song))


songs = pd.read_csv("/Users/kade/LocalDocs/subset.csv")
flat_songs = []
for index, _ in songs.iterrows():
    song = songs.iloc[index]

    if type(song['audio_features']) is float:
        continue

    flat_songs.append(flatten_song(song))

# arbitrary mock data
x_train = flat_songs[0:10]
y_train = [ x % 2 for x in range(10)]
x_test = flat_songs[10:20]

model = linear_model.LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print y_pred
