import os
import glob
import ast

keys = ['song_id', 'title', 'segments_timbre', 'artist_name', 'sentiment_score', 'year']
delimiter = '|||'

MIN_TEMPO = 45.508
MAX_TEMPO = 229.864
MIN_YEAR = 1959.0
MAX_YEAR = 2009.0

#song to txt
def encode_song(song):
    return delimiter.join(map(str, song.values()))

#txt to song
def decode_song(songTxt):
    values = songTxt.split(delimiter)
    i = 0
    song = {}
    for key in keys:
        song[key] = values[i]
        i = i + 1
    return song

def normalize(val, minVal, maxVal):
    return (float(val) - minVal) / maxVal

def flatten_song(song):
    songArray = []
    songArray.append(float(song['sentiment_score']))
    songArray.append(float(song['popularity']))

    audio_features = ['acousticness', 'tempo', 'instrumentalness', 'liveness', 'speechiness', 'valence', 'danceability']
    song_audio_features = ast.literal_eval(song['audio_features'])[0]

    for feature in audio_features:
        # normalize
        if feature == 'tempo':
            songArray.append(normalize(song_audio_features['tempo'], MIN_TEMPO, MAX_TEMPO))
        else:
            songArray.append(float(song_audio_features[feature]))

    return songArray

# how much overlap is there in our playlists and the MSD
def compute_overlap():
    songs = {}
    hits = 0
    misses = 0
    for root, dirs, files in os.walk('/Users/kade/LocalDocs/MillionSongSubset/data'):
        files = glob.glob(os.path.join(root, '*.txt'))
        for f in files:
            with open(f, 'r') as inFile:
                lines = [line.rstrip('\n') for line in inFile]
                for line in lines:
                    song = decode_song(line)
                    songs[(song['artist_name'] + ', ' + song['title']).lower()] = True

    for root, dirs, files in os.walk('./playlists'):
        files = glob.glob(os.path.join(root, '*.txt'))
        for f in files:
            with open(f, 'r') as inFile:
                lines = [line.rstrip('\n') for line in inFile]
                for line in lines:
                    if line.lower() in songs:
                        hits = hits + 1
                    else:
                        misses = misses + 1


# import pandas as pd
# songs = pd.read_csv("/Users/kade/LocalDocs/subset.csv")
# names = []
# for index, _ in songs.iterrows():
#     names.append(str(songs.iloc[index]['artist_name']) + ', ' + str(songs.iloc[index]['title']))
#
# for name in sorted(names):
#     print name
