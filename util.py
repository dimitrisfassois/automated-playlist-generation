import os
import glob
import ast
import numpy

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
    songArray.append(float(song['sentiment_score']) / 4) # 0/1 is too extreme
    songArray.append(float(song['popularity']))
    songArray.append(normalize(song['year'], MIN_YEAR, MAX_YEAR))

    audio_features = ['acousticness', 'tempo', 'instrumentalness', 'liveness', 'speechiness', 'valence', 'danceability']
    song_audio_features = ast.literal_eval(song['audio_features'])[0]

    # print song_audio_features

    for feature in audio_features:
        # normalize
        if feature == 'tempo':
            songArray.append(normalize(song_audio_features['tempo'], MIN_TEMPO, MAX_TEMPO))
        else:
            songArray.append(float(song_audio_features[feature]))

    return songArray

def distance(song1, song2):
    return numpy.linalg.norm(song1-song2)

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

msd = [
    '/Users/kade/LocalDocs/A-enhanced-trimmed.csv',
    '/Users/kade/LocalDocs/B-enhanced-trimmed.csv',
    '/Users/kade/LocalDocs/C-enhanced-trimmed.csv',
    '/Users/kade/LocalDocs/D-enhanced-trimmed.csv',
    '/Users/kade/LocalDocs/E-enhanced-trimmed.csv',
    '/Users/kade/LocalDocs/F-enhanced-trimmed.csv',
    '/Users/kade/LocalDocs/G-enhanced-trimmed.csv',
    '/Users/kade/LocalDocs/H-enhanced-trimmed.csv',
    '/Users/kade/LocalDocs/I-enhanced-trimmed.csv',
    '/Users/kade/LocalDocs/J-enhanced-trimmed.csv',
    '/Users/kade/LocalDocs/K-enhanced-trimmed.csv',
    '/Users/kade/LocalDocs/L-enhanced-trimmed.csv',
    '/Users/kade/LocalDocs/N-enhanced-trimmed.csv',
]

msd_test = ['/Users/kade/LocalDocs/C-enhanced-trimmed.csv']

best_playlists = [
    './playlists/60s, 70s, 80s Classic Rock.txt',

]

def song_key(artist, title):
    return artist.lower() + ', ' + title.lower()
