import os
import glob
import ast
import numpy
import math

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
    songArray.append(float(song['lda_probs_topic_1']))
    songArray.append(float(song['lda_probs_topic_2']))
    songArray.append(float(song['lda_probs_topic_3']))
    songArray.append(float(song['hidden_path_avg']))

    songArray.append(normalize(song['year'], MIN_YEAR, MAX_YEAR))

    audio_features = ['acousticness', 'tempo', 'instrumentalness', 'liveness', 'speechiness', 'valence', 'danceability']
    song_audio_features = ast.literal_eval(song['audio_features'])[0]

    for feature in audio_features:
        if feature == 'tempo':
            songArray.append(normalize(song_audio_features['tempo'], MIN_TEMPO, MAX_TEMPO))
        else:
            songArray.append(float(song_audio_features[feature]))

    return songArray

def distance(song1, song2):
    return numpy.linalg.norm(song1-song2)

# we have 13 dimensions, so farthest possible is sqrt(13)
def normalize_distance(d):
    return normalize(d, 0, math.sqrt(13))

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
    '/Users/kade/LocalDocs/M-enhanced-trimmed.csv',
    '/Users/kade/LocalDocs/N-enhanced-trimmed.csv',
    '/Users/kade/LocalDocs/O-enhanced-trimmed.csv',
]

msd_test = ['/Users/kade/LocalDocs/A-enhanced-trimmed.csv']

# 41 playlists
best_playlists = [
'./playlists/Breakup Songs - Getting Over Him.txt',
'./playlists/90s Smash Hits.txt',
'./playlists/British invasion.txt',
"./playlists/2000's pop,emo,punk rock.txt",
'./playlists/Happy Hour Classic Rock.txt',
'./playlists/Throw Back Playlist.txt',
"./playlists/Mammas, Don't Let Your Babies Grow Up to Be Cowboys.txt",
'./playlists/Growing up 90s: Canadian Rock Radio.txt',
'./playlists/60s - british invasion.txt',
"./playlists/British Invasion (60's).txt",
'./playlists/Metal, Rock & Alternative (happy mix).txt',
'./playlists/GROWING UP.txt',
'./playlists/HITS 2000+.txt',
'./playlists/Lowrider oldies.txt',
'./playlists/Wedding Dance Music 2018.txt',
'./playlists/Winter playlist.txt',
'./playlists/60s, 70s, 80s Classic Rock.txt',
# './playlists/British Punk.txt',
'./playlists/Good old songs .txt',
'./playlists/Kisstory - Old School Anthems.txt',
'./playlists/falling in love at a coffee shop.txt',
'./playlists/90s Rock Renaissance.txt',
'./playlists/Middle School Panty Droppers.txt',
'./playlists/Good "old" party hits.txt',
'./playlists/96.3 Easy Rock.txt',
'./playlists/God Save the Queen - The Best of British Post-Punk, New Wave, Madchester, and Brit Pop.txt',
'./playlists/Middle School Dance Playlist.txt',
'./playlists/Baladas Americanas.txt',
'./playlists/middle school throwbacks.txt',
'./playlists/Empowering Breakup Songs .txt',
"./playlists/JOJO'S BIZARRE ADVENTURE.txt",
'./playlists/Middle School Punk Rock Boner Jams.txt',
'./playlists/Happy rock car....txt',
'./playlists/Summer Hits of the 2000s.txt',
'./playlists/Oldies....txt',
"./playlists/Barb's Spotify.txt",
'./playlists/Best of British Punk.txt',
'./playlists/old songs.txt',
'./playlists/Workday: Rock Classics.txt',
'./playlists/Classic Rock.txt',
'./playlists/Disney 2000s Oldies Hits -  Various Artists.txt',
'./playlists/Funk, Soul, Classic Rock and Happy Blues.txt'
]

def song_key(artist, title):
    return artist.lower() + ', ' + title.lower()
