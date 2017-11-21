import os
import glob

keys = ['song_id', 'title', 'segments_timbre', 'energy', 'tempo', 'artist_name', 'sentiment_score', 'danceability', 'year']
delimiter = '|||'

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
    for key in song:
        songArray.append(float(song[key]))
    return songArray

MIN_TEMPO = 45.508
MAX_TEMPO = 229.864
MIN_YEAR = 1959.0
MAX_YEAR = 2009.0

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
                    songs[song['artist_name'] + ', ' + song['title']] = True

    for root, dirs, files in os.walk('./playlists'):
        files = glob.glob(os.path.join(root, '*.txt'))
        for f in files:
            with open(f, 'r') as inFile:
                lines = [line.rstrip('\n') for line in inFile]
                for line in lines:
                    if line in songs:
                        hits = hits + 1
                    else:
                        misses = misses + 1

    print 'Hits: ' + str(hits)
    print 'Misses: ' + str(misses)


compute_overlap()
