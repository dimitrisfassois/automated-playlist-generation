import os
import glob
import pandas as pd

from util import *

# evaluates our algorithm against a particular playlist

msd = [
    '/Users/kade/LocalDocs/A-enhanced-trimmed.csv',
    '/Users/kade/LocalDocs/B-enhanced-trimmed.csv',
    '/Users/kade/LocalDocs/C-enhanced-trimmed.csv'
]

def song_key(artist, title):
    return artist.lower() + ', ' + title.lower()

for root, dirs, files in os.walk('./playlists'):
    files = glob.glob(os.path.join(root, '*.txt'))
    for f in files:
        playlist_song_titles = {}

        with open(f, 'r') as inFile:
            lines = [line.rstrip('\n') for line in inFile]
            print 'Playlist: ' + str(f)
            for line in lines:
                playlist_song_titles[line.lower()] = True

        playlist_songs = {}
        for subset_file in msd:
            songs = pd.read_csv(subset_file)
            for index, _ in songs.iterrows():
                song = songs.iloc[index]
                key = song_key(song['artist_name'],song['title'])

                if key in playlist_song_titles:
                    playlist_songs[key] = (flatten_song(song))

        print len(playlist_songs.keys())

# TODO assign songs to positive/negative examples

# TODO do the CLASSIFICATIONS
