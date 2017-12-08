import os
import glob
import pandas as pd

from util import *

# prints the count of songs we have data on from each playlist

msd = [
    '/Users/kade/LocalDocs/A-enhanced-trimmed.csv',
    '/Users/kade/LocalDocs/B-enhanced-trimmed.csv',
    '/Users/kade/LocalDocs/C-enhanced-trimmed.csv',
    '/Users/kade/LocalDocs/D-enhanced-trimmed.csv'
]

def song_key(artist, title):
    return artist.lower() + ', ' + title.lower()

msd_song_titles = {}
for subset_file in msd:
    songs = pd.read_csv(subset_file)
    for index, _ in songs.iterrows():
        song = songs.iloc[index]
        key = song_key(song['artist_name'],song['title'])
        msd_song_titles[key] = True

for root, dirs, files in os.walk('./playlists'):
    files = glob.glob(os.path.join(root, '*.txt'))
    for f in files:
        playlist_song_titles = {}

        with open(f, 'r') as inFile:
            lines = [line.rstrip('\n') for line in inFile]
            overlap = 0
            for line in lines:
                if line.lower() in msd_song_titles:
                    overlap = overlap + 1
            if overlap > 2:
                print 'Playlist: ' + str(f)
                print overlap
