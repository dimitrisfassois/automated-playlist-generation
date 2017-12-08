import pandas as pd
import tables
import os
import glob

songs = pd.read_csv("/Users/kade/LocalDocs/F.csv")

def open_h5_file_read(h5filename):
    return tables.open_file(h5filename, mode='r+')

def get_song_id(h5,songidx=0):
    return h5.root.metadata.songs.cols.song_id[songidx]

def get_track_id(h5,songidx=0):
    return h5.root.analysis.songs.cols.track_id[songidx]

def get_artist_name(h5,songidx=0):
    return h5.root.metadata.songs.cols.artist_name[songidx]

def get_title(h5,songidx=0):
    return h5.root.metadata.songs.cols.title[songidx]

def get_num_songs(h5):
    return h5.root.metadata.songs.nrows

obj = {}
cnt = 0
for root, dirs, files in os.walk(os.path.normpath('/Volumes/TIME/msd-zip/F')):
    files = glob.glob(os.path.join(root, '*.h5'))
    for f in files :
        h5File = open_h5_file_read(f)
        n = get_num_songs(h5File)
        for i in range(0, n):
            cnt = cnt + 1
            song_id = get_song_id(h5File, i)
            artist_name = get_artist_name(h5File, i)
            title = get_title(h5File, i)
            key = song_id + artist_name + title
            obj[key] = get_track_id(h5File, i)
            if cnt % 100 == 0:
                print cnt
        h5File.close()

print 'Finished reading files'
print len(obj.keys())

def find_track(song):
    key = song['song_id'] + song['artist_name'] + song['title']
    return obj[key]

enhanced_songs = []
for index, _ in songs.iterrows():
    song = songs.iloc[index].to_dict()
    if song['artist_name'] == 'ERROR':
        continue
    song['track_id'] = find_track(song)
    enhanced_songs.append(song)

output = pd.DataFrame(enhanced_songs)
output.to_csv("/Users/kade/LocalDocs/F-enhanced.csv")
