"""
Crawls a directory of hdf files, converting them to txts and removing songs
for which we don't have lyrics

@author: Kade Keith
"""

import tables
import os
import glob
import pprint

from sentiment import get_sentiment_score
from util import encode_song

pp = pprint.PrettyPrinter(indent=4)

def open_h5_file_read(h5filename):
    return tables.open_file(h5filename, mode='r+')

def get_num_songs(h5):
    return h5.root.metadata.songs.nrows

def get_artist_location(h5,songidx=0):
    return h5.root.metadata.songs.cols.artist_location[songidx]

def get_artist_latitude(h5,songidx=0):
    return h5.root.metadata.songs.cols.artist_latitude[songidx]

def get_artist_longitude(h5,songidx=0):
    return h5.root.metadata.songs.cols.artist_longitude[songidx]

def get_artist_name(h5,songidx=0):
    return h5.root.metadata.songs.cols.artist_name[songidx]

def get_title(h5,songidx=0):
    return h5.root.metadata.songs.cols.title[songidx]

def get_song_id(h5,songidx=0):
    return h5.root.metadata.songs.cols.song_id[songidx]

def get_year(h5, songidx=0):
    return h5.root.musicbrainz.songs.cols.year[songidx]

def get_track_id(h5,songidx=0):
    return h5.root.analysis.songs.cols.track_id[songidx]

def get_segments_timbre(h5,songidx=0):
    segments_timbre = h5.root.analysis.segments_timbre
    idx_segments_timbre = h5.root.analysis.songs.cols.idx_segments_timbre

    if h5.root.analysis.songs.nrows == songidx + 1:
        return segments_timbre[idx_segments_timbre[songidx]:,:]

    return segments_timbre[idx_segments_timbre[songidx]:idx_segments_timbre[songidx+1],:]

def apply_to_n_files(basedir, n, func=lambda x: x):
    cnt = 0
    for root, dirs, files in os.walk(basedir):
        files = glob.glob(os.path.join(root, '*.h5'))
        for f in files :
            cnt += 1
            if cnt > n :
                return n
            # print 'Looking through file: ' + f
            func(f)
            print cnt
    return cnt

def clean_data():
    existingFieldnames = [
        'artist_name',
        'title',
        'year',
        'song_id'
    ]

    def toTxt(filename):
        h5File = open_h5_file_read(filename)
        n = get_num_songs(h5File)
        didWrite = False
        with open(filename + '.txt', 'w+') as out:
            for i in range(0, n):
                isValid = True
                song = {}

                # grab the scalar fields from MSD
                for field in existingFieldnames:
                    data = globals()['get_' + field](h5File, i)
                    song[field] = data

                #run nlp
                song['sentiment_score'] = get_sentiment_score(get_track_id(h5File, i))
                if song['sentiment_score'] != 0 and song['sentiment_score'] != 1:
                    isValid = False # this means we don't have lyrics

                song['segments_timbre'] = get_segments_timbre(h5File, i).tolist()

                if isValid:
                    print 'Writing valid song'
                    txt = encode_song(song)
                    out.write(txt + '\n')
                    didWrite = True
                else:
                    print "Missing lyrics. Skipping"

        # save output file
        h5File.close()
        out.close()

        #delete the hdf5 file
        os.remove(filename)
        if not didWrite:
            os.remove(filename + '.txt')

    apply_to_n_files(os.path.normpath('/Volumes/TIME/msd/O'), 10000000, toTxt)

clean_data()
