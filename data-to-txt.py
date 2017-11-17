"""
Thierry Bertin-Mahieux (2010) Columbia University
tb2332@columbia.edu


This code contains a set of getters functions to access the fields
from an HDF5 song file (regular file with one song or
aggregate / summary file with many songs)

This is part of the Million Song Dataset project from
LabROSA (Columbia University) and The Echo Nest.


Copyright 2010, Thierry Bertin-Mahieux

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import tables
import os
import glob
import csv
import json
import pprint

from sentiment import get_sentiment_score
from util import encode_song

pp = pprint.PrettyPrinter(indent=4)

def open_h5_file_read(h5filename):
    """
    Open an existing H5 in read mode.
    Same function as in hdf5_utils, here so we avoid one import
    """
    return tables.open_file(h5filename, mode='r+')

def get_num_songs(h5):
    """
    Return the number of songs contained in this h5 file, i.e. the number of rows
    for all basic informations like name, artist, ...
    """
    if not 'metadata' in h5.root._v_children:
        return h5.root.dataGroup.dataTable.nrows
    return h5.root.metadata.songs.nrows

def get_artist_location(h5,songidx=0):
    """
    Get artist location from a HDF5 song file, by default the first song in it
    """
    if not 'metadata' in h5.root._v_children:
        return h5.root.dataGroup.dataTable.cols.artist_location[songidx]
    return h5.root.metadata.songs.cols.artist_location[songidx]

def get_artist_name(h5,songidx=0):
    """
    Get artist name from a HDF5 song file, by default the first song in it
    """
    if not 'metadata' in h5.root._v_children:
        return h5.root.dataGroup.dataTable.cols.artist_name[songidx]
    return h5.root.metadata.songs.cols.artist_name[songidx]

def get_title(h5,songidx=0):
    """
    Get title from a HDF5 song file, by default the first song in it
    """
    if not 'metadata' in h5.root._v_children:
        return h5.root.dataGroup.dataTable.cols.title[songidx]
    return h5.root.metadata.songs.cols.title[songidx]

def get_danceability(h5,songidx=0):
    """
    Get danceability from a HDF5 song file, by default the first song in it
    """
    if not 'metadata' in h5.root._v_children:
        return h5.root.dataGroup.dataTable.cols.danceability[songidx]
    return h5.root.analysis.songs.cols.danceability[songidx]

def get_energy(h5,songidx=0):
    """
    Get energy from a HDF5 song file, by default the first song in it
    """
    if not 'metadata' in h5.root._v_children:
        return h5.root.dataGroup.dataTable.cols.energy[songidx]
    return h5.root.analysis.songs.cols.energy[songidx]

def get_tempo(h5,songidx=0):
    """
    Get tempo from a HDF5 song file, by default the first song in it
    """
    if not 'metadata' in h5.root._v_children:
        return h5.root.dataGroup.dataTable.cols.tempo[songidx]
    return h5.root.analysis.songs.cols.tempo[songidx]

def get_song_id(h5,songidx=0):
    """
    Get song id from a HDF5 song file, by default the first song in it
    """
    if not 'metadata' in h5.root._v_children:
        return h5.root.dataGroup.dataTable.cols.song_id[songidx]
    return h5.root.metadata.songs.cols.song_id[songidx]

def get_year(h5, songidx=0):
    """
    Get release year from a HDF5 song file, by default the first song in it
    """
    if not 'metadata' in h5.root._v_children:
        return h5.root.dataGroup.dataTable.cols.year[songidx]
    return h5.root.musicbrainz.songs.cols.year[songidx]

def get_track_id(h5,songidx=0):
    """
    Get track id from a HDF5 song file, by default the first song in it
    """
    return h5.root.analysis.songs.cols.track_id[songidx]

def get_segments_timbre(h5,songidx=0):
    """
    Get segments timbre array. Takes care of the proper indexing if we are in aggregate
    file. By default, return the array for the first song in the h5 file.
    To get a regular numpy ndarray, cast the result to: numpy.array( )
    """
    if not 'metadata' in h5.root._v_children:
        segments_timbre = h5.root.dataGroup.segments_timbre
        idx_segments_timbre = h5.root.dataGroup.dataTable.cols.idx_segments_timbre

        if h5.root.dataGroup.dataTable.nrows == songidx + 1:
            return segments_timbre[idx_segments_timbre[songidx]:,:]

        return segments_timbre[idx_segments_timbre[songidx]:idx_segments_timbre[songidx+1],:]

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
        'danceability',
        'energy',
        'tempo',
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
                    if (field == 'year' or field == 'tempo') and data == 0:
                        isValid = False

                #run nlp
                song['sentiment_score'] = get_sentiment_score(get_track_id(h5File, i))
                if song['sentiment_score'] != 0 and song['sentiment_score'] != 1:
                    isValid = False

                song['segments_timbre'] = get_segments_timbre(h5File, i).tolist()

                #TODO enhance with spotify data

                if isValid:
                    print 'Writing valid song'
                    #write song to line in file
                    txt = encode_song(song)
                    out.write(txt + '\n')
                    didWrite = True
                else:
                    # pp.pprint(song)
                    print "Song missing some data. skipping"

        # save output file
        h5File.close()
        out.close()

        #delete the hdf5 file
        os.remove(filename)
        if not didWrite:
            os.remove(filename + '.txt')

    print apply_to_n_files(os.path.normpath('/Users/kade/LocalDocs/MillionSongSubset2/data'), 400, toTxt)
    # pp.pprint(attrCounts)

clean_data()
