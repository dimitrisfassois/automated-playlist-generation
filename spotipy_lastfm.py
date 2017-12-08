from __future__ import division

# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 14:36:49 2017

@author: dimit_000
"""

import spotipy
import pprint
from spotipy.oauth2 import SpotifyClientCredentials
from util import decode_song
import os
import glob
# from bs4 import BeautifulSoup
import requests
import pandas as pd
import ast
import pprint
pp = pprint.PrettyPrinter(indent=4)

last_fm_username = "Muse_"
last_fm_key = '48e2003d5cdcfa7cf07f3fdfc60d074d'

API_URL = "http://ws.audioscrobbler.com/2.0/"
params = {"api_key": last_fm_key}
r = requests.get('http://ws.audioscrobbler.com/2.0/', params=params)

# html = BeautifulSoup(r.text)

pp = pprint.PrettyPrinter(indent=4)

client_credentials_manager = SpotifyClientCredentials(client_id='b2707e3822454b72a26f8a321607727b', client_secret='2d543d57fa6747d798a3c595acde37bb')
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

#load songs
songs = []
for root, dirs, files in os.walk('/Volumes/TIME/msd/L'):
    files = glob.glob(os.path.join(root, '*.txt'))
    for f in files:
        with open(f, 'r') as inFile:
            lines = [line.rstrip('\n') for line in inFile]
            for line in lines:
                song = decode_song(line)
                songs.append(song)


# Add popularity from Spotify and tags from Last.fm
i = 0
screwupCount = 0
for song in songs:
    print str(i) + '/' + str(len(songs))
    print song['title']
    i = i + 1

    song['segments_timbre'] = ast.literal_eval(song['segments_timbre'])

    artistName = song['artist_name']
    trackName = song['title']

    try:
        result = sp.search(q="artist:%s track:%s" %(artistName, trackName), type="track", limit=1)

        if result['tracks']['items']:
            topResult = result['tracks']['items'][0]

            song['popularity'] = topResult['popularity'] / 100
            song_uri = topResult['uri']

            song['audio_features'] = sp.audio_features(tracks=[song_uri])
            album = sp.album(topResult['album']['uri'])

            song['release_date'] = album['release_date']
            song['release_date_precision'] = album['release_date_precision']
    except:
        print 'SOMETHING WENT WRONG WITH ' + str(artistName) + ', ' + str(trackName)
        song['artist_name'] = 'ERROR'
        song['title'] = 'ERROR'
        screwupCount = screwupCount + 1

    # comment this out to speed things up for now since we aren't using the tags

    # params = {"api_key": last_fm_key, "track": trackName, "artist": artistName, "method":"track.getInfo", "user": last_fm_username}
    # r = requests.get('http://ws.audioscrobbler.com/2.0/', params=params)
    #
    # top_tags = []
    # tags = []

    # html = BeautifulSoup(r.text)
    # for tag in html.findAll('tag'):
    #     tag = tag.find('name').contents
    #     top_tags.extend(tag)
    #
    # params = {"api_key": last_fm_key, "track": trackName, "artist": artistName, "method":"track.getTags", "user": last_fm_username}
    # r = requests.get('http://ws.audioscrobbler.com/2.0/', params=params)
    # for tag in html.findAll('tag'):
    #     tag = tag.find('name').contents
    #     if tag:
    #         tags.extend(tag)

    # all_tags = list(set(tags).union(set(top_tags)))
    #
    # song['tags'] = all_tags

output = pd.DataFrame(songs)
output.to_csv("/Volumes/TIME/msd/L.csv")

print 'SCREWUP COUNT: ' + str(screwupCount)
