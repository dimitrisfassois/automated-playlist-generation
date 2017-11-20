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
from bs4 import BeautifulSoup
import requests

last_fm_username = "Muse_"
last_fm_key = '48e2003d5cdcfa7cf07f3fdfc60d074d'

API_URL = "http://ws.audioscrobbler.com/2.0/" 
params = {"api_key": last_fm_key} 
r = requests.get('http://ws.audioscrobbler.com/2.0/', params=params)

html = BeautifulSoup(r.text)

pp = pprint.PrettyPrinter(indent=4)

client_credentials_manager = SpotifyClientCredentials(client_id='b2707e3822454b72a26f8a321607727b', client_secret='2d543d57fa6747d798a3c595acde37bb')
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

username = 'spotify'

result = sp.search(q="artist:The Box Tops track:Soul Deep", type="track", limit=1)
result['tracks']['items'][0]['popularity']
result['tracks']['items'][0]['popularity']

#load songs
songs = []
for root, dirs, files in os.walk('D:\\Docs\\Stanford - Mining Massive Datasets\\CS229\\Project\\MillionSongSubset\\data'):
    files = glob.glob(os.path.join(root, '*.txt'))
    for f in files:
        with open(f, 'r') as inFile:
            lines = [line.rstrip('\n') for line in inFile]
            for line in lines:
                song = decode_song(line)
                songs.append(song)

# Add popularity from Spotify and tags from Last.fm                
for song in songs:
    artistName = song['artist_name']
    trackName = song['title']
    result = sp.search(q="artist:%s track:%s" %(artistName, trackName), type="track", limit=1)
    if result['tracks']['items']:
        song['popularity'] = result['tracks']['items'][0]['popularity']
    params = {"api_key": last_fm_key, "track": trackName, "artist": artistName, "method":"track.getInfo", "user": last_fm_username} 
    r = requests.get('http://ws.audioscrobbler.com/2.0/', params=params)
    
    top_tags = []
    tags = []
    
    html = BeautifulSoup(r.text)
    for tag in html.findAll('tag'):
        tag = tag.find('name').contents
        '''
        page = urllib2.urlopen(url)
        soup = BeautifulSoup(page.read())
        mydiv = soup.find("h1", { "class" : "header-title" }).contents
        clean_tag = mydiv[0].strip()
        '''
        top_tags.extend(tag)
            
    params = {"api_key": last_fm_key, "track": trackName, "artist": artistName, "method":"track.getTags", "user": last_fm_username} 
    r = requests.get('http://ws.audioscrobbler.com/2.0/', params=params)
    for tag in html.findAll('tag'):
        tag = tag.find('name').contents
        if tag:
            tags.extend(tag)
            
    all_tags = list(set(tags).union(set(top_tags)))
    
    song['tags'] = all_tags