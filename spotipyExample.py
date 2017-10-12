import spotipy
import pprint
import json
import sys
from spotipy.oauth2 import SpotifyClientCredentials

pp = pprint.PrettyPrinter(indent=4)

client_credentials_manager = SpotifyClientCredentials(client_id='b2707e3822454b72a26f8a321607727b', client_secret='2d543d57fa6747d798a3c595acde37bb')
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

username = 'spotify'
allTracks = {}

def saveTracks(tracks):
    for i, trackData in enumerate(tracks):
        track = trackData['track']
        if not track['id'] in allTracks:
            allTracks[track['id']] = track

playlists = sp.user_playlists(username)
while playlists:

    for i, playlist in enumerate(playlists['items']):
        # stop after 10 playlists
        if i > 10:
            with open('data.txt', 'w') as outfile:
                json.dump(allTracks, outfile)
                print 'SAVED ' + str(len(allTracks)) + ' TRACKS'
                sys.exit()

        print("%4d %s %s" % (i + 1 + playlists['offset'], playlist['uri'],  playlist['name']))

        results = sp.user_playlist(username, playlist['id'], fields="tracks,next")
        tracks = results['tracks']

        saveTracks(tracks['items'])
        while tracks['next']:
            tracks = sp.next(tracks)
            saveTracks(tracks['items'])

    if playlists['next']:
        playlists = sp.next(playlists)
    else:
        playlists = None
