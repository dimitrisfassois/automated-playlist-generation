import spotipy
import pprint
import json
import sys
from spotipy.oauth2 import SpotifyClientCredentials

pp = pprint.PrettyPrinter(indent=4)

client_credentials_manager = SpotifyClientCredentials(client_id='b2707e3822454b72a26f8a321607727b', client_secret='2d543d57fa6747d798a3c595acde37bb')
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

searchResults = sp.search(sys.argv[1], limit=10, type='playlist')

for i, playlist in enumerate(searchResults['playlists']['items']):

    # don't let weirdly named playlists mess things up
    if '/' in playlist['name']:
        continue

    print("%s %s" % (playlist['uri'],  playlist['name']))

    results = sp.user_playlist(playlist['owner']['id'], playlist['id'], fields="tracks,next")
    tracks = results['tracks']

    with open(playlist['name'] + '.txt', 'w') as out:

        for song in tracks['items']:
            out.write(song['track']['artists'][0]['name'].encode('utf-8') + ', ' + song['track']['name'].encode('utf-8') + '\n')

        while tracks['next']:
            tracks = sp.next(tracks)
            for song in tracks['items']:
                out.write(song['track']['artists'][0]['name'].encode('utf-8') + ', ' + song['track']['name'].encode('utf-8') + '\n')
