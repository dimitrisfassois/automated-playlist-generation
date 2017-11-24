import os
import glob
from sklearn import linear_model
import pandas as pd
import pprint
pp = pprint.PrettyPrinter(indent=4)
\
from util import *

#load songs
# songs = []
# for root, dirs, files in os.walk('/Users/kade/LocalDocs/MillionSongSubset2/data'):
#     files = glob.glob(os.path.join(root, '*.txt'))
#     for f in files:
#         with open(f, 'r') as inFile:
#             lines = [line.rstrip('\n') for line in inFile]
#             for line in lines:
#                 song = decode_song(line)
#                 song['year'] = normalize(song['year'], MIN_YEAR, MAX_YEAR)
#                 song['tempo'] = normalize(song['tempo'], MIN_TEMPO, MAX_TEMPO)
#                 # remove non-numeric attributes
#                 song.pop('song_id', None)
#                 song.pop('title', None)
#                 song.pop('segments_timbre', None)
#                 song.pop('artist_name', None)
#                 songs.append(flatten_song(song))

# TODO ADD INTERCEPT

songs = pd.read_csv("/Users/kade/LocalDocs/subset.csv")
flat_songs = []
i = 0
test_titles = []
name_counts = {}
for index, _ in songs.iterrows():
    song = songs.iloc[index]

    if type(song['audio_features']) is float:
        continue

    # Get indices of certain artist songs
    # if song['artist_name'] == 'Aerosmith':
    #   print song['title']

    if song['artist_name'] == 'Bon Jovi':
        print i
        print song['title']
        print float(song['sentiment_score'])
        print float(song['popularity'])
        pp.pprint(ast.literal_eval(song['audio_features'])[0])

    if song['artist_name'] in name_counts:
        name_counts[song['artist_name']] = name_counts[song['artist_name']] + 1
    else:
        name_counts[song['artist_name']] = 1

    if i > 1695 and i < 1705:
        test_titles.append(song['artist_name'] + ', ' + song['title'])

    flat_songs.append(flatten_song(song))
    i = i + 1

aero_songs = [flat_songs[292], flat_songs[312], flat_songs[843],
    flat_songs[1019], flat_songs[1093], flat_songs[1238],  flat_songs[1458]]

jovi_songs = [flat_songs[9], flat_songs[72], flat_songs[108],
    flat_songs[309], flat_songs[847], flat_songs[1659]]

# arbitrary mock data
x_train = jovi_songs
y_train = [ 1 for x in range(len(jovi_songs))]

# some random negative examples
x_train = x_train + flat_songs[200:210]
y_train = y_train + [ 0 for x in range(10)]

x_test = flat_songs[1695:1705]

model = linear_model.LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

for key in name_counts:
    if name_counts[key] > 5:
        print key + ': ' + str(name_counts[key])

for i in range(9):
    print test_titles[i] + ': ' + str(y_pred[i])

# Aerosmith song indices
# 292, 312, 843, 1019, 1093, 1238, 1458, 1511

# CLASSIFICATIONS FROM THE AEROSMITH TRAINING DATA
# Winds Of Plague, Origins And Endings: 0.386695109939
# The Ataris, Make It Last: 0.44327217148
# Sweet, Neon Psychedelia: 0.948058795077
# Funeral For A Friend, Your Revolution Is A Joke: 0.0841323790295
# The Black Crowes, Good Morning Captain: 0.299697761404
# Aerosmith, Reefer Head Woman: 1.47295049055
# Neil Diamond, Brooklyn On A Saturday Night: 1.07268129368
# OV7, Volvere: -0.399370795516
# Tha Liks, Da Da Da Da: -0.339378432386

# CLASSIFICATIONS FROM THE BON JOVI TRAINING DATA
# Johnny Horton, The Golden Rocket: 0.71933531467
# Roger Miller, Husbands And Wives: -0.0658498377106
# Andy & Lucas, Hasta Los Huesos: -0.10891481801
# Hot Tuna, Hesitation Blues: -0.346326437273
# Olga Tanon, Como Olvidar (Merengue Version): -0.101924816847
# Nick Cave & The Bad Seeds, New Morning (Live): 0.357192781716
# Bon Jovi, Only Lonely: -0.258888975598
# Frost, Take a Ride: 0.128365501962
# Christina Aguilera, Cruz: -0.66089836728
