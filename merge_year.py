# merges the 'release_date' from spotify and 'year' from msd, and ditches song with no spotify info

import pandas as pd

songs = pd.read_csv("/Users/kade/LocalDocs/L-enhanced.csv")
names = []
i = 0
has_data = [];
for index, _ in songs.iterrows():
    i = i + 1
    song = songs.iloc[index]

    # skip songs we didn't get spotify info from
    if type(song['audio_features']) is float:
        continue

    song_dict = song.to_dict()
    song_dict.pop('Unnamed: 0', None)
    song_dict.pop('Unnamed: 0.1', None)
    song_dict.pop('Unnamed: 0.1.1', None)

    if song_dict['year'] == 0:
        song_dict['year'] = song_dict['release_date'][0:4]

    song_dict.pop('release_date', None)
    song_dict.pop('release_date_precision', None)

    has_data.append(song_dict)

output = pd.DataFrame(has_data)
output.to_csv("/Users/kade/LocalDocs/L-enhanced-trimmed.csv")
