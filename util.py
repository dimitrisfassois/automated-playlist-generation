keys = ['song_id', 'title', 'segments_timbre', 'energy', 'tempo', 'artist_name', 'artist_latitude', 'danceability', 'year', 'sentiment_score', 'artist_longitude']
delimiter = '|||'

#song to txt
def encode_song(song):
    return delimiter.join(map(str, song.values()))

#txt to song
def decode_song(songTxt):
    values = songTxt.split(delimiter)
    i = 0
    song = {}
    for key in keys:
        song[key] = values[i]
        i = i + 1
    return song

def normalize(val, minVal, maxVal):
    return (float(val) - minVal) / maxVal

def flatten_song(song):
    songArray = []
    for key in song:
        songArray.append(float(song[key]))
    return songArray

MIN_TEMPO = 45.508
MAX_TEMPO = 229.864
MIN_YEAR = 1959.0
MAX_YEAR = 2009.0
