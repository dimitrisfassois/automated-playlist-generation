keys = ['song_id', 'title', 'segments_timbre', 'energy', 'tempo', 'artist_name', 'sentiment_score', 'danceability', 'year']
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
