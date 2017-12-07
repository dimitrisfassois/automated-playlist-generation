import pandas as pd

songs = pd.read_csv("/Users/kade/LocalDocs/A.csv")

f = open('mxm_dataset_full.txt', 'r')
song_lyrics = {}
for line in f.xreadlines():
    if line == '' or line[0] == '#':
        continue
    elif line[0] == '%':
        topwords = line.strip()[1:].split(',')
    else:
        lineparts = line.strip().split(',')
        bag = {}
        id = lineparts[0]
        for wordcnt in lineparts[2:]:
            wordid, cnt = wordcnt.split(':')
            # it's 1-based!
            bag[topwords[int(wordid) - 1]] = True

        song_lyrics[id] = bag
f.close()

def todo(words):
    # TODO
    return words

enhanced_songs = []
for index, _ in songs.iterrows():
    song = songs.iloc[index].to_dict()
    track_id = song['track_id']
    if track_id in song_lyrics:
        words = song_lyrics[track_id]
        song['todo'] = todo(words)
        enhanced_songs.append(song)
    else:
        print song_id
        break

output = pd.DataFrame(enhanced_songs)
output.to_csv("/Users/kade/LocalDocs/A-enhanced.csv")
