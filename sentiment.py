# baseline naive bayes. not multinomial. uses movie review corpus
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews

def word_features(words):
    return dict([(word, True) for word in words])

negative_ids = movie_reviews.fileids('neg')
positive_ids = movie_reviews.fileids('pos')

negative_features = [(word_features(movie_reviews.words(fileids=[f])), 'neg') for f in negative_ids]
positive_features = [(word_features(movie_reviews.words(fileids=[f])), 'pos') for f in positive_ids]

train_features = negative_features + positive_features
classifier = NaiveBayesClassifier.train(train_features)

# create dict of words
trainf = 'mxm_dataset_full.txt'
f = open(trainf, 'r')
songs = {}
for line in f.xreadlines():
    if line == '' or line[0] == '#':
        continue
    elif line[0] == '%':
        topwords = line.strip()[1:].split(',')
    else:
        lineparts = line.strip().split(',')
        bag = {}
        for wordcnt in lineparts[2:]:
            wordid, cnt = wordcnt.split(':')
            # it's 1-based!
            bag[topwords[int(wordid) - 1]] = True
        songs[lineparts[0]] = bag
f.close()

def get_sentiment_score(song_id):
    if song_id in songs:
        distribution = classifier.prob_classify(songs[song_id])
        if distribution.prob('pos') > distribution.prob('neg'):
            return 1
        else:
            return 0

#TODO. Make version of this that runs on pre-processed dataset and updates
