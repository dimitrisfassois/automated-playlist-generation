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
for line in f.xreadlines():
    if line == '':
        continue
    if line[0] == '%':
        topwords = line.strip()[1:].split(',')
        f.close()
        break

def get_sentiment_score(song_id):
    f = open(trainf, 'r')
    cnt_lines = 0
    for line in f.xreadlines():
        if line == '' or line.strip() == '':
            continue
        if line[0] in ('#', '%'):
            continue
        lineparts = line.strip().split(',')
        tid = lineparts[0]
        if tid == song_id:
            bag = {}
            for wordcnt in lineparts[2:]:
                wordid, cnt = wordcnt.split(':')
                bag[topwords[int(wordid)]] = True
            distribution = classifier.prob_classify(bag)
            if distribution.prob('pos') > distribution.prob('neg'):
                return 1
            else:
                return 0

    f.close()
