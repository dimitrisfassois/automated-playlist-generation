
import os
import glob
import pandas as pd
from sklearn import linear_model
import numpy as np
import ast
from collections import defaultdict



os.chdir("C:\\Users\\fade7001\\Documents\\Resources\\CS229\\CS229 Project")

from util import *

import sys
import csv
maxInt = sys.maxsize
decrement = True

while decrement:
    # decrease the maxInt value by factor 10 
    # as long as the OverflowError occurs.

    decrement = False
    try:
        csv.field_size_limit(maxInt)
    except OverflowError:
        maxInt = int(maxInt/10)
        decrement = True
        
songs = pd.read_csv("A_N_lda_hmm3.csv", engine='python')



# evaluates our algorithm against a particular playlist
playlist_song_titles = {}
with open('./playlists/60s, 70s, 80s Classic Rock.txt', 'r') as inFile:
    lines = [line.rstrip('\n') for line in inFile]
    for line in lines:
        playlist_song_titles[line.lower()] = True

playlist_songs = {}
neg_examples = [] # random selection of negative examples
i = 0

for index, _ in songs.iterrows():
    song = songs.iloc[index]

    if not ast.literal_eval(song['audio_features'])[0]:
        continue

    key = song['song_artist_title'] 

    if key in playlist_song_titles:
        playlist_songs[key] = (flatten_song(song))
    elif i < 100:
        i = i + 1
        neg_examples.append(flatten_song(song))

playlist = list(playlist_songs.values())
playlist_len = len(playlist)
mid = playlist_len * 3 / 4
print( 'We have ' + str(playlist_len) + ' songs')

pos_train = playlist[0:int(mid)] # 3/4 of playlist_songs
neg_train = neg_examples[0:int(mid)] # random songs not in playlist

pos_test = playlist[int(mid):playlist_len] # other 1/4 of playlist_songs
neg_test = neg_examples[int(mid):playlist_len] # random songs not in playlist

x_train = pos_train + neg_train
y_train = [ 1 for x in range(len(pos_train))] + [ 0 for x in range(len(neg_train))]

x_train = np.array(x_train)
y_train = np.array(y_train)

x_test = pos_test + neg_test
y_test = [ 1 for x in range(len(pos_test))] + [ 0 for x in range(len(neg_test))]

x_test = np.array(x_test)
y_test = np.array(y_test)

model = linear_model.LogisticRegression(C=1e5)

model.fit(x_train, y_train)

train_preds = model.predict(x_train)
print('Train accuracy')
print((train_preds == y_train).sum()/y_train.shape[0])
print(round(float(sum(train_preds)) / float(len(x_train)), 2))

test_preds = model.predict(x_test)
print('Test accuracy')
print((test_preds == y_test).sum()/y_test.shape[0])




############## Append model evaluation script here since it's using the same data! ######

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 20:55:47 2017

@author: dimit_000
"""

from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import cross_val_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve
from sklearn.learning_curve import validation_curve
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.cross_validation import StratifiedKFold

# Split dataset in train, test 
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

# Tuning hyperparameters for logistic regression
pipe_logistic = Pipeline([('scl', StandardScaler()), 
                     ('clf', LogisticRegression(penalty='l2'))])
    
param_grid = {'clf__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }

gs = GridSearchCV(estimator = pipe_logistic,
                  param_grid = param_grid,
                  scoring = 'accuracy',
                  cv = 10,
                  n_jobs = -1)

gs.fit(x_train, y_train)
clf = gs.best_estimator_

gs = gs.fit(x_train, y_train)
print(gs.best_score_)
print(gs.best_params_)

clf.fit(x_train, y_train)

print('Train accuracy %.3f' %clf.score(x_train, y_train))

print('Test accuracy %.3f' %clf.score(x_test, y_test))




# Tuning hyperparameters for svc via grid search
pipe_svc = Pipeline([('scl', StandardScaler()), 
                     ('clf', SVC(random_state=1, probability=True))])

param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

param_grid = [{'clf__C': param_range,
               'clf__kernel':['linear']},
                {'clf__C': param_range,
                'clf__gamma': param_range,
                'clf__kernel': ['rbf']}]

gs = GridSearchCV(estimator = pipe_svc,
                  param_grid = param_grid,
                  scoring = 'accuracy',
                  cv = 10,
                  n_jobs = -1)
           
# Accuracy 
scores = cross_val_score(gs, x_train, y_train, scoring = 'accuracy', cv = 5)
print('CV accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))
print('Trainaccuracy %.3f' %clf.score(x_train, y_train))

gs = gs.fit(x_train, y_train)
print(gs.best_score_)
print(gs.best_params_)

clf = gs.best_estimator_
clf.fit(x_train, y_train)
print('Train accuracy %.3f' %clf.score(x_train, y_train))

print('Test accuracy %.3f' %clf.score(x_test, y_test))

# Learning curves
# pipe_lr = Pipeline([
#                    ('scl', StandardScaler()),
#                    ('clf', LogisticRegression(
#                            penalty = 'l2', random_state = 0))])

indeces = np.random.permutation(range(y_train.shape[0]))
train_sizes, train_scores, test_scores = learning_curve(
                            estimator = clf,
                            X = x_train[indeces],
                            y = y_train[indeces],
                            train_sizes = np.linspace(0.5, 1.0,5),
                            cv = 10,
                            n_jobs = 1)
                            
                            
train_mean = np.mean(train_scores, axis = 1)
train_std = np.std(train_scores, axis = 1)
test_mean = np.mean(test_scores, axis = 1)
test_std = np.std(test_scores, axis = 1)
plt.plot(train_sizes, train_mean,
         color = 'blue', marker = 'o',
         label = 'training accuracy')
plt.fill_between(train_sizes,
                 train_mean + train_std,
                 train_mean - train_std,
                 alpha = 0.15, color = 'blue')

plt.plot(train_sizes, test_mean,
         color = 'green', linestyle = '--',
         marker = 's',
         label = 'validation accuracy')
plt.fill_between(train_sizes,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha = 0.15, color = 'blue')
plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.8, 1.0])
plt.savefig('learning_curve.png')
plt.show()



# Validation curves
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

train_scores, test_scores = validation_curve(
                            estimator = clf,
                            X = x_train,
                            y = y_train,
                            param_name = 'clf__C',
                            param_range = param_range,
                            cv = 10)
                            
train_mean = np.mean(train_scores, axis = 1)
train_std = np.std(train_scores, axis = 1)
test_mean = np.mean(test_scores, axis = 1)
test_std = np.std(test_scores, axis = 1)

plt.plot(param_range, train_mean,
         color = 'blue', marker = 'o',
         label = 'training accuracy')
plt.fill_between(param_range,
                 train_mean + train_std,
                 train_mean - train_std,
                 alpha = 0.15, color = 'blue')

plt.plot(param_range, test_mean,
         color = 'green', linestyle = '--',
         marker = 's',
         label = 'validation accuracy')
plt.fill_between(param_range,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha = 0.15, color = 'blue')
plt.grid()
plt.xscale('log')
plt.xlabel('Parameter C')
plt.legend(loc = 'lower right')
plt.ylabel('Accuracy')
plt.ylim([0.8, 1.0])
plt.savefig('validation_curve.png')
plt.show()

# Confusion matrix
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
confmat = confusion_matrix(y_true = y_test, y_pred = y_pred)

fig, ax = plt.subplots(figsize =(2.5,2.5))
ax.matshow(confmat, cmap = plt.cm.Blues, alpha = 0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i,
                s = confmat[i, j],
                va = 'center', ha = 'center')

plt.xlabel('predicted label')
plt.ylabel('true label')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()


# Precision, recall, F1

print('Precision: %.3f' % precision_score(
                y_true = y_test, y_pred = y_pred))
                
print('Recall: %.3f' % recall_score(
                y_true = y_test, y_pred = y_pred))
                
print('F1: %.3f' % f1_score(
                y_true = y_test, y_pred = y_pred))
                

# ROC curve
cv = StratifiedKFold(y_train,
                     n_folds = 3,
                     random_state = 1)

fig = plt.figure(figsize=(7,5))
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []

for i, (train, test) in enumerate(cv):
    probas = clf.fit(x_train[train], y_train[train]).predict_proba(x_train[test])
    fpr, tpr, thresholds = roc_curve(y_train[test], probas[:,1], pos_label=1)
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr,
             tpr,
             lw=1,
             label = 'ROC fold %d (area - %0.2f)' %(i+1, roc_auc))
plt.plot([0, 1],
         [0, 1],
        linestyle = '--',
        color = (0.6, 0.6, 0.6),
        label = 'random guessing')

mean_tpr /= len(cv)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, 'k--',
         label = 'mean ROC (area = %0.2f)' %mean_auc, lw=2)
plt.plot([0, 0, 1],
         [0, 0, 1],
        lw =2,
        linestyle = ':',
        color = 'black',
        label = 'perfect performance')
        
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.title('Receiver Operator Characteristic')
plt.legend(loc = "lower right")
plt.tight_layout()
plt.savefig('ROC_curve.png')
plt.show()




# Select playlists with more than 30 songs overlap
msd_song_titles = {}

for index, _ in songs.iterrows():
    key = songs.loc[index, 'song_artist_title']
    msd_song_titles[key] = True

msd_song_titles = songs['song_artist_title']
playlist_song_titles = {}
for root, dirs, files in os.walk('./playlists'):
    files = glob.glob(os.path.join(root, '*.txt'))
    for f in files:
        playlist_name = f[12:-4]
        playlist_name = playlist_name.encode(encoding='ascii', errors='ignore')
        with open(f, 'r', encoding="utf8") as inFile:
            lines = [line.rstrip('\n') for line in inFile]
            overlap = 0
            for line in lines:
                if line.lower() in msd_song_titles.unique():
                    overlap = overlap + 1
            if overlap > 50:
                print('Playlist: ' + str(f))
                print(overlap)
                playlists_songs=[]
                for line in lines:
                    if line.lower() in msd_song_titles.unique():
                        playlists_songs.append(line.lower())
                playlist_song_titles[playlist_name] = playlists_songs
                
                
playlist_songs = defaultdict(list)

for index, playlist in enumerate(playlist_song_titles):
    for playlist_song in playlist_song_titles[playlist]:
        msd_song = songs.loc[songs.song_artist_title==playlist_song]
        if not msd_song['audio_features'].tolist():
            continue
        audio_features = msd_song['audio_features'].tolist()
        audio_features = audio_features[0]
        
        songArray = []
        songArray.append(float(msd_song['sentiment_score']) / 4) # 0/1 is too extreme
        songArray.append(float(msd_song['popularity']))
        songArray.append(float(msd_song['lda_probs_topic_1']))
        songArray.append(float(msd_song['lda_probs_topic_2']))
        songArray.append(float(msd_song['lda_probs_topic_3']))
        songArray.append(float(msd_song['hidden_path_avg']))
    
        songArray.append(normalize(msd_song['year'], MIN_YEAR, MAX_YEAR))
        
        song_audio_features = ast.literal_eval(audio_features)[0]
        audio_features = ['acousticness', 'tempo', 'instrumentalness', 'liveness', 'speechiness', 'valence', 'danceability']
        
        if not song_audio_features:
            continue
        for feature in audio_features:
            if feature == 'tempo':
                songArray.append(normalize(song_audio_features['tempo'], MIN_TEMPO, MAX_TEMPO))
            else:
                songArray.append(float(song_audio_features[feature]))
    
        playlist_songs[index].append(songArray)
        
X = []
y = []

for y_playlist, data in playlist_songs.items():
    for x in data:
        y.extend([y_playlist])
        X.extend([x])

X = np.array(X)
y = np.array(y)

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.cross_validation import train_test_split

x_train, x_test, y_train, y_test = \
    train_test_split(X, y,
    test_size = 0.2, random_state = 0)
    
# Tuning hyperparameters via grid search
model_to_set = OneVsRestClassifier(SVC(random_state=1, probability=True))
    
sc = StandardScaler()
sc.fit(x_train)
x_train_std = sc.transform(x_train)
x_test_std = sc.transform(x_test)

param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

param_grid = [{'estimator__C': param_range,
               'estimator__kernel':['linear']},
                {'estimator__C': param_range,
                'estimator__gamma': param_range,
                'estimator__kernel': ['rbf']}]

gs = GridSearchCV(estimator = model_to_set,
                  param_grid = param_grid,
                  scoring = 'accuracy',
                  cv = 5,
                  n_jobs = -1)

scores = cross_val_score(gs, x_train_std, y_train, scoring = 'accuracy', cv = 5)
print('CV accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))
print('Trainaccuracy %.3f' %clf.score(x_train, y_train))

gs.fit(x_train_std, y_train)

print(gs.best_score_)
print(gs.best_params_)

clf = gs.best_estimator_
print('Trainaccuracy %.3f' %clf.score(x_train_std, y_train))

print('Test accuracy %.3f' %clf.score(x_test_std, y_test))


example_model = OneVsRestClassifier(SVC(random_state=1, probability=True, kernel='rbf'))
example_model.fit(x_train_std, y_train)
preds=example_model.predict(x_train_std)
(preds==y_train).sum()/y_train.shape[0]
test_preds=example_model.predict(x_test_std)
(test_preds==y_test).sum()/y_test.shape[0]


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=20)

# use a full grid over all parameters
param_grid = {"n_estimators": [20, 50, 100],
                "max_depth": [3, None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

grid_search  = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)

grid_search.fit(x_train_std, y_train)

print(grid_search.best_score_)
print(grid_search.best_params_)

clf = grid_search.best_estimator_
print('Trainaccuracy %.3f' %clf.score(x_train_std, y_train))
print('Test accuracy %.3f' %clf.score(x_test_std, y_test))




importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(x_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

xlabels = ['sentiment_score', 'popularity', 'lda_probs_topic_1', 'lda_probs_topic_2', 'lda_probs_topic_3', 'hidden_path_avg', 'year', 'acousticness', 'tempo', 'instrumentalness', 'liveness', 'speechiness', 'valence', 'danceability']
xlabels = [xlabels[i] for i in indices]

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
ax = plt.gca()
plt.setp(ax.get_xticklabels(), fontsize=10, rotation='vertical')
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), xlabels)
plt.xlim([-1, X.shape[1]])
plt.tight_layout()
plt.savefig('feature_importance2.png')
plt.show()



# It is important to train the ensemble of trees on a different subset
# of the training data than the linear regression model to avoid
# overfitting, in particular if the total number of leaves is
# similar to the number of training samples
X_train, X_train_lr, y_train_2, y_train_lr = train_test_split(x_train,
                                                            y_train,
                                                            test_size=0.5)

#### Feature transformations with ensembles of trees
# Unsupervised transformation based on totally random trees
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

rt = RandomTreesEmbedding(max_depth=3, n_estimators=100,
    random_state=0)

rt_lm = OneVsRestClassifier(LogisticRegression())
pipeline = make_pipeline(rt, rt_lm)
pipeline.fit(X_train, y_train_2)

y_pred_rt = pipeline.predict_proba(x_test)
y_pred_rt = np.argmax(y_pred_rt, axis = 1)
accuracy = (y_pred_rt == y_test).sum() / y_test.shape[0]

n_estimator = 100
# Supervised transformation based on random forests
sc_X = StandardScaler()
sc_X.fit(X_train)
X_train_std = sc.transform(x_train)

rf = RandomForestClassifier(max_depth=3, n_estimators=n_estimator)
rf_enc = OneHotEncoder()
rf_lm = OneVsRestClassifier(LogisticRegression())
rf.fit(X_train, y_train_2)
rf_enc.fit(rf.apply(X_train))
rf_lm.fit(rf_enc.transform(rf.apply(X_train_lr)), y_train_lr)

y_pred_rf_lm = rf_lm.predict_proba(rf_enc.transform(rf.apply(x_test)))
y_pred_rf_lm = np.argmax(y_pred_rf_lm, axis = 1)
accuracy_rf_lm  = (y_pred_rf_lm == y_test).sum() / y_test.shape[0]


from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import six
from sklearn.base import clone
from sklearn.pipeline import _name_estimators
import numpy as np
import operator


class MajorityVoteClassifier(BaseEstimator, 
                             ClassifierMixin):
    """ A majority vote ensemble classifier

    Parameters
    ----------
    classifiers : array-like, shape = [n_classifiers]
      Different classifiers for the ensemble

    vote : str, {'classlabel', 'probability'} (default='label')
      If 'classlabel' the prediction is based on the argmax of
        class labels. Else if 'probability', the argmax of
        the sum of probabilities is used to predict the class label
        (recommended for calibrated classifiers).

    weights : array-like, shape = [n_classifiers], optional (default=None)
      If a list of `int` or `float` values are provided, the classifiers
      are weighted by importance; Uses uniform weights if `weights=None`.

    """
    def __init__(self, classifiers, vote='classlabel', weights=None):

        self.classifiers = classifiers
        self.named_classifiers = {key: value for key, value
                                  in _name_estimators(classifiers)}
        self.vote = vote
        self.weights = weights

    def fit(self, X, y):
        """ Fit classifiers.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Matrix of training samples.

        y : array-like, shape = [n_samples]
            Vector of target class labels.

        Returns
        -------
        self : object

        """
        if self.vote not in ('probability', 'classlabel'):
            raise ValueError("vote must be 'probability' or 'classlabel'"
                             "; got (vote=%r)"
                             % self.vote)

        if self.weights and len(self.weights) != len(self.classifiers):
            raise ValueError('Number of classifiers and weights must be equal'
                             '; got %d weights, %d classifiers'
                             % (len(self.weights), len(self.classifiers)))

        # Use LabelEncoder to ensure class labels start with 0, which
        # is important for np.argmax call in self.predict
        self.lablenc_ = LabelEncoder()
        self.lablenc_.fit(y)
        self.classes_ = self.lablenc_.classes_
        self.classifiers_ = []
        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(X, self.lablenc_.transform(y))
            self.classifiers_.append(fitted_clf)
        return self

    def predict(self, X):
        """ Predict class labels for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Matrix of training samples.

        Returns
        ----------
        maj_vote : array-like, shape = [n_samples]
            Predicted class labels.
            
        """
        if self.vote == 'probability':
            maj_vote = np.argmax(self.predict_proba(X), axis=1)
        else:  # 'classlabel' vote

            #  Collect results from clf.predict calls
            predictions = np.asarray([clf.predict(X)
                                      for clf in self.classifiers_]).T

            maj_vote = np.apply_along_axis(
                                      lambda x:
                                      np.argmax(np.bincount(x,
                                                weights=self.weights)),
                                      axis=1,
                                      arr=predictions)
        maj_vote = self.lablenc_.inverse_transform(maj_vote)
        return maj_vote

    def predict_proba(self, X):
        """ Predict class probabilities for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        avg_proba : array-like, shape = [n_samples, n_classes]
            Weighted average probability for each class per sample.

        """
        probas = np.asarray([clf.predict_proba(X)
                             for clf in self.classifiers_])
        avg_proba = np.average(probas, axis=0, weights=self.weights)
        return avg_proba

    def get_params(self, deep=True):
        """ Get classifier parameter names for GridSearch"""
        if not deep:
            return super(MajorityVoteClassifier, self).get_params(deep=False)
        else:
            out = self.named_classifiers.copy()
            for name, step in six.iteritems(self.named_classifiers):
                for key, value in six.iteritems(step.get_params(deep=True)):
                    out['%s__%s' % (name, key)] = value
            return out
        
        
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier 

clf1 =  OneVsRestClassifier(LogisticRegression(penalty='l2', 
                          C=0.01,
                          random_state=0))

clf2 = OneVsRestClassifier(SVC(random_state=1, probability=True, kernel='rbf', C=10))

clf3 = DecisionTreeClassifier(max_depth=3,
                              criterion='entropy',
                              random_state=0)

clf4 = grid_search.best_estimator_

clf5 = KNeighborsClassifier(n_neighbors=3,
                            p=2,
                            metric='minkowski')

pipe1 = Pipeline([['sc', StandardScaler()],
                  ['clf', clf1]])
pipe2 = Pipeline([['sc', StandardScaler()],
                  ['clf', clf2]])
    
pipe3 = Pipeline([['sc', StandardScaler()],
                  ['clf', clf5]])

clf_labels = ['Logistic Regression', 'SVC', 'Decision Tree', 'Random Forest', 'KNN']


mv_clf = MajorityVoteClassifier(classifiers=[pipe1, pipe2, clf3, clf4, pipe3])

clf_labels += ['Majority Voting']
all_clf = [pipe1, pipe2, clf3, clf4, pipe3, mv_clf]

for clf, label in zip(all_clf, clf_labels):
    scores = cross_val_score(estimator=clf,
                             X=x_train_std,
                             y=y_train,
                             cv=5,
                             scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]"
          % (scores.mean(), scores.std(), label))