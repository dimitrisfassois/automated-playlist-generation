
import os
import glob
import pandas as pd
from sklearn import linear_model
import numpy as np
import ast

from util import *

os.chdir("C:\\Users\\fade7001\\Documents\\Resources\\CS229\\CS229 Project")

songs = pd.read_csv("A_N_lda.csv")

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

# Tuning hyperparameters via grid search
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

gs = gs.fit(x_train, y_train)
print(gs.best_score_)
print(gs.best_params_)

clf = gs.best_estimator_
clf.fit(x_train, y_train)
print('Test accuracy %.3f' %clf.score(x_test, y_test))

# Best model
clf = gs.best_estimator_
clf.fit(X_train, y_train)
print('Test accuracy: %.3f' % clf.score(X_test, y_test))

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
plt.show



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
plt.savefig('ROC_curve.png')
plt.show()