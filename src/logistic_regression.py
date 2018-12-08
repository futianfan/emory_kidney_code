import sys
import argparse
import os
import numpy as np
from time import time
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

dataFolder = '/Users/futianfan/Downloads/Gatech_Courses/emory_kidney/data'
train_data = np.load('{}/train.npy'.format(dataFolder))
test_data = np.load('{}/test.npy'.format(dataFolder))
train_label = train_data[:,0]
train_feature = train_data[:,1:]
test_label = test_data[:,0]
test_feature = test_data[:,1:]

print('training num is {}'.format(train_data.shape[0]))
print('test num is {}'.format(test_data.shape[0]))
print('in training, positive num is {}'.format(np.sum(train_label)))
print('in test, positive num is {}'.format(np.sum(test_label)))
lr = LogisticRegression(C=1000.0, random_state=1)
t1 = time()
lr.fit(train_feature, train_label)
prediction = lr.predict_proba(test_feature)
prediction = list(prediction[:,1])
print('accuracy: {} , cost {} seconds'.format(str(roc_auc_score(test_label, prediction)), str(time()-t1)[:5]))


