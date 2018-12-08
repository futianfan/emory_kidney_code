from __future__ import print_function 
import numpy as np
import pandas as pd
import time
from itertools import combinations
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from collections import defaultdict
import argparse
from sklearn.metrics import roc_auc_score
import vfdt 

#######################################################################################
### data prepare
#######################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--train_feature', type=str, help='feature file')
parser.add_argument('--train_label', type=str, help='label file')
parser.add_argument('--test_feature', type=str, help='feature file')
parser.add_argument('--test_label', type=str, help='label file')
parser.add_argument('--num', type=int, help='feature num')
parser.add_argument('--output_assign', type=str, help='feature num')
args = parser.parse_args()

feature_file = open(args.train_feature, 'r')
label_file = open(args.train_label, 'r')
test_feature_file = open(args.test_feature, 'r')
test_label_file = open(args.test_label, 'r')
feature_dim = args.num
assign_file = open(args.output_assign, 'w')

feature_lines = feature_file.readlines()
label_lines = label_file.readlines()
test_feature_lines = test_feature_file.readlines()
test_label_lines = test_label_file.readlines()


train_mat = np.zeros((len(feature_lines), feature_dim + 1), dtype = float)
for i, line in enumerate(feature_lines):
    line = line.rstrip().split() 
    line = [int(i) for i in line]
    for j in line:
        train_mat[i,j] += 1
    lab = int(label_lines[i])
    train_mat[i, -1] = lab

test_mat = np.zeros((len(test_feature_lines), feature_dim + 1), dtype = float)
for i, line in enumerate(test_feature_lines):
    line = line.rstrip().split() 
    line = [int(i) for i in line]
    for j in line:
        test_mat[i,j] += 1
    lab = int(test_label_lines[i])
    test_mat[i, -1] = lab

#######################################################################################
### data prepare
#######################################################################################

tree = Vfdt([i for i in range(feature_dim)] , delta=0.01, nmin=70, tau=0.5)  ## 70->155, 
print('initialize tree')
t1 = time.time()
#for i in range(7000):
for i in range(train_mat.shape[0]):
    if i % 100 == 0:
        t2 = time.time()
        print('finish update ' + str(i) + '-th data, cost ' + str(t2 - t1)[:4] + ' seconds')
        t1 = t2 
    tree.update(train_mat[i,:-1], train_mat[i,-1])



y_pred = tree.predict(train_mat[:,:-1])
y_test = train_mat[:,-1]
y_test = list(y_test)
y_pred = list(y_pred)
print(roc_auc_score(y_test, y_pred))


y_pred = tree.predict(test_mat[:,:-1])
y_test = test_mat[:,-1]
y_test = list(y_test)
y_pred = list(y_pred)
print(roc_auc_score(y_test, y_pred))

leaf_node_set = defaultdict(lambda : [])

for i in range(train_mat.shape[0]):
    root_node = tree.root
    _, indx = root_node.sort_example2(train_mat[i,:-1], '')
    leaf_node_set[indx] += [i]

for k in leaf_node_set:
    #print(k)
    v = leaf_node_set[k]
    lst = [str(i) for i in v]
    string = ' '.join(lst)
    assign_file.write(string + '\n')
print(len(leaf_node_set))
assign_file.close()




