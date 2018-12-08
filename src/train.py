
from __future__ import print_function
import pandas as pd
import numpy as np
import pickle 		#import cPickle as pickle
import model 
import torch
from torch import nn, optim 
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
from time import time
from sklearn.metrics import roc_auc_score
torch.manual_seed(1)    # reproducible
np.random.seed(1)

def training(nnet, optimizer, data, label, stt, endn, lossform):
	batch_label = Variable( torch.from_numpy(label[stt:endn]) )
	batch_output = nnet(data[stt:endn,:])
	loss = lossform(batch_output, batch_label)
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	return loss 

def test(nnet, data, label, batch_size, lossform):
	N = data.shape[0]
	iter_in_a_epoch = int(np.ceil(N / batch_size))
	for it in range(iter_in_a_epoch):
		batch_output = nnet(data[it * batch_size: it * batch_size + batch_size])
		output = batch_output if it == 0 else torch.cat([output, batch_output], 0)
	y_pred = [float(output[j][1]) for j in range(output.shape[0])]
	y_label = list(label)
	return roc_auc_score(y_label, y_pred)


def total_learning(batch_size = 4096, EPOCH = 500, lr = 1e-1, lossform = nn.CrossEntropyLoss()):
	### Processing data 
	dataFolder = '/Users/futianfan/Downloads/Gatech_Courses/emory_kidney/data'
	'''
	trainData = pd.read_csv('{}/trainData_1yr.csv'.format(dataFolder))
	testData = pd.read_csv('{}/testData_1yr.csv'.format(dataFolder))
	label_name = 'gf_1year'
	train_label = np.array(trainData[label_name])
	test_label = np.array(testData[label_name])
	trainData.drop([label_name],axis = 'columns',inplace = True)
	trainData = np.array(trainData).astype(np.float)
	testData.drop([label_name],axis = 'columns',inplace = True)
	testData = np.array(testData).astype(np.float)
	'''

	train_data = np.load('{}/train.npy'.format(dataFolder))
	test_data = np.load('{}/test.npy'.format(dataFolder))
	train_label = train_data[:,0]
	trainData = train_data[:,1:]
	test_label = test_data[:,0]
	testData = test_data[:,1:]
	train_label = train_label.astype(int)
	test_label = test_label.astype(int)

	### model, parameter, optimizer 
	Nsample, dim = trainData.shape 
	Nsample_test, dim = testData.shape 
	Nclass = int( train_label.max() + 1 )
	print(trainData.dtype)
	param = model.NNParam(input_dim = dim, output_dim = Nclass, hidden_size = 50, prototype_num = 10, num_of_hidden_layer = 1)
	nnet = model.BaseFCNN(param)
	optimizer = optim.SGD(nnet.parameters(), lr=lr)

	iter_in_a_epoch = int(np.ceil(Nsample / batch_size))
	best_accu = 0.0
	for epoch in range(EPOCH):
		t1 = time()
		train_accu = test(nnet, trainData, train_label, batch_size, lossform)
		accu = test(nnet, testData, test_label, batch_size, lossform)
		best_accu = max(accu, best_accu)
		loss_in_a_epoch = 0
		for it in range(iter_in_a_epoch):
			loss = training(nnet, optimizer, trainData, train_label, it * batch_size, it * batch_size + batch_size, lossform)
			loss_in_a_epoch += loss.data[0]
		loss_in_a_epoch /= (Nsample / batch_size)
		#accu = test(nnet, testData, test_label, batch_size, lossform)
		print('{} Epoch, {} sec, loss:{}, train AUC: {}, AUC:{}, best AUC:{}'.format(epoch+1, \
			str(float(time()-t1))[:3], str(float(loss_in_a_epoch))[:6], str(train_accu)[:5], str(accu)[:5], str(best_accu)[:5]))
	return 

if __name__ == '__main__':
	total_learning()


