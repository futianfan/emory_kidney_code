
from __future__ import print_function
import sys
import torch
from torch import nn 
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np 
from time import time
from sklearn.metrics import roc_auc_score
torch.manual_seed(2)    # reproducible
np.random.seed(3)


class NNParam:
	def __init__(self, input_dim, output_dim, hidden_size = 20, prototype_num = 10, num_of_hidden_layer = 1, batch_normalization = True):
		self.input_dim = input_dim
		self.output_dim = output_dim 
		self.hidden_size = hidden_size 
		self.prototype_num = prototype_num 
		self.num_of_hidden_layer = num_of_hidden_layer
		self.batch_normalization = batch_normalization

class BaseFCNN(torch.nn.Module):
	def __init__(self, param):
		super(BaseFCNN, self).__init__()
		self.input_dim = param.input_dim 
		self.output_dim = param.output_dim 
		self.hidden_size = param.hidden_size 
		self.num_of_hidden_layer = param.num_of_hidden_layer
		self.fc_in = nn.Linear(self.input_dim, self.hidden_size)
		self.hidden_layer = nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size) for \
			_ in range(self.num_of_hidden_layer)])
		self.fc_out = nn.Linear(self.hidden_size, self.output_dim)
		self.f = F.relu  ## sigmoid  relu 

		self.do_bn = param.batch_normalization 
		self.bn_input = nn.BatchNorm1d(1, momentum=0.5)
		self.single_layer = nn.Linear(self.input_dim, self.output_dim)

	def forward_hidden(self, x):
		for layer in range(self.num_of_hidden_layer):
			x = self.f(self.hidden_layer[layer](x))
		return x

	def forward_a(self, X_batch):
		X_batch = torch.from_numpy(X_batch).float()
		X_batch = Variable(X_batch)
		X_hid = self.f(self.fc_in(X_batch))
		X_hid = self.forward_hidden(X_hid)
		return X_hid

	def forward_logistic_regression(self, X_batch):
		X_batch = Variable(torch.from_numpy(X_batch).float())
		return self.single_layer(X_batch)

	def forward(self, X_batch):
		#return self.forward_logistic_regression(X_batch)
		X_hid = self.forward_a(X_batch) 
		X_out = self.fc_out(X_hid)
		return X_out

	def test(self, X, batch_size = 128):
		N, _ = X.shape 
		num_iter = int(np.ceil(N / batch_size))
		for i in range(num_iter):
			X_batch = X[i * batch_size: i * batch_size + batch_size]
			X_batch_out = self.forward(X_batch)
			X_out = torch.cat([X_out, X_batch_out], 0) if i > 0 else X_batch_out
		return X_out 

class PrototypeNN(BaseFCNN, torch.nn.Module):
	def __init__(self, param, assignment):
		super(PrototypeNN, self).__init__(param)
		self.prototype_num = param.prototype_num
		self.assignment = assignment
		self.fc_out = nn.Linear(self.prototype_num, self.output_dim)
		self.prototype = Variable(torch.randn(self.prototype_num, self.hidden_size), requires_grad = False)

	def generate_average_vector_in_a_cluster(self, X, lst, batch_size):
		leng = len(lst)
		X_in = X[lst]
		it_num = int(np.ceil(leng / batch_size))
		for it in range(it_num):
			X_hid = self.forward_a(X_in[it * batch_size: it * batch_size + batch_size])
			X_all = torch.cat([X_all, X_hid], 0) if it > 0 else X_hid
		return torch.mean(X_all, 0)

	def generate_prototype(self, X, assignment, batch_size):
		assert len(assignment) == self.prototype_num
		for i,assign in enumerate(assignment):
			#print('generate {} prototype'.format(i))
			self.prototype[i,:] = self.generate_average_vector_in_a_cluster(X, assignment[i], batch_size).data
		assert self.prototype.requires_grad == False 

	def forward_prototype(self, X_hid, assignment):
		N, d = X_hid.shape 
		p = self.prototype_num
		norm = X_hid.norm(p=2, dim=1, keepdim=True) 
		X_hid = X_hid.div(norm)  ### normalized 
		X_hid = X_hid.view(N,d,1)
		X_hid_ext = X_hid.expand(N,d,p)
		proto_vector = self.prototype.data 
		proto_vector = Variable(proto_vector, requires_grad = False)
		norm = proto_vector.norm(p=2, dim = 1, keepdim = True)
		proto_vector = proto_vector.div(norm)  ### normalized 
		proto_vector = proto_vector.view(1,d,p)
		proto_vector = proto_vector.expand(N,d,p)
		assert X_hid_ext.requires_grad == True and proto_vector.requires_grad == False 
		inner_product = X_hid_ext * proto_vector
		inner_product = torch.sum(inner_product, 1)
		inner_product = inner_product.view(N,p)
		return inner_product 		

	def forward(self, X_batch, assignment):
		X_hid = self.forward_a(X_batch)
		X_out = self.forward_prototype(X_hid, assignment)
		return self.fc_out(X_out)









