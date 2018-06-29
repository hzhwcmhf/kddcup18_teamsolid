import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Parameter
from torch.autograd import Variable
import numpy as np
import math
import pickle as pkl
import logging
import os

class LinearNetwork(nn.Module):
	def __init__(self, model_num, output_num):
		super(LinearNetwork, self).__init__()
		self.weight = Parameter(torch.Tensor(model_num, output_num))
		self.factor = Parameter(torch.Tensor(output_num))
		stdv = 1.0 / math.sqrt(model_num)
		self.weight.data.uniform_(-stdv, stdv)
		self.factor.data.zero_()
		self.factor.data.add_(1)

	def forward(self, X):
		#batch_size * model_num * output_num
		return self.factor * torch.sum(X * F.softmax(self.weight.t()).t(), 1)

	def loss(self, X, Y):
		Y_pred = self.forward(X)
		return torch.mean(torch.abs(Y - Y_pred) * 2 / (Y + Y_pred))

class LinearEnsemble:

	def __init__(self, cachename, model_num, output_num):
		self.net = None
		self.cachename = cachename
		self.net = LinearNetwork(model_num, output_num)

	def save(self):
		torch.save({"weights": self.net.state_dict()}, self.cachename)
		logging.info("%s saved", self.cachename)

	def restore(self):
		if not os.path.exists(self.cachename):
			return False
		checkpoint = torch.load(self.cachename, map_location=lambda storage, loc: storage)
		self.net.load_state_dict(checkpoint["weights"])
		logging.info("%s loaded", self.cachename)
		return True
		
	def fit(self, X, Y, lr = 1e-3):
		self.net.cuda()
		optimizer = optim.Adam(self.net.parameters(), lr=lr)

		X = Variable(torch.Tensor(np.stack(X, 1))).cuda()
		Y = Variable(torch.Tensor(Y)).cuda()
		bestloss = 1e10
		last_update = 0
		iter = 0
		args = None
		while True:
			optimizer.zero_grad()
			loss = self.net.loss(X, Y)
			lossval = loss.data.cpu().numpy()[0]
			if lossval < bestloss:
				bestloss = lossval
				lastupdate = iter
				args = (self.net.weight.data.cpu().numpy(), self.net.factor.data.cpu().numpy())
			if iter - lastupdate > 100:
				break
			loss.backward()
			optimizer.step()
			logging.info("SGD iter:%d loss:%f", iter, lossval)
			iter += 1

		self.net.weight.data, self.net.factor.data = torch.Tensor(args[0]), torch.Tensor(args[1])
		self.net.cuda()
		loss = self.net.loss(X, Y)
		logging.info("optimized iter: %d, bestloss: %f", iter, loss.data.cpu().numpy()[0])
		self.net.cpu()

	def predict(self, X):
		Y = Variable(torch.Tensor(np.stack(X, 1)))
		return self.net.forward(Y).data.numpy()