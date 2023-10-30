import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import os
import math
import numpy as np
import pandas as pd
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(__file__)))))))
from utils import mapping

from sklearn.model_selection import train_test_split


USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')
print("Train on [[[  {}  ]]] device.".format(device))


def normal_pdf(x):
	return torch.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)

def normal_cdf(y, h=0.01, tau=0.5):
	# Approximation of Q-function given by Lopez-Benitez & Casadevall (2011)
	# based on a second-order exponential function & Q(x) = 1 - Q(-x):
	Q_fn = lambda x: torch.exp(-0.4920 * x**2 - 0.2887*x - 1.1893)

	m = len(y)
	y_prime = (tau - y) / h
	summation = torch.sum(Q_fn(y_prime[y_prime > 0])) + torch.sum(1 - Q_fn(torch.abs(y_prime[y_prime < 0]))) + 0.5 * len(y_prime[y_prime == 0])

	return summation / m

def Huber_loss(x, delta):
	if abs(x) < delta:
		return (x**2) / 2
	else:
		return delta * (x.abs() - delta / 2)

def Huber_loss_derivative(x, delta):
	if x > delta:
		return delta
	elif x < -delta:
		return -delta
	return x

def get_fairness_metrics(Y, Z, Ytilde, classes, protect_attrs):
	DDP = 0
	DEO = 0
	for y in classes:
		Pr_Ytilde_y = (Ytilde == y).mean()
		Ytilde_y_given_Y_y = np.logical_and(Ytilde==y, Y==y)
		for z in range(n_sensitive_attrs):
			DDP += abs(np.logical_and(Ytilde==y, Z==z).mean() / (Z==z).mean() - Pr_Ytilde_y)
			DEO += abs(np.logical_and(Ytilde_y_given_Y_y==y, Z==z).mean() / np.logical_and(Y==y, Z==z).mean() - Ytilde_y_given_Y_y.mean() / (Y==y).mean())
	return DDP, DEO


class BCELossAccuracy():
	def __init__(self):
		self.loss_function = nn.BCELoss()

	@staticmethod
	def accuracy(y_hat, labels):
		with torch.no_grad():
			y_tilde = (y_hat > 0.5).int()
			accuracy = (y_tilde == labels.int()).float().mean().item()
		return accuracy

	def __call__(self, y_hat, labels):
		loss = self.loss_function(y_hat, labels)
		accuracy = self.accuracy(y_hat, labels)
		return loss, accuracy


class CELossAccuracy():
	def __init__(self):
		self.loss_function = nn.CrossEntropyLoss()

	@staticmethod
	def accuracy(y_hat, labels):
		with torch.no_grad():
			y_tilde = y_hat.argmax(axis=1)
			accuracy = (y_tilde == labels).float().mean().item()
		return accuracy

	def __call__(self, y_hat, labels):
		loss = self.loss_function(y_hat, labels)
		accuracy = self.accuracy(y_hat, labels)
		return loss, accuracy


class FairnessLoss():
	def __init__(self, h, tau, delta, notion, n_classes, n_sensitive_attrs, sensitive_attr):
		self.h = h
		self.tau = tau
		self.delta = delta
		self.fairness_notion = notion
		self.n_classes = n_classes
		self.n_sensitive_attrs = n_sensitive_attrs
		self.sensitive_attr = sensitive_attr

		if self.n_classes > 2:
			self.tau = 0.5

		assert self.fairness_notion in ['DP', 'EO']

	def DDP_loss(self, y_hat, Z):
		m = y_hat.shape[0]
		backward_loss = 0
		logging_loss = 0

		if self.n_classes == 2:
			Pr_Ytilde1 = normal_cdf(y_hat.detach(), self.h, self.tau)
			for z in self.sensitive_attr:
				Pr_Ytilde1_Z = normal_cdf(y_hat.detach()[Z==z], self.h, self.tau)
				m_z = Z[Z==z].shape[0]

				Prob_diff_Z = Pr_Ytilde1_Z - Pr_Ytilde1
				
				_dummy = \
				torch.dot(
					normal_pdf((self.tau - y_hat.detach()[Z==z]) / self.h).view(-1),
					y_hat[Z==z].view(-1)
					) / (self.h * m_z) -\
				torch.dot(
					normal_pdf((self.tau - y_hat.detach()) / self.h).view(-1),
					y_hat.view(-1)
					) / (self.h * m)

				_dummy *= Huber_loss_derivative(Prob_diff_Z, self.delta)

				backward_loss += _dummy

				logging_loss += Huber_loss(Prob_diff_Z, self.delta)

		else:
			idx_set = list(range(self.n_classes)) if self.n_classes > 2 else [0]
			for y in idx_set:
				Pr_Ytilde1 = normal_cdf(y_hat[:, y].detach(), self.h, self.tau)
				for z in self.sensitive_attr:
					Pr_Ytilde1_Z = normal_cdf(y_hat[:,y].detach(), self.h, self.tau)
					m_z = Z[Z==z].shape[0]

					Prob_diff_Z = Pr_Ytilde1_Z - Pr_Ytilde1
					_dummy = Huber_loss_derivative(Prob_diff_Z, self.delta)
					_dummy *= \
					torch.dot(
						normal_pdf((self.tau - y_hat[:,y].detach()[Z==z]) / self.h).view(-1),
						y_hat[:,y][Z==z].view(-1)
						) / (self.h * m_z) -\
					torch.dot(
						normal_pdf((self.tau - y_hat[:,y].detach()) / self.h).view(-1),
						y_hat[:,y].view(-1)
						) / (self.h * m)

					backward_loss += _dummy
					logging_loss += Huber_loss(Prob_diff_Z, self.delta).item()

		return backward_loss, logging_loss

	def DEO_loss(self, y_hat, Y, Z):
		backward_loss = 0
		logging_loss = 0

		if self.n_classes == 2:
			for y in [0, 1]:
				Pr_Ytilde1_Y = normal_cdf(y_hat[Y==y].detach(), self.h, self.tau)
				m_y = (Y==y).sum().item()
				for z in self.sensitive_attr:
					Pr_Ytilde1_YZ = normal_cdf(y_hat[torch.logical_and(Y==y, Z==z)].detach(), self.h, self.tau)
					m_zy = torch.logical_and(Y==y, Z==z).sum().item()

					Prob_diff_Z = Pr_Ytilde1_YZ - Pr_Ytilde1_Y
					_dummy = Huber_loss_derivative(Prob_diff_Z, self.delta)
					_dummy *= \
					torch.dot(
						normal_pdf((self.tau - y_hat[torch.logical_and(Y==y, Z==z)].detach()) / self.h).view(-1),
						y_hat[torch.logical_and(Y==y, Z==z)].view(-1)
						) / (self.h * m_zy) -\
					torch.dot(
						normal_pdf((self.tau - y_hat[Y==y].detach()) / self.h).view(-1),
						y_hat[torch.logical_and(Y==y, Z==z)].view(-1)
						) / (self.h * m_y)

					backward_loss += _dummy
					logging_loss += Huber_loss(Prob_diff_Z, self.delta).item()
		else:
			for y in range(self.n_classes):
				Pr_Ytilde1_Y = normal_cdf(y_hat[:,y][Y==y].detach(), self.h, self.tau)
				m_y = (Y==y).sum().item()
				for z in self.sensitive_attr:
					Pr_Ytilde1_YZ = normal_cdf(y_hat[:,y][torch.logical_and(Y==y, Z==z)].detach(), self.h, self.tau)
					m_zy = torch.logical_and(Y==y, Z==z).sum().item()

					Prob_diff_Z = Pr_Ytilde1_YZ - Pr_Ytilde1_Y
					_dummy = Huber_loss_derivative(Prob_diff_Z, self.delta)
					_dummy *= \
					torch.dot(
						normal_pdf((self.tau - y_hat[:,y][torch.logical_and(Y==y, Z==z)].detach()) / self.h).view(-1),
						y_hat[:,y][torch.logical_and(Y==y, Z==z)].view(-1)
						) / (self.h * m_zy) -\
					torch.dot(
						normal_pdf((self.tau - y_hat[:,y][Y==y].detach()) / self.h).view(-1),
						y_hat[:,y][Y==y].view(-1)
						) / (self.h * m_y)

				backward_loss += _dummy
				logging_loss += Huber_loss(Prob_diff_Z, self.delta).item()

		return backward_loss, logging_loss

	def __call__(self, y_hat, Y, Z):
		if self.fairness_notion == 'DP':
			return self.DDP_loss(y_hat, Z)
		else:
			return self.DEO_loss(y_hat, Y, Z)


class KDEDataset(Dataset):
	def __init__(self, X, y, z):
		self.cls2val_t, self.val2cls_t, target = mapping(y)
		self.cls2val_b, self.val2cls_b, bias = mapping(z)

		self.X = torch.Tensor(X).to(device)
		self.y = torch.Tensor(y).type(torch.long).to(device)
		self.z = torch.Tensor(z).type(torch.long).to(device)

	def __len__(self):
		return self.y.size(0)

	def __getitem__(self, idx):
		return self.X[idx], self.y[idx], self.z[idx]


class KDEmodel:
	def __init__(self, rawdata, fairness_notion, batch_size, n_epoch, learning_rate, h=0.01, tau=0.5, delta=0.5, l=0.1, model=None, train_ratio=0.8, seed=777):
		self.batch_size = batch_size
		self.n_epoch = n_epoch
		self.l = l

		self.n_class = len(np.unique(rawdata.target))
		self.n_protect = len(np.unique(rawdata.bias))

		# Split data
		X_train, X_test, y_train, y_test, z_train, z_test = train_test_split(
			rawdata.feature, rawdata.target, rawdata.bias,
			train_size=train_ratio, random_state=seed)

		self.train_data = KDEDataset(X_train, y_train, z_train)
		self.test_data = KDEDataset(X_test, y_test, z_test)
		self.all_data = KDEDataset(rawdata.feature, rawdata.target, rawdata.bias)

		# model setting
		if model:
			self.model = model
		else:
			self.model = nn.Linear(rawdata.feature.shape[-1], self.n_class)
		self.model = self.model.to(device)

		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

		# loss functions
		self.ce_loss = CELossAccuracy()
		self.fairness_loss = FairnessLoss(h, tau, delta, fairness_notion, self.n_class, self.n_protect, np.unique(y_train))


	def train(self):
		# setting dataloader
		trainloader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, drop_last=True)

		# setting model
		self.model.train()

		print('Train model start.')
		for ep in range(self.n_epoch):
			for idx, (X, y, z) in enumerate(trainloader):
				self.optimizer.zero_grad()

				pred = self.model(X)
				tilde = torch.round(pred.detach().reshape(-1))

				p_loss, acc = self.ce_loss(pred.squeeze(), y)
				f_loss, f_loss_item = self.fairness_loss(pred, y, z)
				cost = (1-self.l)*p_loss + self.l*f_loss

				if (torch.isnan(cost)).any(): continue

				cost.backward()
				self.optimizer.step()

				if (idx+1) % 10 == 0 or (idx+1) == len(trainloader):
					print('Epoch [{}/{}], Batch [{}/{}], Cost: {:.4f}'.format(
						ep+1, self.n_epoch,
						idx+1, len(trainloader),
						cost.item()), end='\r')
		print('Train model done.')


	def evaluation(self, all_data=False):
		# setting dataloader
		if all_data:
			testloader = DataLoader(self.all_data, batch_size=self.batch_size, shuffle=False, drop_last=False)
		else:
			testloader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, drop_last=False)

		# model setting
		self.model.eval()

		print('Evaluation start.')
		prediction = []
		for X, y, z in testloader:
			pred = self.model(X)
			pred = pred.argmax(dim=1)
			prediction.append(pred)

		prediction = torch.cat(prediction)
		print('Evaluation done.')

		return prediction