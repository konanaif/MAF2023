import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(__file__)))))))
from utils import mapping

from sklearn.model_selection import train_test_split


USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')
print("Trained on [[[  {}  ]]] device.".format(device))


class FFDdataset(Dataset):
	def __init__(self, X, y, z, image_shape):
		self.X = torch.Tensor(X).view(-1, image_shape[0], image_shape[1], image_shape[2]).to(device)
		self.y = torch.Tensor(y).type(torch.long).to(device)
		self.z = torch.Tensor(z).type(torch.long).to(device)
		#self.attr = torch.Tensor(attr).to(device)

	def __len__(self):
		return self.y.size(0)

	def __getitem__(self, index):
		return self.X[index], self.y[index], self.z[index]


def channel_shuffle(x, n_groups):
	# type: (torch.Tensor, int) -> torch.Tensor
	batch_size, num_channels, height, width = x.data.size()
	channels_per_group = num_channels // n_groups

	# reshape
	x = x.view(batch_size, n_groups, channels_per_group, height, width)
	x = torch.transpose(x, 1, 2).contiguous()

	# flatten
	x = x.view(batch_size, -1, height, width)

	return x


class InvertedResidual(nn.Module):
	def __init__(self, input_size, output_size, stride):
		super(InvertedResidual, self).__init__()

		if not (1 <= stride <= 3):
			raise ValueError('illegal stride value')
		self.stride = stride

		branch_output_size = output_size // 2
		assert (self.stride != 1) or (input_size == branch_output_size << 1)

		self.branch1 = nn.Sequential(
			nn.Conv2d(input_size, input_size, 3, self.stride, padding=1, bias=False, groups=input_size),
			nn.BatchNorm2d(input_size),
			nn.Conv2d(input_size, branch_output_size, 1, 1, 0, bias=False),
			nn.BatchNorm2d(branch_output_size),
			nn.ReLU(inplace=True))

		self.branch2 = nn.Sequential(
			nn.Conv2d(input_size if self.stride > 1 else branch_output_size, branch_output_size, 1, 1, 0, bias=False),
			nn.BatchNorm2d(branch_output_size),
			nn.ReLU(inplace=True),
			nn.Conv2d(branch_output_size, branch_output_size, 3, self.stride, padding=1, bias=False, groups=branch_output_size),
			nn.BatchNorm2d(branch_output_size),
			nn.Conv2d(branch_output_size, branch_output_size, 1, 1, 0, bias=False),
			nn.BatchNorm2d(branch_output_size),
			nn.ReLU(inplace=True))

	def forward(self, x):
		if self.stride == 1:
			x1, x2 = x.chunk(2, dim=1)
			out = torch.cat((x1, self.branch2(x2)), dim=1)
		else:
			out1 = self.branch1(x)
			out2 = self.branch2(x)
			out = torch.cat((out1, out2), dim=1)
		
		out = channel_shuffle(out, 2)

		return out


class Shufflenet(nn.Module):
	def __init__(self, n_class):
		super(Shufflenet, self).__init__()

		self.conv1 = nn.Sequential(
			nn.Conv2d(3, 24, 3, 2, 1, bias=False),
			nn.BatchNorm2d(24),
			nn.ReLU(inplace=True))

		self.maxpool = nn.MaxPool2d(3, 2, 1)

		self.conv2 = nn.Sequential(
			InvertedResidual(24, 116, 2),
			InvertedResidual(116, 116, 1),
			InvertedResidual(116, 116, 1),
			InvertedResidual(116, 116, 1))

		self.conv3 = nn.Sequential(
			InvertedResidual(116, 232, 2),
			InvertedResidual(232, 232, 1),
			InvertedResidual(232, 232, 1),
			InvertedResidual(232, 232, 1),
			InvertedResidual(232, 232, 1),
			InvertedResidual(232, 232, 1),
			InvertedResidual(232, 232, 1),
			InvertedResidual(232, 232, 1))

		self.conv4 = nn.Sequential(
			InvertedResidual(232, 464, 2),
			InvertedResidual(464, 464, 1),
			InvertedResidual(464, 464, 1),
			InvertedResidual(464, 464, 1))

		self.conv5 = nn.Sequential(
			nn.Conv2d(464, 1024, 1, 1, 0, bias=False),
			nn.BatchNorm2d(1024),
			nn.ReLU(inplace=True))

		self.fc = nn.Linear(1024, n_class)

	def forward(self, x, get_inter=False):
		x = self.conv1(x)
		x = self.maxpool(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.conv4(x)
		h = self.conv5(x)
		h1 = h.mean([2, 3])  # global pool
		out = self.fc(h1)

		if get_inter:
			return h, out
		else:
			return out


class MMDLoss(nn.Module):
	def __init__(self, w_m, sigma, n_class, n_group, kernel):
		super(MMDLoss, self).__init__()

		self.w_m = w_m
		self.sigma = sigma
		self.n_group = n_group
		self.n_class = n_class
		self.kernel = kernel

	@staticmethod
	def pdist(e1, e2, eps=1e-12, kernel='rbf', sigma_base=1.0, sigma_avg=None):
		if len(e1) == 0 or len(e2) == 0:
			res = torch.zeros(1)
		else:
			if kernel == 'rbf':
				e1_square = e1.pow(2).sum(dim=1)
				e2_square = e2.pow(2).sum(dim=1)
				prod = e1 @ e2.t()
				res = (e1_square.unsqueeze(1) + e2_square.unsqueeze(0) - 2*prod).clamp(min=eps)
				res = res.clone()
				sigma_avg = res.mean().detach() if sigma_avg is None else sigma_avg
				res = torch.exp(-res / (2*sigma_base*sigma_avg))
			elif kernel == 'poly':
				res = torch.matmul(e1, e2.t()).pow(2)

		return res, sigma_avg

	def forward(self, feature_student, feature_teacher, bias_batch, target_batch, jointfeature=False):
		if self.kernel == 'poly':
			student = F.normalize(feature_student.view(feature_student.shape[0], -1), dim=1)
			teacher = F.normalize(feature_teacher.view(feature_teacher.shape[0], -1), dim=1).detach()
		else:
			student = feature_student.view(feature_student.shape[0], -1)
			teacher = feature_teacher.view(feature_teacher.shape[0], -1).detach()

		mmd_loss = 0

		if jointfeature:
			K_TS, sigma_avg = self.pdist(teacher, student, sigma_base=self.sigma, kernel=self.kernel)
			K_TT, _ = self.pdist(teacher, teacher, sigma_base=self.sigma, sigma_avg=sigma_avg, kernel=self.kernel)
			K_SS, _ = self.pdist(student, student, sigma_base=self.sigma, sigma_avg=sigma_avg, kernel=self.kernel)

			mmd_loss += K_TT.mean() + K_SS.mean() + K_TS.mean()

		else:
			with torch.no_grad():
				_, sigma_avg = self.pdist(teacher, student, sigma_base=self.sigma, kernel=self.kernel)

			for target in target_batch.unique():
				if len(teacher[target_batch==target]) == 0:
					continue
				for bias in bias_batch.unique():
					if len(teacher[target_batch==target]) == 0:
						continue
					K_TS, _ = self.pdist(
						teacher[target_batch == target], 
						student[(target_batch == target) * (bias_batch == bias)], 
						sigma_base=self.sigma, sigma_avg=sigma_avg, 
						kernel=self.kernel)
					K_SS, _ = self.pdist(
						student[(target_batch == target) * (bias_batch == bias)],
						student[(target_batch == target) * (bias_batch == bias)],
						sigma_base=self.sigma, sigma_avg=sigma_avg,
						kernel=self.kernel)
					K_TT, _ = self.pdist(
						teacher[target_batch == target],
						teacher[target_batch == target],
						sigma_base=self.sigma, sigma_avg=sigma_avg,
						kernel=self.kernel)

					mmd_loss += K_TT.mean() + K_SS.mean() - 2*K_TS.mean()

		loss = 0.5 * self.w_m * mmd_loss

		return loss


def compute_hinton_loss(outputs, teacher_outputs=None, teacher=None, teacher_inputs=None, kd_temp=3):
	if teacher_outputs is None:
		if (teacher_inputs is not None and teacher is not None):
			teacher_outputs = teacher(teacher_inputs)
		else:
			Exception('Nothing is given to compute hinton loss')

	soft_label = F.softmax(teacher_outputs / kd_temp, dim=1).detach()
	kd_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs / kd_temp, dim=1),
		soft_label) * (kd_temp * kd_temp)

	return kd_loss



class FFD:
	def __init__(self, rawdataset, n_epoch, batch_size, learning_rate, image_shape, lambh=4, lambf=1, sigma=1.0, kernel='rbf', jointfeature=False, seed=88):
		# Dataloader
		X_train, X_test, y_train, y_test, z_train, z_test = train_test_split(
			rawdataset.feature_only, rawdataset.target, rawdataset.bias,
			train_size=0.8, random_state=seed)

		self.train_dataset = FFDdataset(X_train, y_train, z_train, image_shape)
		self.test_dataset = FFDdataset(X_test, y_test, z_test, image_shape)

		self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
		self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

		self.n_epoch = n_epoch
		self.batch_size = batch_size
		self.learning_rate = learning_rate

		self.lambh = lambh
		self.lambf = lambf
		self.sigma = sigma
		self.kernel = kernel
		self.jointfeature = jointfeature

		self.seed = seed

		self.n_class = len(np.unique(rawdataset.target))
		self.n_protect = len(np.unique(rawdataset.bias))

		self.teacher = Shufflenet(self.n_class).to(device)
		self.student = Shufflenet(self.n_class).to(device)

		self.teacher_optimizer = torch.optim.Adam(self.teacher.parameters(), lr=learning_rate)
		self.student_optimizer = torch.optim.Adam(self.student.parameters(), lr=learning_rate)

		self.criterion = nn.CrossEntropyLoss()

	def train_teacher(self):
		print('Train teacher start')
		self.teacher.train()

		for ep in range(self.n_epoch):
			for idx, (X, y, z) in enumerate(self.train_loader):
				self.teacher_optimizer.zero_grad()

				pred = self.teacher(X)
				loss = self.criterion(pred, y)

				loss.backward()
				self.teacher_optimizer.step()

				# print log
				if idx % 10 == 0:
					print('Epoch [{}/{}], Batch [{}/{}], Loss {}'.format(ep+1, self.n_epoch, idx, len(self.train_loader), loss))
		print('Train teacher done.')

	def train_student(self):
		print('Train student start')

		# distillation
		distiller = MMDLoss(w_m=self.lambf, sigma=self.sigma, n_class=self.n_class, n_group=self.n_protect, kernel=self.kernel)

		self.teacher.eval()
		self.student.train()

		for ep in range(self.n_epoch):
			for idx, (X, y, z) in enumerate(self.train_loader):
				self.student_optimizer.zero_grad()

				teacher_input = X.to()
				teacher_output = self.teacher(teacher_input, get_inter=True)
				teacher_logit = teacher_output[-1]

				student_output = self.student(X, get_inter=True)
				student_logit = student_output[-1]

				kd_loss = compute_hinton_loss(student_logit, teacher_outputs=teacher_logit, kd_temp=3) if self.lambh != 0 else 0

				loss = self.criterion(student_logit, y)
				loss = loss + self.lambh * kd_loss

				feature_student = student_output[-2]
				feature_teacher = teacher_output[-2]
				mmd_loss = distiller.forward(feature_student, feature_teacher, bias_batch=z, target_batch=y, jointfeature=self.jointfeature)

				loss = loss + mmd_loss

				loss.backward()
				self.student_optimizer.step()

				# print log
				if idx % 10 == 0:
					print('Epoch [{}/{}], Batch [{}/{}], Loss {}'.format(ep+1, self.n_epoch, idx, len(self.train_loader), loss))

		print('Train student end.')

	def evaluation(self):
		self.student.eval()

		predictions = []
		for idx, (X, y, z) in enumerate(self.test_loader):
			pred = self.student(X)
			pred = pred.argmax(dim=1)
			predictions.append(pred)
		predictions = torch.cat(predictions)

		print('Evaluation finished.')
		return predictions