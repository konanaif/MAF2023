import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torchvision import transforms

import os
import numpy as np
import random

from sklearn.model_selection import train_test_split


USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')
print("Train on [[[  {}  ]]] device.".format(device))


############## Dataset ###############
class fVAEDataset(torch.utils.data.Dataset):
	def __init__(self, feature, target, bias, channel, height, width):
		# normalize feature
		feature = feature / np.linalg.norm(feature)

		self.feature = torch.Tensor(feature)
		self.feature = self.feature.to(device)
		self.target = torch.Tensor(target).type(torch.long)
		self.target = self.target.to(device)
		self.bias = torch.Tensor(bias).type(torch.long)
		self.bias = self.bias.to(device)

		self.channel = channel
		self.height = height
		self.width = width

	def __len__(self):
		return self.target.size(0)

	def __getitem__(self, index):
		# select random index image
		index_ = random.choice(range(self.target.size(0)))

		# Convert Tensor size (length) ------> (channel, height, width)
		images1 = self.feature[index].view(self.channel, self.height, self.width)
		images2 = self.feature[index_].view(self.channel, self.height, self.width)

		return images1, images2, self.target[index], self.bias[index]


############ Loss functions ##############
def recon_loss(x, x_recon):
	n = x.size(0)
	loss = F.binary_cross_entropy_with_logits(x_recon, x, reduction='sum').div(n)
	return loss

def kl_divergence(mu, logvar):
	kld = -0.5 * (1 + logvar - mu**2 - logvar.exp()).sum(1).mean()
	return kld

def reparameterize(mu, logvar):
	std = logvar.mul(0.5).exp_()
	eps = std.data.new(std.size()).normal_()

	return eps.mul(std).add_(mu)

class GradReverse(torch.autograd.Function):
	@staticmethod
	def forward(ctx, x):
		return x.view_as(x)

	@staticmethod
	def backward(ctx, grad_output):
		return grad_output.neg() * 10

def grad_reverse(x):
	return GradReverse.apply(x)


############### Training functions ################
def mapping(class_vector):
	# Flatten
	class_vector = class_vector.ravel()

	cls2val = np.unique(class_vector)
	val2cls = dict(zip(cls2val, range(len(cls2val))))

	converted_vector = [val2cls[int(v)] for v in class_vector]

	return cls2val, val2cls, converted_vector

def permute_dims(z, dims):
	_, tar_dim, c_dim, pro_dim = dims
	assert z.dim() == 2

	B, _ = z.size()
	perm_z_list = []
	z_j = z.split(1, 1)
	perm = torch.randperm(B).to(device)
	perm_c = torch.randperm(B).to(device)

	for j in range(int(z.size(1))):
		if j < int(z.size(1)-pro_dim-c_dim):
			perm_z_j = z_j[j]
		elif j < int(z.size(1)-pro_dim):
			perm_z_j = z_j[j][perm_c]
		else:
			perm_z_j = z_j[j][perm]
		perm_z_list.append(perm_z_j)
	perm_z = torch.cat(perm_z_list, 1)

	return perm_z


############## VAE model ################
class VAE(nn.Module):
	def __init__(self, z_dim, channel, kernel, stride, padding, init_mode='normal'):
		super(VAE, self).__init__()

		self.z_dim = z_dim

		self.encoder = nn.Sequential(
			nn.Conv2d(channel, 32, kernel_size=kernel, stride=stride, padding=padding),
			nn.ReLU(True),
			nn.Conv2d(32, 32, kernel, stride, padding),
			nn.ReLU(True),
			nn.Conv2d(32, 64, kernel, stride, padding),
			nn.ReLU(True),
			nn.Conv2d(64, 64, kernel, stride, padding),
			nn.ReLU(True),
			nn.Conv2d(64, 128, kernel, 1),
			nn.ReLU(True),
			nn.Conv2d(128, 2*z_dim, 1))
		self.encoder = self.encoder.to(device)

		self.decoder = self.decoder = nn.Sequential(
			nn.Conv2d(z_dim, 128, 1),
			nn.ReLU(True),
			nn.ConvTranspose2d(128, 64, kernel),
			nn.ReLU(True),
			nn.ConvTranspose2d(64, 64, kernel, stride, padding),
			nn.ReLU(True),
			nn.ConvTranspose2d(64, 32, kernel, stride, padding),
			nn.ReLU(True),
			nn.ConvTranspose2d(32, 32, kernel, stride, padding),
			nn.ReLU(True),
			nn.ConvTranspose2d(32, channel, kernel, stride, padding))
		self.decoder = self.decoder.to(device)

		#### Weight initialize
		for block in self._modules:
			for m in self._modules[block]:
				if isinstance(m, (nn.Linear, nn.Conv2d)):
					if init_mode == 'kaiming':
						init.kaiming_normal_(m.weight)
					else:
						init.normal_(m.weight, 0, 0.02)
					if m.bias is not None:
						m.bias.data.fill_(0)
				elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
					m.weight.data.fill_(1)
					if m.bias is not None:
						m.bias.data.fill_(0)

	def forward(self, x, no_decoder=False):
		stats = self.encoder(x)
		mu = stats[:, :self.z_dim]
		logvar = stats[:, self.z_dim:]
		z = reparameterize(mu, logvar)

		if no_decoder:
			return z.squeeze()
		else:
			x_recon = self.decoder(z)
			return x_recon, mu, logvar, z.squeeze()


######################################################################################
class FairnessVAE:
	def __init__(self, dataset, z_dim, batch_size, num_epochs, image_shape=(3, 64, 64), learning_rate=1e-4, alpha=0.005, gamma=0.01, beta=0.05, grl=0.001, seed=777):
		##############################################
		# dataset : torch.utils.data.Dataset
		# dims = [z, t, c, p]
		##############################################

		### Dataset convert
		self.cls2val_t, self.val2cls_t, self.target = mapping(dataset.target)
		self.cls2val_b, self.val2cls_b, self.bias = mapping(dataset.bias)

		self.num_classes = len(self.cls2val_t)
		self.num_protected = len(self.cls2val_b)

		X_train, X_test, y_train, y_test, z_train, z_test = train_test_split(
			dataset.feature_only, self.target, self.bias,
			train_size=0.8, random_state=seed)

		### fVAE dataset
		channel, height, width = image_shape
		self.z_dim = z_dim
		self.tar_dim = z_dim // 2
		self.cls_dim = z_dim // 2
		self.pro_dim = z_dim // 2

		self.train_dataset = fVAEDataset(X_train, y_train, z_train, channel, height, width)
		self.test_dataset = fVAEDataset(X_test, y_test, z_test, channel, height, width)

		self.batch_size = batch_size
		self.num_epochs = num_epochs
		self.alpha = alpha
		self.gamma = gamma
		self.beta = beta
		self.grl = grl
		self.seed = seed


		#######################################
		########### Public models #############
		#######################################

		VAE_model = VAE(self.z_dim, channel, kernel=4, stride=2, padding=1)
		self.VAE_model = VAE_model.to(device)

		discriminator = nn.Sequential(
			nn.Linear(self.z_dim, 1000),
			nn.LeakyReLU(0.2, True),
			nn.Linear(1000, 1000),
			nn.LeakyReLU(0.2, True),
			nn.Linear(1000, 1000),
			nn.LeakyReLU(0.2, True),
			nn.Linear(1000, 1000),
			nn.LeakyReLU(0.2, True),
			nn.Linear(1000, 1000),
			nn.LeakyReLU(0.2, True),
			nn.Linear(1000, 2))
		self.discriminator = discriminator.to(device)


		#########################################
		########### Upstream models #############
		#########################################

		# 1. predict Target with Target Attribute Latent (tcls)
		target_TAL_classifier = nn.Sequential(
			nn.Linear(self.tar_dim, 1000),
			nn.LeakyReLU(0.2, True),
			nn.Linear(1000, 1000),
			nn.LeakyReLU(0.2, True),
			nn.Linear(1000, 1000),
			nn.LeakyReLU(0.2, True),
			nn.Linear(1000, 1000),
			nn.LeakyReLU(0.2, True),
			nn.Linear(1000, 1000),
			nn.LeakyReLU(0.2, True),
			nn.Linear(1000, self.num_classes))
		self.target_TAL_classifier = target_TAL_classifier.to(device)

		# 2. predict Bias with Target Attribute Latent (trevcls)
		protected_TAL_classifier = nn.Sequential(
			nn.Linear(self.tar_dim, 1000),
			nn.LeakyReLU(0.2, True),
			nn.Linear(1000, 1000),
			nn.LeakyReLU(0.2, True),
			nn.Linear(1000, 1000),
			nn.LeakyReLU(0.2, True),
			nn.Linear(1000, 1000),
			nn.LeakyReLU(0.2, True),
			nn.Linear(1000, 1000),
			nn.LeakyReLU(0.2, True),
			nn.Linear(1000, self.num_protected))
		self.protected_TAL_classifier = protected_TAL_classifier.to(device)

		# 3. predict Target with Protected Attribute Latent (pacls)
		target_PAL_classifier = nn.Sequential(
			nn.Linear(self.pro_dim, 1000),
			nn.LeakyReLU(0.2, True),
			nn.Linear(1000, 1000),
			nn.LeakyReLU(0.2, True),
			nn.Linear(1000, 1000),
			nn.LeakyReLU(0.2, True),
			nn.Linear(1000, 1000),
			nn.LeakyReLU(0.2, True),
			nn.Linear(1000, 1000),
			nn.LeakyReLU(0.2, True),
			nn.Linear(1000, self.num_classes))
		self.target_PAL_classifier = target_PAL_classifier.to(device)

		# 4. predict Bias with Protected Attribute Latent (revcls)
		protected_PAL_classifier = nn.Sequential(
			nn.Linear(self.pro_dim, 1000),
			nn.LeakyReLU(0.2, True),
			nn.Linear(1000, 1000),
			nn.LeakyReLU(0.2, True),
			nn.Linear(1000, 1000),
			nn.LeakyReLU(0.2, True),
			nn.Linear(1000, 1000),
			nn.LeakyReLU(0.2, True),
			nn.Linear(1000, 1000),
			nn.LeakyReLU(0.2, True),
			nn.Linear(1000, self.num_protected))
		self.protected_PAL_classifier = protected_PAL_classifier.to(device)

		## Optimizers
		self.VAE_optimizer = torch.optim.Adam(self.VAE_model.parameters(), lr=learning_rate)
		self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=learning_rate)

		self.target_TAL_optimizer = torch.optim.Adam(self.target_TAL_classifier.parameters(), lr=learning_rate)
		self.protected_TAL_optimizer = torch.optim.Adam(self.protected_TAL_classifier.parameters(), lr=learning_rate)
		self.target_PAL_optimizer = torch.optim.Adam(self.target_PAL_classifier.parameters(), lr=learning_rate)
		self.protected_PAL_optimizer = torch.optim.Adam(self.protected_PAL_classifier.parameters(), lr=learning_rate)


		###########################################
		########### Downstream models #############
		###########################################

		target_classifier = nn.Sequential(
			nn.Linear(self.tar_dim, 1000),
			nn.LeakyReLU(0.2, True),
			nn.Linear(1000, 1000),
			nn.LeakyReLU(0.2, True),
			nn.Linear(1000, 1000),
			nn.LeakyReLU(0.2, True),
			nn.Linear(1000, 1000),
			nn.LeakyReLU(0.2, True),
			nn.Linear(1000, 1000),
			nn.LeakyReLU(0.2, True),
			nn.Linear(1000, self.num_classes))
		self.target_classifier = target_classifier.to(device)

		PAL_target_classifier = nn.Sequential(
			nn.Linear(self.pro_dim, 1000),
			nn.LeakyReLU(0.2, True),
			nn.Linear(1000, 1000),
			nn.LeakyReLU(0.2, True),
			nn.Linear(1000, 1000),
			nn.LeakyReLU(0.2, True),
			nn.Linear(1000, 1000),
			nn.LeakyReLU(0.2, True),
			nn.Linear(1000, 1000),
			nn.LeakyReLU(0.2, True),
			nn.Linear(1000, self.num_classes))
		self.PAL_target_classifier = PAL_target_classifier.to(device)

		TAL_protected_classifier = nn.Sequential(
			nn.Linear(self.tar_dim, 1000),
			nn.LeakyReLU(0.2, True),
			nn.Linear(1000, 1000),
			nn.LeakyReLU(0.2, True),
			nn.Linear(1000, 1000),
			nn.LeakyReLU(0.2, True),
			nn.Linear(1000, 1000),
			nn.LeakyReLU(0.2, True),
			nn.Linear(1000, 1000),
			nn.LeakyReLU(0.2, True),
			nn.Linear(1000, self.num_protected))
		self.TAL_protected_classifier = TAL_protected_classifier.to(device)

		PAL_protected_classifier = nn.Sequential(
			nn.Linear(self.pro_dim, 1000),
			nn.LeakyReLU(0.2, True),
			nn.Linear(1000, 1000),
			nn.LeakyReLU(0.2, True),
			nn.Linear(1000, 1000),
			nn.LeakyReLU(0.2, True),
			nn.Linear(1000, 1000),
			nn.LeakyReLU(0.2, True),
			nn.Linear(1000, 1000),
			nn.LeakyReLU(0.2, True),
			nn.Linear(1000, self.num_protected))
		self.PAL_protected_classifier = PAL_protected_classifier.to(device)

		CAL_target_classifier = nn.Sequential(
			nn.Linear(self.cls_dim, 1000),
			nn.LeakyReLU(0.2, True),
			nn.Linear(1000, 1000),
			nn.LeakyReLU(0.2, True),
			nn.Linear(1000, 1000),
			nn.LeakyReLU(0.2, True),
			nn.Linear(1000, 1000),
			nn.LeakyReLU(0.2, True),
			nn.Linear(1000, 1000),
			nn.LeakyReLU(0.2, True),
			nn.Linear(1000, self.num_classes))
		self.CAL_target_classifier = CAL_target_classifier.to(device)

		CAL_protected_classifier = nn.Sequential(
			nn.Linear(self.cls_dim, 1000),
			nn.LeakyReLU(0.2, True),
			nn.Linear(1000, 1000),
			nn.LeakyReLU(0.2, True),
			nn.Linear(1000, 1000),
			nn.LeakyReLU(0.2, True),
			nn.Linear(1000, 1000),
			nn.LeakyReLU(0.2, True),
			nn.Linear(1000, 1000),
			nn.LeakyReLU(0.2, True),
			nn.Linear(1000, self.num_protected))
		self.CAL_protected_classifier = CAL_protected_classifier.to(device)

		self.fully_connected = nn.Linear(self.cls_dim, self.cls_dim)

		## Optimizer
		self.tarcls_optimizer = torch.optim.Adam(self.target_classifier.parameters(), lr=learning_rate)
		self.PAL_target_optimizer = torch.optim.Adam(self.PAL_target_classifier.parameters(), lr=learning_rate)
		self.TAL_protected_optimizer = torch.optim.Adam(self.TAL_protected_classifier.parameters(), lr=learning_rate)
		self.PAL_protected_optimizer = torch.optim.Adam(self.PAL_protected_classifier.parameters(), lr=learning_rate)
		self.CAL_target_optimizer = torch.optim.Adam(self.CAL_target_classifier.parameters(), lr=learning_rate)
		self.CAL_protected_optimizer = torch.optim.Adam(self.CAL_protected_classifier.parameters(), lr=learning_rate)
		self.fc_optimizer = torch.optim.Adam(self.fully_connected.parameters(), lr=learning_rate)


	def train_upstream(self):
		### Traning start
		train_loader = torch.utils.data.DataLoader(
			self.train_dataset,
			batch_size=self.batch_size,
			shuffle=True, num_workers=0, drop_last=True)

		self.VAE_model.train()
		self.discriminator.train()
		self.target_TAL_classifier.train()
		self.protected_TAL_classifier.train()
		self.target_PAL_classifier.train()
		self.protected_PAL_classifier.train()

		print('### Training upstream start')
		for epoch in range(self.num_epochs):
			vae_loss_list = []
			discri_loss_list = []
			for batch_idx, (img, img_, label, protected) in enumerate(train_loader):
				# Value
				img = img.to(device)
				img_ = img_.to(device)
				label = label.to(device)
				protected = protected.to(device)

				# Initialize optimizer
				self.VAE_optimizer.zero_grad()
				self.discriminator.zero_grad()
				self.target_PAL_optimizer.zero_grad()
				self.protected_PAL_optimizer.zero_grad()
				self.target_TAL_optimizer.zero_grad()
				self.protected_TAL_optimizer.zero_grad()

				# Make features
				img_recon, mu, logvar, z = self.VAE_model(img)
				discri_z = self.discriminator(z)

				TAL = z.split(self.pro_dim, 1)[-1]
				PAL = z.split(self.pro_dim, 1)[0]

				PAL_output = self.target_PAL_classifier(PAL)
				z_reverse = grad_reverse(PAL)
				PAL_reverse_output = self.protected_PAL_classifier(z_reverse)

				TAL_output = self.target_TAL_classifier(TAL)
				z_t_reverse = grad_reverse(TAL)
				TAL_reverse_output = self.protected_TAL_classifier(z_t_reverse)

				# Calculate VAE loss
				vae_recon_loss = recon_loss(img, img_recon)
				vae_kld = kl_divergence(mu, logvar)
				vae_tc_loss = (discri_z[:, :1] - discri_z[:, 1:]).mean()

				target_PAL_loss = F.cross_entropy(PAL_output, label)
				protected_PAL_loss = F.cross_entropy(PAL_reverse_output, protected)
				target_TAL_loss = F.cross_entropy(TAL_output, label)
				protected_TAL_loss = F.cross_entropy(TAL_reverse_output, protected)

				vae_loss = vae_recon_loss + vae_kld + (self.alpha*self.gamma*vae_tc_loss) + (self.beta*protected_PAL_loss) + (self.beta*target_PAL_loss) + (self.grl*protected_TAL_loss) + (self.grl*target_TAL_loss)
				vae_loss_list.append(vae_loss)

				## Optimize the models
				vae_loss.backward(retain_graph=True)

				self.VAE_optimizer.step()
				self.target_PAL_optimizer.step()
				self.protected_PAL_optimizer.step()
				self.target_TAL_optimizer.step()
				self.protected_TAL_optimizer.step()

				# Calculate discriminator loss
				z_ = self.VAE_model.forward(img_, no_decoder=True)
				z_pperm = permute_dims(z_, [self.z_dim, self.tar_dim, self.cls_dim, self.pro_dim])

				discri_z_pperm = self.discriminator(z_pperm)

				ones = torch.ones(self.batch_size, dtype=torch.long).to(device)
				zeros = torch.zeros(self.batch_size, dtype=torch.long).to(device)

				discri_tc_loss = (F.cross_entropy(discri_z_pperm, zeros) + F.cross_entropy(discri_z_pperm, ones))
				discri_loss_list.append(discri_tc_loss)
				self.discriminator_optimizer.step()

			avg_vae_loss = sum(vae_loss_list) / len(vae_loss_list)
			avg_discri_loss = sum(discri_loss_list) / len(discri_loss_list)
			print('Epoch [{:03d}]   VAE loss: {:.3f}   Discriminator loss: {:.3f}'.format(epoch+1, avg_vae_loss, avg_discri_loss))
		print('### Upstream training done.')


	def train_downstream(self):
		## train mode
		self.target_classifier.train()
		self.PAL_target_classifier.train()
		self.TAL_protected_classifier.train()
		self.PAL_protected_classifier.train()
		self.CAL_target_classifier.train()
		self.fully_connected.train()
		self.CAL_protected_classifier.train()

		## Convert mode
		for name, param in self.VAE_model.named_parameters():
			param.requires_grad = False

		### Traning start
		train_loader = torch.utils.data.DataLoader(
			self.train_dataset,
			batch_size=self.batch_size,
			shuffle=True, num_workers=0, drop_last=True)

		print('### Downstream training start.')
		for epoch in range(self.num_epochs):
			vae_loss_list = []
			target_loss_list = []
			for batch_idx, (img, img_, label, protected) in enumerate(train_loader):
				self.PAL_target_optimizer.zero_grad()
				self.TAL_protected_optimizer.zero_grad()
				self.PAL_protected_optimizer.zero_grad()
				self.CAL_target_optimizer.zero_grad()
				self.fc_optimizer.zero_grad()
				self.CAL_protected_optimizer.zero_grad()
				self.tarcls_optimizer.zero_grad()

				img = img.to(device)
				img_ = img_.to(device)
				label = label.to(device)
				protected = protected.to(device)

				img_recon, mu, logvar, z = self.VAE_model(img)

				vae_recon_loss = recon_loss(img, img_recon)
				vae_kld = kl_divergence(mu, logvar)
				discri_z = self.discriminator(z)
				vae_tc_loss = (discri_z[:, :1] - discri_z[:, 1:]).mean()

				TAL = z.split(self.pro_dim, 1)[-1]
				TL = z.split(self.tar_dim, 1)[0]
				CL = z.split(self.cls_dim, 1)[1]

				pa_target = self.PAL_target_classifier(TAL)
				target_pa = self.TAL_protected_classifier(TL)
				pa_pa = self.PAL_protected_classifier(TAL)

				ca_target = self.CAL_target_classifier(CL)
				filtered = self.fully_connected(CL)
				ca_pa = self.CAL_protected_classifier(grad_reverse(filtered))

				target = self.target_classifier(TL + 0.05*filtered)

				## loss
				tarcls_loss = F.cross_entropy(target, label)
				pa_target_loss = F.cross_entropy(pa_target, label)
				target_pa_loss = F.cross_entropy(target_pa, protected)
				pa_pa_loss = F.cross_entropy(pa_pa, protected)

				ca_target_loss = F.cross_entropy(ca_target, label)
				ca_pa_loss = F.cross_entropy(ca_pa, protected)

				vae_loss = vae_recon_loss + vae_kld + self.gamma*vae_tc_loss
				target_loss = pa_target_loss + 10*tarcls_loss + target_pa_loss + pa_pa_loss + ca_pa_loss
				vae_loss_list.append(vae_loss)
				target_loss_list.append(target_loss)

				target_loss.backward()

				self.PAL_target_optimizer.step()
				self.TAL_protected_optimizer.step()
				self.PAL_protected_optimizer.step()
				self.CAL_target_optimizer.step()
				self.fc_optimizer.step()
				self.CAL_protected_optimizer.step()
				self.tarcls_optimizer.step()

			avg_vae_loss = sum(vae_loss_list) / len(vae_loss_list)
			avg_target_loss = sum(target_loss_list) / len(target_loss_list)
			print('Epoch [{:03d}]   VAE loss: {:.3f}   Discriminator loss: {:.3f}'.format(epoch+1, avg_vae_loss, avg_target_loss))
		print('### Downstream training done.')


	def evaluation(self):
		## evaluation mode
		self.target_classifier.eval()
		self.PAL_target_classifier.eval()
		self.TAL_protected_classifier.eval()
		self.PAL_protected_classifier.eval()
		self.CAL_target_classifier.eval()
		self.fully_connected.eval()
		self.CAL_protected_classifier.eval()

		ones = torch.ones(self.batch_size, dtype=torch.long, device=device)
		zeros = torch.zeros(self.batch_size, dtype=torch.long, device=device)

		test_dataloader = torch.utils.data.DataLoader(
			self.test_dataset,
			batch_size=self.batch_size,
			shuffle=True, num_workers=0, drop_last=True)

		prediction = []
		print('### Evaluation start.')
		for batch_idx, (img, img_, label, protected) in enumerate(test_dataloader):
			img = img.to(device)
			img_ = img_.to(device)
			label = label.to(device)
			protected = protected.to(device)

			img_recon, mu, logvar, z = self.VAE_model(img)

			TAL = z.split(self.pro_dim, 1)[-1]
			TL = z.split(self.tar_dim, 1)[0]
			CL = z.split(self.cls_dim, 1)[1]

			pa_target = self.PAL_target_classifier(TAL)
			target_pa = self.TAL_protected_classifier(TL)
			pa_pa = self.PAL_protected_classifier(TAL)

			ca_target = self.CAL_target_classifier(CL)
			filtered = self.fully_connected(CL)
			ca_pa = self.CAL_protected_classifier(grad_reverse(filtered))

			target = self.target_classifier(TL + 0.05*filtered)
			prediction.append(target.argmax(dim=1))

		prediction = torch.cat(prediction)
		print('### Evaluation done.')

		# Convert prediction value vector to class value vector
		#prediction.apply_(lambda x: self.val2cls_t[x])

		return prediction