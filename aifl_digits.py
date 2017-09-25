# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
import os, sys, time, argparse
sys.path.append('utils')
from data import *
from nets import *
from hyperboard import Agent
import numpy as np


class AIFL_Digits(object):
	def __init__(self, E, D, M, data_A, data_B, exp, cuda=True, port=5000):
		self.E = E
		self.D = D
		self.M = M
		self.data_A = data_A
		self.data_B = data_B
		self.exp = exp
		self.cuda = cuda
		self.port = port

		assert self.data_A.channel == self.data_B.channel
		assert self.data_A.size == self.data_B.size
		assert self.data_A.n_class == self.data_B.n_class
		self.channel = self.data_A.channel
		self.size = self.data_A.size

		self.registe_curves()

		if self.cuda:
			self.E.cuda()
			self.D.cuda()
			self.M.cuda()

	def registe_curves(self):
		self.agent = Agent(username='', password='', address='127.0.0.1', port=self.port)
		loss_D_exp = {self.exp: "D loss: D predicts samples' attributes"}
		loss_E_exp = {self.exp: 'E loss: E encodes samples'}
		loss_M_exp = {self.exp: 'M loss: M classifies samples'}
		acc_A_exp = {self.exp: 'Categorization accuracy on data A'}
		acc_B_exp = {self.exp: 'Categorization accuracy on data B'}
		pre_loss_E_exp = {self.exp: 'Pretrain E loss: E encodes samples'}
		pre_loss_M_exp = {self.exp: 'Pretrain M loss: M classifies samples'}
		pre_acc_A_exp = {self.exp: 'Pretrain categorization accuracy on data A'}
		pre_acc_B_exp = {self.exp: 'Pretrain categorization accuracy on data B'}
		lr_exp = {self.exp: 'Learning rate at training phase(log scale)'}
		pre_lr_exp = {self.exp: 'Learning rate at pretraining phase(log scale)'}
		self.d_loss = self.agent.register(loss_D_exp, 'D loss', overwrite=True)
		self.e_loss = self.agent.register(loss_E_exp, 'E loss', overwrite=True)
		self.m_loss = self.agent.register(loss_M_exp, 'M loss', overwrite=True)
		self.acc_A = self.agent.register(acc_A_exp, 'acc', overwrite=True)
		self.acc_B = self.agent.register(acc_B_exp, 'acc', overwrite=True)
		self.pre_e_loss = self.agent.register(pre_loss_E_exp, 'E loss', overwrite=True)
		self.pre_m_loss = self.agent.register(pre_loss_M_exp, 'M loss', overwrite=True)
		self.pre_acc_A = self.agent.register(pre_acc_A_exp, 'acc', overwrite=True)
		self.pre_acc_B = self.agent.register(pre_acc_B_exp, 'acc', overwrite=True)
		self.tlr = self.agent.register(lr_exp, 'lr', overwrite=True)
		self.plr = self.agent.register(pre_lr_exp, 'lr', overwrite=True)

	def train(self, ckpt_dir, test_A, test_B, init_lr_E=1e-3, init_lr_D=1e-3, init_lr_M=1e-3, \
				batch_size=64, training_epochs=50000):
		x = Variable(torch.FloatTensor(batch_size, self.channel, self.size, self.size))
		y = Variable(torch.LongTensor(batch_size))
		s = Variable(torch.FloatTensor(batch_size))

		att_pred_criterion = nn.BCELoss()
		cat_criterion = nn.CrossEntropyLoss()

		if self.cuda:
			x = x.cuda()
			y = y.cuda()
			s = s.cuda()
			att_pred_criterion = att_pred_criterion.cuda()
			cat_criterion = cat_criterion.cuda()

		optimizer_D = optim.Adam(self.D.parameters(), lr=init_lr_D, betas=(0.5, 0.999))
		optimizer_E = optim.Adam(self.E.parameters(), lr=init_lr_E, betas=(0.5, 0.999))
		optimizer_M = optim.Adam(self.M.parameters(), lr=init_lr_M, betas=(0.5, 0.999))

		# scheduler_D = lr_scheduler.StepLR(optimizer_D, step_size=1000, gamma=0.9)
		# scheduler_E = lr_scheduler.StepLR(optimizer_E, step_size=1000, gamma=0.9)
		# scheduler_M = lr_scheduler.StepLR(optimizer_M, step_size=1000, gamma=0.9)
		scheduler_D = lr_scheduler.ReduceLROnPlateau(optimizer_D, mode='max', min_lr=1e-7, patience=5, factor=0.65, verbose=True)
		scheduler_E = lr_scheduler.ReduceLROnPlateau(optimizer_E, mode='max', min_lr=1e-7, patience=5, factor=0.65, verbose=True)
		scheduler_M = lr_scheduler.ReduceLROnPlateau(optimizer_M, mode='max', min_lr=1e-7, patience=5, factor=0.65, verbose=True)

		for epoch in range(training_epochs):
			# scheduler_D.step()
			# scheduler_E.step()
			# scheduler_M.step()

			begin_time = time.time()

			# fetch data
			batch_x_A, batch_y_A = self.data_A(batch_size//2)
			batch_x_B, batch_y_B = self.data_B(batch_size - batch_x_A.shape[0])
			x.data.copy_(torch.from_numpy(np.concatenate([batch_x_A, batch_x_B])))
			y.data.copy_(torch.from_numpy(np.concatenate([batch_y_A, batch_y_B])))
			s.data.copy_(torch.from_numpy(np.array([0]*batch_x_A.shape[0] + [1]*batch_x_B.shape[0])))

			# update D
			self.D.zero_grad()
			h = self.E(x)
			pred_s = self.D(h.detach())
			D_loss = att_pred_criterion(pred_s, s)
			D_loss.backward()
			optimizer_D.step()

			# update E and M
			self.E.zero_grad()
			self.M.zero_grad()
			pred_s = self.D(h)
			pred_y = self.M(h)
			M_loss = cat_criterion(pred_y, y)
			E_loss = -att_pred_criterion(pred_s, s) + M_loss
			E_loss.backward()
			optimizer_E.step()
			optimizer_M.step()

			# registe data on curves
			self.agent.append(self.d_loss, epoch, float(D_loss.data[0]))
			self.agent.append(self.e_loss, epoch, float(E_loss.data[0]))
			self.agent.append(self.m_loss, epoch, float(M_loss.data[0]))

			elapsed_time = time.time() - begin_time
			print('Epoch[%06d], D_loss: %.4f, E_loss: %.4f, M_loss: %.4f, elapsed_time: %.4ssecs.' % \
					(epoch+1, D_loss.data[0], E_loss.data[0], M_loss.data[0], elapsed_time))

			if epoch % 500 == 0:
				acc = {'A': 0, 'B': 0}
				val_data = {'A': test_A, 'B': test_B}
				for domain in val_data:
					while val_data[domain].has_next():
						batch_x, batch_y = val_data[domain](batch_size)
						x.data.copy_(torch.from_numpy(batch_x))
						n = int(np.sum(batch_y != -1))
						acc[domain] += np.sum(np.argmax(self.M(self.E(x)).cpu().data.numpy(), 1)[:n] == batch_y[:n])
					acc[domain] /= float(val_data[domain].N)

					val_data[domain].reset()  # reset so that next time when evaluates, cursor would start from 0

				print('Epoch[%06d], acc_A: %.4f, acc_B: %.4f' % (epoch+1, acc['A'], acc['B']))
				
				self.agent.append(self.acc_A, epoch, acc['A'])
				self.agent.append(self.acc_B, epoch, acc['B'])

				scheduler_D.step((acc['A'] + acc['B']) / 2)
				scheduler_E.step((acc['A'] + acc['B']) / 2)
				scheduler_M.step((acc['A'] + acc['B']) / 2)

				self.agent.append(self.tlr, epoch, float(np.log(optimizer_E.param_groups[0]['lr'])))

			if epoch % 10000 == 9999 or epoch == training_epochs-1:
				torch.save(self.E.state_dict(), os.path.join(ckpt_dir, 'E_epoch-%s.pth' % str(epoch+1).zfill(6)))
				torch.save(self.M.state_dict(), os.path.join(ckpt_dir, 'M_epoch-%s.pth' % str(epoch+1).zfill(6)))
				torch.save(self.D.state_dict(), os.path.join(ckpt_dir, 'D_epoch-%s.pth' % str(epoch+1).zfill(6)))

	def pretrain(self, ckpt_dir, test_A, test_B, init_lr_E=1e-3, init_lr_M=1e-3, batch_size=64, pretrain_epochs=5000):
		x = Variable(torch.FloatTensor(batch_size, self.channel, self.size, self.size))
		y = Variable(torch.LongTensor(batch_size))

		cat_criterion = nn.CrossEntropyLoss()

		if self.cuda:
			x = x.cuda()
			y = y.cuda()
			cat_criterion = cat_criterion.cuda()

		optimizer_E = optim.Adam(self.E.parameters(), lr=init_lr_E, betas=(0.5, 0.999))
		optimizer_M = optim.Adam(self.M.parameters(), lr=init_lr_M, betas=(0.5, 0.999))

		# scheduler_E = lr_scheduler.StepLR(optimizer_E, step_size=1000, gamma=0.3)
		# scheduler_M = lr_scheduler.StepLR(optimizer_M, step_size=1000, gamma=0.3)
		scheduler_E = lr_scheduler.ReduceLROnPlateau(optimizer_E, mode='max', min_lr=1e-7, patience=5, factor=0.65, verbose=True)
		scheduler_M = lr_scheduler.ReduceLROnPlateau(optimizer_M, mode='max', min_lr=1e-7, patience=5, factor=0.65, verbose=True)

		for epoch in range(pretrain_epochs):
			# scheduler_E.step()
			# scheduler_M.step()

			begin_time = time.time()

			# fetch data
			batch_x_A, batch_y_A = self.data_A(batch_size//2)
			batch_x_B, batch_y_B = self.data_B(batch_size - batch_x_A.shape[0])
			x.data.copy_(torch.from_numpy(np.concatenate([batch_x_A, batch_x_B])))
			y.data.copy_(torch.from_numpy(np.concatenate([batch_y_A, batch_y_B])))

			# update E and M
			self.E.zero_grad()
			self.M.zero_grad()

			h = self.E(x)
			pred_y = self.M(h)
			M_loss = cat_criterion(pred_y, y)
			E_loss = M_loss
			E_loss.backward()
			optimizer_E.step()
			optimizer_M.step()

			# registe data on curves
			self.agent.append(self.pre_e_loss, epoch, float(E_loss.data[0]))
			self.agent.append(self.pre_m_loss, epoch, float(M_loss.data[0]))

			elapsed_time = time.time() - begin_time
			print('Pretrain epoch[%06d], E_loss(= M_loss): %.4f, elapsed_time: %.4ssecs.' % \
					(epoch+1, E_loss.data[0], elapsed_time))

			if epoch % 500 == 0:
				acc = {'A': 0, 'B': 0}
				val_data = {'A': test_A, 'B': test_B}
				for domain in val_data:
					while val_data[domain].has_next():
						batch_x, batch_y = val_data[domain](batch_size)
						x.data.copy_(torch.from_numpy(batch_x))
						n = int(np.sum(batch_y != -1))
						acc[domain] += np.sum(np.argmax(self.M(self.E(x)).cpu().data.numpy(), 1)[:n] == batch_y[:n])
					acc[domain] /= float(val_data[domain].N)

					val_data[domain].reset()  # reset so that next time when evaluates, cursor would start from 0

				print('Pretrain epoch[%06d], acc_A: %.4f, acc_B: %.4f' % (epoch+1, acc['A'], acc['B']))
				
				self.agent.append(self.pre_acc_A, epoch, acc['A'])
				self.agent.append(self.pre_acc_B, epoch, acc['B'])

				scheduler_E.step((acc['A'] + acc['B']) / 2)
				scheduler_M.step((acc['A'] + acc['B']) / 2)

				self.agent.append(self.plr, epoch, float(np.log(optimizer_E.param_groups[0]['lr'])))

			if epoch % 10000 == 9999 or epoch == pretrain_epochs-1:
				torch.save(self.E.state_dict(), os.path.join(ckpt_dir, 'pretrain_E_epoch-%s.pth' % str(epoch+1).zfill(6)))
				torch.save(self.M.state_dict(), os.path.join(ckpt_dir, 'pretrain_M_epoch-%s.pth' % str(epoch+1).zfill(6)))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-g', '--gpu', type=str, default='', help='gpu(s) to use(default="")')
	parser.add_argument('-hd', '--h_dim', type=int, default=50, help='h_dim: dim of latent codes(default: 50)')
	parser.add_argument('-dh', '--D_hidden_size', type=int, default=20, \
						help='D_hidden_size: num of hidden nodes of D(default: 20)')
	parser.add_argument('-p', '--pretrain', type=int, default=5000, \
						help='pretrain: number of epochs for pretrain(default: 5000)')
	parser.add_argument('-b', '--batch_size', type=int, default=64, help='batch_size(default: 64)')
	parser.add_argument('-t', '--training_epochs', type=int, default=50000, \
						help='training_epochs: num of epochs for training IAFL. If 0, only preform pretraining,' + \
							' that is, supervised training.')

	args = parser.parse_args()

	img_size = 64

	gpu = args.gpu
	h_dim = args.h_dim
	D_hidden_size = args.D_hidden_size
	batch_size = args.batch_size
	pretrain = args.pretrain
	training_epochs = args.training_epochs

	os.environ['CUDA_VISIBLE_DEVICES'] = gpu

	cuda = len(os.environ['CUDA_VISIBLE_DEVICES']) > 0
	exp = 'iafl: mnist+svhn(h=%d, dh=%d, pretrain=%d, train=%d, ReduceLROnPlateau)' % (h_dim, D_hidden_size, pretrain, training_epochs)

	ckpt_dir = 'Models/mnist+svhn.%d.%d.pretrain_%d.train_%d.ReduceLROnPlateau' % (h_dim, D_hidden_size, pretrain, training_epochs)

	if not os.path.exists(ckpt_dir):
		os.makedirs(ckpt_dir)

	data_A = mnist(img_size=img_size, one_hot=False, channel=3)
	data_B = svhn(img_size=img_size, one_hot=False)

	test_A = mnist_test(img_size=img_size, one_hot=False, channel=3)
	test_B = svhn_test(img_size=img_size, one_hot=False)

	E = D_conv_classification(n_class=h_dim, channel=3, img_size=img_size)
	D = D_MLP(input_size=h_dim, hidden_size=D_hidden_size, output_size=1)
	M = D_MLP(input_size=h_dim, hidden_size=D_hidden_size, output_size=data_A.n_class)

	print('E:\n', E)
	print('D:\n', D)
	print('M:\n', M)

	aifl = AIFL_Digits(E, D, M, data_A, data_B, exp, cuda=cuda, port=5000)

	if pretrain:
		aifl.pretrain(ckpt_dir, test_A, test_B, batch_size=batch_size, pretrain_epochs=pretrain)
		init_lr_E = init_lr_M = init_lr_D = 1e-3
	else:
		init_lr_E = init_lr_M = init_lr_D = 1e-3

	aifl.train(ckpt_dir, test_A, test_B, init_lr_E=init_lr_E, init_lr_D=init_lr_D, init_lr_M=init_lr_M, \
				batch_size=batch_size, training_epochs=training_epochs)

