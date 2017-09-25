import os,sys,pickle
from PIL import Image
import scipy.misc
import scipy.io
from glob import glob
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from tensorflow.examples.tutorials.mnist import input_data
from torchvision import datasets, transforms

prefix = '~/dataset/'

def get_img(img_path, is_crop=True, crop_h=256, resize_h=64):
	img=scipy.misc.imread(img_path, mode='RGB').astype(np.float)
	resize_w = resize_h
	if is_crop:
		crop_w = crop_h
		h, w = img.shape[:2]
		j = int(round((h - crop_h)/2.))
		i = int(round((w - crop_w)/2.))
		cropped_image = scipy.misc.imresize(img[j:j+crop_h, i:i+crop_w],[resize_h, resize_w])
	else:
		cropped_image = scipy.misc.imresize(img,[resize_h, resize_w])
	return np.transpose(np.array(cropped_image)/255.0, [2, 0, 1])


class celebA():
	def __init__(self, img_size=64):
		datapath = prefix + 'CelebA/images'
		self.z_dim = 100
		self.size = img_size
		self.channel = 3
		self.data = glob(os.path.join(datapath, '*.jpg'))

		self.batch_count = 0

	def __call__(self,batch_size):
		batch_number = len(self.data)/batch_size
		if self.batch_count < batch_number-2:
			self.batch_count += 1
		else:
			self.batch_count = 0

		path_list = self.data[self.batch_count*batch_size:(self.batch_count+1)*batch_size]

		batch = [get_img(img_path, True, 128, self.size) for img_path in path_list]
		batch_imgs = np.array(batch).astype(np.float32)
		
		return batch_imgs

	def data2fig(self, samples):
		N_samples = samples.shape[0]
		N_row = N_col = int(np.ceil(N_samples**0.5))
		fig = plt.figure(figsize=(N_row, N_col))
		gs = gridspec.GridSpec(N_row, N_col)
		gs.update(wspace=0.05, hspace=0.05)
		samples = np.transpose(samples, [0, 2, 3, 1])

		for i, sample in enumerate(samples):
			ax = plt.subplot(gs[i])
			plt.axis('off')
			ax.set_xticklabels([])
			ax.set_yticklabels([])
			ax.set_aspect('equal')
			plt.imshow(sample)
		return fig

class cifar():
	def __init__(self, one_hot=False, img_size=64):
		datapath = prefix + 'cifar10'
		self.z_dim = 100
		self.size = img_size
		self.channel = 3
		self.data = glob(os.path.join(datapath, '*'))
		self.n_class = 10
		self.one_hot = one_hot

		self.batch_count = 0

	def __call__(self,batch_size):
		batch_number = len(self.data)/batch_size
		if self.batch_count < batch_number-2:
			self.batch_count += 1
		else:
			self.batch_count = 0

		path_list = self.data[self.batch_count*batch_size:(self.batch_count+1)*batch_size]
		batch_labels = np.array([int(os.path.basename(pl).split('_')[0].strip('label')) for pl in path_list])
		if self.one_hot:
			tmp = np.zeros((batch_size, self.n_class))
			tmp[np.arange(batch_size), labels] = 1.
			batch_labels = tmp

		batch = [get_img(img_path, False, 128, self.size) for img_path in path_list]
		batch_imgs = np.array(batch).astype(np.float32)
	
		return batch_imgs, batch_labels

	def data2fig(self, samples):
		N_samples = samples.shape[0]
		N_row = N_col = int(np.ceil(N_samples**0.5))
		fig = plt.figure(figsize=(N_row, N_col))
		gs = gridspec.GridSpec(N_row, N_col)
		gs.update(wspace=0.05, hspace=0.05)
		samples = np.transpose(samples, [0, 2, 3, 1])

		for i, sample in enumerate(samples):
			ax = plt.subplot(gs[i])
			plt.axis('off')
			ax.set_xticklabels([])
			ax.set_yticklabels([])
			ax.set_aspect('equal')
			plt.imshow(sample)
		return fig


class mnist():
	def __init__(self, one_hot=False, img_size=64, channel=1):
		datapath = prefix + 'mnist'
		self.one_hot = one_hot
		self.z_dim = 100
		self.size = img_size
		self.channel = channel
		self.data = input_data.read_data_sets(datapath, one_hot=one_hot)
		# if os.path.exists(os.path.join(datapath, 'processed/training.pt')):
		# 	download = False
		# else:
		# 	download = True
		# self.data = datasets.MNIST(datapath, train=True, download=download, \
		# 							transform=transforms.Compose([
		# 								transforms.Scale([self.size, self.size]), 
		# 								transforms.ToTensor()])
		# 							)
		self.n_class = 10
		self.N = 50000
		print('MNIST train', (50000, self.channel, self.size, self.size), (50000,))

	def __call__(self, batch_size):
		# idx = np.random.randint(0, len(self.data), batch_size)
		# batch_imgs = np.vstack([self.data[i][0].data.numpy() for i in idx])
		# y = np.array([self.data[i][1] for i in idx])
		# if self.one_hot:
		# 	y_onehot = np.zeros((batch_size, self.n_class))
		# 	y_onehot[np.arange(batch_size), y] = 1.
		# 	return batch_imgs, y_onehot
		# return batch_imgs, y
		batch_imgs = np.zeros([batch_size, self.size, self.size, 1])

		batch_x,y = self.data.train.next_batch(batch_size)
		batch_x = np.reshape(batch_x, (batch_size, 28, 28, 1))
		for i in range(batch_size):
			img = batch_x[i,:,:,0]
			batch_imgs[i,:,:,0] = scipy.misc.imresize(img, [self.size, self.size])
		batch_imgs /= 255.
		batch_imgs = np.transpose(batch_imgs, [0, 3, 1, 2])
		batch_imgs = np.tile(batch_imgs, [1, self.channel, 1, 1])
		return batch_imgs, y
		

	def data2fig(self, samples):
		N_samples = samples.shape[0]
		N_row = N_col = int(np.ceil(N_samples**0.5))
		fig = plt.figure(figsize=(N_row, N_col))
		gs = gridspec.GridSpec(N_row, N_col)
		gs.update(wspace=0.05, hspace=0.05)
		samples = np.transpose(samples, [0, 2, 3, 1])

		for i, sample in enumerate(samples):
			ax = plt.subplot(gs[i])
			plt.axis('off')
			ax.set_xticklabels([])
			ax.set_yticklabels([])
			ax.set_aspect('equal')
			plt.imshow(sample.reshape(self.size,self.size), cmap='Greys_r')
		return fig


class mnist_test():
	def __init__(self, one_hot=False, img_size=64, channel=1):
		datapath = prefix + 'mnist'
		self.one_hot = one_hot
		self.z_dim = 100
		self.size = img_size
		self.channel = channel
		self.data = np.fromfile(os.path.join(datapath, 't10k-images.idx3-ubyte'), dtype=np.uint8)
		self.data = self.data[16:].reshape((10000, 28, 28, 1)).astype(np.float32)
		tmp = np.zeros((10000, 1, self.size, self.size))
		for i in range(10000):
			tmp[i,0,:,:] = scipy.misc.imresize(self.data[i,:,:,0], [self.size, self.size])
		self.data = np.tile(tmp/255., [1, self.channel, 1, 1])
		self.label = np.fromfile(os.path.join(datapath, 't10k-labels.idx1-ubyte'), dtype=np.uint8)
		self.label = self.label[8:].astype(np.int32)
		self.n_class = 10
		self.cursor = 0
		self.N = self.data.shape[0]
		print('MNIST test', self.data.shape, self.label.shape)

	def has_next(self):
		return self.cursor < 10000

	def reset(self):
		self.cursor = 0

	def __call__(self, batch_size):
		if self.cursor + batch_size <= 10000:
			batch_x = self.data[self.cursor:self.cursor+batch_size]
			batch_y = self.label[self.cursor:self.cursor+batch_size]
			if self.one_hot:
				batch_y = np.eye(self.n_class)[batch_y]
		else:
			pad_n = batch_size - (10000 - self.cursor)
			batch_x = np.concatenate([self.data[self.cursor:], np.zeros([pad_n, self.channel, self.size, self.size])])
			if self.one_hot:
				batch_y = np.concatenate([np.eye(self.n_class)[batch_y], np.zeros((pad_n, self.n_class))])
			else:
				batch_y = np.concatenate([self.label[self.cursor:], [-1] * pad_n])

		self.cursor += batch_size

		return batch_x, batch_y
		
	def data2fig(self, samples):
		N_samples = samples.shape[0]
		N_row = N_col = int(np.ceil(N_samples**0.5))
		fig = plt.figure(figsize=(N_row, N_col))
		gs = gridspec.GridSpec(N_row, N_col)
		gs.update(wspace=0.05, hspace=0.05)
		samples = np.transpose(samples, [0, 2, 3, 1])

		for i, sample in enumerate(samples):
			ax = plt.subplot(gs[i])
			plt.axis('off')
			ax.set_xticklabels([])
			ax.set_yticklabels([])
			ax.set_aspect('equal')
			plt.imshow(sample.reshape(self.size,self.size), cmap='Greys_r')
		return fig


class svhn():
	def __init__(self, img_size=64, one_hot=False):
		datapath = prefix + 'svhn'
		self.size = img_size
		self.one_hot = one_hot
		data = scipy.io.loadmat(os.path.join(datapath, 'train_32x32.mat'))
		self.data = data['X']
		self.label = data['y'].flatten()
		self.label[self.label==10] = 0
		self.channel = 3
		tmp = np.zeros((self.data.shape[3], self.size, self.size, self.channel))
		for i in range(self.data.shape[3]):
			tmp[i,:,:,:] = scipy.misc.imresize(self.data[:,:,:,i], [self.size, self.size])
		self.data = np.transpose(tmp/255., [0, 3, 1, 2])
		self.n_class = 10
		self.N = self.data.shape[0]
		print('SVHN train', self.data.shape, self.label.shape)

	def __call__(self, batch_size):
		idx = np.random.choice(self.data.shape[0], size=batch_size, replace=False)
		batch_x = self.data[idx]
		batch_y = self.label[idx]
		if self.one_hot:
			batch_y = np.eye(self.n_class)[batch_y]
		return batch_x, batch_y

	def data2fig(self, samples):
		N_samples = samples.shape[0]
		N_row = N_col = int(np.ceil(N_samples**0.5))
		fig = plt.figure(figsize=(N_row, N_col))
		gs = gridspec.GridSpec(N_row, N_col)
		gs.update(wspace=0.05, hspace=0.05)
		samples = np.transpose(samples, [0, 2, 3, 1])

		for i, sample in enumerate(samples):
			ax = plt.subplot(gs[i])
			plt.axis('off')
			ax.set_xticklabels([])
			ax.set_yticklabels([])
			ax.set_aspect('equal')
			plt.imshow(sample.reshape(self.size,self.size), cmap='Greys_r')
		return fig


class svhn_test():
	def __init__(self, img_size=64, one_hot=False):
		datapath = prefix + 'svhn'
		self.size = img_size
		self.one_hot = one_hot
		data = scipy.io.loadmat(os.path.join(datapath, 'test_32x32.mat'))
		self.data = data['X']
		self.label = data['y'].flatten()
		self.label[self.label==10] = 0
		self.channel = 3
		tmp = np.zeros((self.data.shape[3], self.size, self.size, self.channel))
		for i in range(self.data.shape[3]):
			tmp[i,:,:,:] = scipy.misc.imresize(self.data[:,:,:,i], [self.size, self.size])
		self.data = np.transpose(tmp/255., [0, 3, 1, 2])
		self.n_class = 10
		self.cursor = 0
		self.N = self.data.shape[0]
		print('SVHN test', self.data.shape, self.label.shape)

	def has_next(self):
		return self.cursor < self.data.shape[0]

	def reset(self):
		self.cursor = 0

	def __call__(self, batch_size):
		if self.cursor + batch_size < self.data.shape[0]:
			batch_x = self.data[self.cursor:self.cursor+batch_size]
			batch_y = self.label[self.cursor:self.cursor+batch_size]
			if self.one_hot:
				batch_y = np.eye(self.n_class)[batch_y]
		else:
			pad_n = batch_size - (self.data.shape[0] - self.cursor)
			batch_x = np.concatenate([self.data[self.cursor:], np.zeros([pad_n, self.channel, self.size, self.size])])
			if self.one_hot:
				batch_y = np.concatenate([np.eye(self.n_class)[batch_y], np.zeros((pad_n, self.n_class))])
			else:
				batch_y = np.concatenate([self.label[self.cursor:], [-1] * pad_n])
		self.cursor += batch_size

		return batch_x, batch_y

	def data2fig(self, samples):
		N_samples = samples.shape[0]
		N_row = N_col = int(np.ceil(N_samples**0.5))
		fig = plt.figure(figsize=(N_row, N_col))
		gs = gridspec.GridSpec(N_row, N_col)
		gs.update(wspace=0.05, hspace=0.05)
		samples = np.transpose(samples, [0, 2, 3, 1])

		for i, sample in enumerate(samples):
			ax = plt.subplot(gs[i])
			plt.axis('off')
			ax.set_xticklabels([])
			ax.set_yticklabels([])
			ax.set_aspect('equal')
			plt.imshow(sample.reshape(self.size,self.size), cmap='Greys_r')
		return fig


class danbooru():
	def __init__(self, one_hot=False, img_size=64):
		datapath = prefix + 'danbooru-faces'
		self.one_hot = one_hot
		self.z_dim = 100
		self.size = img_size
		self.channel = 3
		self.escape_dir = [':d', ':o']
		self.label_names = sorted(os.listdir(datapath))
		for l in self.escape_dir:
			self.label_names.remove(l)
		self.labels = {l: i for i, l in enumerate(self.label_names)}

		self.data_ = {}
		for l in self.label_names:
			tmp = {img: self.labels[l] for img in glob(os.path.join(datapath, os.path.join(l, '*')))}
			self.data_.update(tmp)
		self.image_names = list(self.data_.keys())
		self.n_class = len(self.labels)

	def __call__(self, batch_size):
		picked = np.random.choice(self.image_names, batch_size)
		batch_imgs = np.array([get_img(img, False, 128, self.size) for img in picked]).astype(np.float32)
		batch_y = [self.data_[img] for img in picked]
		if self.one_hot:
			y = np.zeros((batch_size, self.n_class))
			y[np.arange(batch_size), batch_y] = 1.
			batch_y = y
		return batch_imgs, batch_y

	def data2fig(self, samples):
		N_samples = samples.shape[0]
		N_row = N_col = int(np.ceil(N_samples**0.5))
		fig = plt.figure(figsize=(N_row, N_col))
		gs = gridspec.GridSpec(N_row, N_col)
		gs.update(wspace=0.05, hspace=0.05)
		samples = np.transpose(samples, [0, 2, 3, 1])

		for i, sample in enumerate(samples):
			ax = plt.subplot(gs[i])
			plt.axis('off')
			ax.set_xticklabels([])
			ax.set_yticklabels([])
			ax.set_aspect('equal')
			plt.imshow(sample)
		return fig


class image_folder():
	def __init__(self, datapath, img_size=None, max_hw=512, sort=False):
		self.datapath = datapath
		self.z_dim = 100
		self.size = img_size
		self.channel = 3
		self.max_hw = max_hw
		self.sort = sort
		self.data = glob(os.path.join(self.datapath, '*'))
		if self.sort:
			self.data = sorted(self.data)

		self.batch_count = 0

	def reset(self, sort=False):
		self.sort = sort
		if self.sort:
			self.data = sorted(self.data)
		else:
			np.random.shuffle(self.data)
		self.batch_count = 0

	def get_img(self, img_path):
		img = scipy.misc.imread(img_path, mode='RGB')
		h, w = img.shape[:2]
		resize = False
		if self.size:
			resize = True
			if hasattr(self.size, '__iter__'):
				resize_h, resize_w = self.size
			else:
				resize_h = resize_w = self.size
		else:
			if h >= w and h > self.max_hw:
				resize_h = self.max_hw
				resize_w = int(w / h * resize_h)
				resize = True
			elif w >= h and w > self.max_hw:
				resize_w = self.max_hw
				resize_h = int(h / w * resize_w)
				resize = True
		if resize:
			img = scipy.misc.imresize(img, [resize_h, resize_w])
			# img = scipy.misc.imresize(img, [64, 64])
		img = np.transpose(img, [2, 0, 1])/255.
		return img.astype(np.float32)

	def __call__(self, batch_size=1):
		batch_number = len(self.data)/batch_size
		if self.batch_count < batch_number-2:
			self.batch_count += 1
		else:
			self.batch_count = 0
			if not self.sort:
				np.random.shuffle(self.data)

		path_list = self.data[self.batch_count*batch_size:(self.batch_count+1)*batch_size]
		
		batch = [self.get_img(img_path) for img_path in path_list]
		batch_imgs = np.array(batch)
	
		return batch_imgs

	def data2fig(self, samples):
		N_samples = samples.shape[0]
		N_row = N_col = int(np.ceil(N_samples**0.5))
		fig = plt.figure(figsize=(N_row, N_col))
		gs = gridspec.GridSpec(N_row, N_col)
		gs.update(wspace=0.05, hspace=0.05)
		samples = np.transpose(samples, [0, 2, 3, 1])

		for i, sample in enumerate(samples):
			ax = plt.subplot(gs[i])
			plt.axis('off')
			ax.set_xticklabels([])
			ax.set_yticklabels([])
			ax.set_aspect('equal')
			plt.imshow(sample)
		return fig 



if __name__ == '__main__':
	data = celebA()
	imgs = data(20)

	fig = data.data2fig(imgs[:16])
	plt.savefig('test.png', bbox_inches='tight')
	plt.close(fig)