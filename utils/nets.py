import torch
import torch.nn as nn
from torch.autograd import Variable


def compute_conv_output_and_padding_param(in_h, in_w, stride, kernel_size):
	out_h, out_w = (in_h + stride - 1) // stride, (in_w + stride - 1) // stride
	padding_h = (stride * out_h - stride + kernel_size - in_h + 1) // 2
	padding_w = (stride * out_w - stride + kernel_size - in_w + 1) // 2
	return (padding_h, padding_w), (out_h, out_w)

def compute_deconv_padding_param(in_h, in_w, out_h, out_w, stride, kernel_size):
	padding_h = padding_w = kernel_size // 2
	output_padding_h = out_h - (in_h - 1) * stride + 2 * padding_h - kernel_size
	output_padding_w = out_w - (in_w - 1) * stride + 2 * padding_w - kernel_size
	return (padding_h, padding_w), (output_padding_h, output_padding_w)


class D_conv_classification(nn.Module):
	def __init__(self, n_class, channel=3, img_size=64, last_layer_with_activation=True, stride=2, kernel_size=3):
		super(D_conv_classification, self).__init__()
		self.n_class = n_class
		self.channel = channel
		self.last_layer_with_activation = last_layer_with_activation
		if hasattr(img_size, '__iter__'):
			self.img_size = img_size
		else:
			self.img_size = (img_size, img_size)
		self.stride = stride
		self.kernel_size = kernel_size

		padding1, out1 = compute_conv_output_and_padding_param(self.img_size[0], self.img_size[1], self.stride, self.kernel_size)
		self.conv1 = nn.Conv2d(self.channel, 64, kernel_size=self.kernel_size, stride=self.stride, padding=padding1, bias=False)
		self.bn1 = nn.BatchNorm2d(64)

		padding2, out2 = compute_conv_output_and_padding_param(out1[0], out1[1], self.stride, self.kernel_size)
		self.conv2 = nn.Conv2d(64, 128, kernel_size=self.kernel_size, stride=self.stride, padding=padding2, bias=False)
		self.bn2 = nn.BatchNorm2d(128)

		padding3, out3 = compute_conv_output_and_padding_param(out2[0], out2[1], self.stride, self.kernel_size)
		self.conv3 = nn.Conv2d(128, 256, kernel_size=self.kernel_size, stride=self.stride, padding=padding3, bias=False)
		self.bn3 = nn.BatchNorm2d(256)

		padding4, out4 = compute_conv_output_and_padding_param(out3[0], out3[1], self.stride, self.kernel_size)
		self.conv4 = nn.Conv2d(256, 512, kernel_size=self.kernel_size, stride=self.stride, padding=padding4, bias=False)
		self.bn4 = nn.BatchNorm2d(512)

		self.d_ = self.infer_size((self.channel, self.img_size[0], self.img_size[1]))
		self.linear1 = nn.Linear(self.d_, 256, bias=True)
		self.linear2 = nn.Linear(256, self.n_class, bias=True)
		self.lrelu = nn.LeakyReLU(negative_slope=0.2)
		self.sigmoid = nn.Sigmoid()

		# initialize weights
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				m.weight.data.normal_(0, 0.02)
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.normal_(1.0, 0.02)
				m.bias.data.fill_(0)
			elif isinstance(m, nn.Linear):
				m.weight.data.normal_(0, 0.02)
				m.bias.data.fill_(0)

	def infer_size(self, shape):
		x = Variable(torch.rand(1, *shape))
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.conv4(x)
		return int(x.data.view(1, -1).size(1))

	def forward(self, x):
		d = self.lrelu(self.bn1(self.conv1(x)))
		d = self.lrelu(self.bn2(self.conv2(d)))
		d = self.lrelu(self.bn3(self.conv3(d)))
		d = self.lrelu(self.bn4(self.conv4(d)))
		d = d.view(-1, self.d_)
		d = self.lrelu(self.linear1(d))
		d = self.linear2(d)
		if self.last_layer_with_activation:
			d = self.sigmoid(d)
		return d


class D_MLP(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(D_MLP, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size

		self.linear1 = nn.Linear(self.input_size, self.hidden_size, bias=True)
		self.linear2 = nn.Linear(self.hidden_size, self.output_size, bias=True) 
		self.lrelu = nn.LeakyReLU(negative_slope=0.2)
		self.sigmoid = nn.Sigmoid()

		# initialize weights
		for m in self.modules():
			if isinstance(m, nn.Linear):
				m.weight.data.normal_(0, 0.02)
				m.bias.data.fill_(0)

	def forward(self, x):
		x = self.lrelu(self.linear1(x))
		x = self.sigmoid(self.linear2(x))
		return x
