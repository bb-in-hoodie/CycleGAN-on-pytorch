import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

# Hyperparameters
channel_size = 3

# Discriminator for a 128 * 128 image
class D_128(nn.Module):
	def __init__(self, first_kernels=64, kernel=4, pad=1, stride=2, relu_slope=0.01, norm="instance"):
		super(D_128, self).__init__()

		self.first_kernels = first_kernels
		self.kernel = kernel # kernel width
		self.pad = pad # padding size
		self.stride = stride # stride		
		self.relu_slope = relu_slope # negative slope of LeakyReLU (0.2 was used in the paper)
		# [instance: InstanceNorm2d, batch: BatchNorm2d]
		if norm == "instance":
			self.norm = nn.InstanceNorm2d
		else:
			self.norm = nn.BatchNorm2d

		self.layer1 = nn.Sequential(
			nn.Conv2d(channel_size, first_kernels, kernel_size=kernel, stride=stride, padding=pad), # 128 -> 64
			nn.LeakyReLU(relu_slope)
			)
		self.layer2 = nn.Sequential(
			nn.Conv2d(first_kernels, first_kernels*2, kernel_size=kernel, stride=stride, padding=pad), # 64 -> 32
			self.norm(first_kernels*2, affine=True),
			nn.LeakyReLU(relu_slope)
			)
		self.layer3 = nn.Sequential(
			nn.Conv2d(first_kernels*2, first_kernels*4, kernel_size=kernel, stride=stride, padding=pad), # 32 -> 16
			self.norm(first_kernels*4, affine=True),
			nn.LeakyReLU(relu_slope)
			)
		self.layer4 = nn.Sequential(
			nn.Conv2d(first_kernels*4, first_kernels*8, kernel_size=kernel, stride=stride, padding=pad), # 16 -> 8
			self.norm(first_kernels*8, affine=True),
			nn.LeakyReLU(relu_slope)
			)
		self.layer5 = nn.Sequential(
			nn.Conv2d(first_kernels*8, first_kernels*16, kernel_size=kernel, stride=stride, padding=pad), # 8 -> 4
			self.norm(first_kernels*16, affine=True),
			nn.LeakyReLU(relu_slope)
			)
		self.score = nn.Conv2d(first_kernels*16, 1, kernel_size=3, stride=1, padding=1) # 4 -> 4

	def forward(self, x):
		act = self.layer1(x) # act: activation
		act = self.layer2(act)
		act = self.layer3(act)
		act = self.layer4(act)
		act = self.layer5(act)
		result = self.score(act)
		return result


# Generator for a 128 * 128 image
class G_128(nn.Module):
	def __init__(self, first_kernels=32, residual_num=6, norm="instance"):
		super(G_128, self).__init__()

		self.first_kernels = first_kernels
		self.residual_num = residual_num
		# [instance: InstanceNorm2d, batch: BatchNorm2d]
		if norm == "instance":
			self.norm = nn.InstanceNorm2d
		else:
			self.norm = nn.BatchNorm2d

		self.conv_down = nn.Sequential(
			nn.ReflectionPad2d(3),
			nn.Conv2d(channel_size, first_kernels, kernel_size=7, stride=1, padding=0),
			self.norm(first_kernels, affine=True),
			nn.ReLU()
			)
		self.down1 = nn.Sequential(
			nn.Conv2d(first_kernels, first_kernels*2, kernel_size=3, stride=2, padding=1),
			self.norm(first_kernels*2, affine=True),
			nn.ReLU()
			)
		self.down2 = nn.Sequential(
			nn.Conv2d(first_kernels*2, first_kernels*4, kernel_size=3, stride=2, padding=1),
			self.norm(first_kernels*4, affine=True),
			nn.ReLU()
			)

		# Store residual blocks into a list
		self.residual_list = []
		for i in range(self.residual_num):
			nn_seq = nn.Sequential(
					nn.ReflectionPad2d(1),
					nn.Conv2d(first_kernels*4, first_kernels*4, kernel_size=3, stride=1, padding=0),
					self.norm(first_kernels*4, affine=True),
					nn.ReLU(),
					nn.ReflectionPad2d(1),
					nn.Conv2d(first_kernels*4, first_kernels*4, kernel_size=3, stride=1, padding=0),
					self.norm(first_kernels*4, affine=True)
					)
			if(torch.cuda.is_available()):
				nn_seq = nn_seq.cuda()
			self.residual_list.append(nn_seq) 

		self.up1 = nn.Sequential(
			nn.ConvTranspose2d(first_kernels*4, first_kernels*2, kernel_size=3, stride=2, padding=1, output_padding=1),
			self.norm(first_kernels*2, affine=True),
			nn.ReLU()
			)
		self.up2 = nn.Sequential(
			nn.ConvTranspose2d(first_kernels*2, first_kernels, kernel_size=3, stride=2, padding=1, output_padding=1),
			self.norm(first_kernels, affine=True),
			nn.ReLU()
			)
		self.conv_up = nn.Sequential(
			nn.ReflectionPad2d(3),
			nn.Conv2d(first_kernels, channel_size, kernel_size=7, stride=1, padding=0),
			nn.Tanh()
			)

	def forward(self, x):
		act = self.conv_down(x) # act: activation
		act = self.down1(act)
		act = self.down2(act)
		for i in range(self.residual_num):		
			act = F.relu(self.residual_list[i](act) + act)
		act = self.up1(act)
		act = self.up2(act)
		result = self.conv_up(act)
		return result		