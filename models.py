import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

# Hyperparameters
channel_size = 3
residual_num = 6

# Discriminator for a 128 * 128 image
class D_128(nn.Module):
	def __init__(self):
		super(D_128, self).__init__()

		kernel = 4 # kernel width
		pad = 1 # padding size
		stride = 2 # stride		
		relu_slope = 0.01 # negative slope of LeakyReLU (0.2 was used in the paper)

		self.layer1 = nn.Sequential(
			nn.Conv2d(channel_size, 64, kernel_size=kernel, stride=stride, padding=pad),
			nn.LeakyReLU(relu_slope)
			)
		self.layer2 = nn.Sequential(
			nn.Conv2d(64, 128, kernel_size=kernel, stride=stride, padding=pad),
			nn.InstanceNorm2d(128, affine=True),
			nn.LeakyReLU(relu_slope)
			)
		self.layer3 = nn.Sequential(
			nn.Conv2d(128, 256, kernel_size=kernel, stride=stride, padding=pad),
			nn.InstanceNorm2d(256, affine=True),
			nn.LeakyReLU(relu_slope)
			)
		self.layer4 = nn.Sequential(
			nn.Conv2d(256, 512, kernel_size=kernel, stride=stride, padding=pad),
			nn.InstanceNorm2d(512, affine=True),
			nn.LeakyReLU(relu_slope)
			)
		self.score = nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1)

	def forward(self, x):
		act = self.layer1(x) # act: activation
		act = self.layer2(act)
		act = self.layer3(act)
		act = self.layer4(act)
		result = self.score(act)
		return result


# Generator for a 128 * 128 image
class G_128(nn.Module):

	def __init__(self, residual_num = 6):
		super(G_128, self).__init__()
		self.residual_num = residual_num

		self.conv_down = nn.Sequential(
			nn.ReflectionPad2d(3),
			nn.Conv2d(channel_size, 32, kernel_size=7, stride=1, padding=0),
			nn.InstanceNorm2d(32, affine=True),
			nn.ReLU()
			)
		self.down1 = nn.Sequential(
			nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
			nn.InstanceNorm2d(64, affine=True),
			nn.ReLU()
			)
		self.down2 = nn.Sequential(
			nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
			nn.InstanceNorm2d(128, affine=True),
			nn.ReLU()
			)

		# Store residual blocks into a list
		self.residual_list = []
		for i in range(self.residual_num):
			nn_seq = nn.Sequential(
					nn.ReflectionPad2d(1),
					nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0),
					nn.InstanceNorm2d(128, affine=True),
					nn.ReLU(),
					nn.ReflectionPad2d(1),
					nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0),
					nn.InstanceNorm2d(128, affine=True)
					)
			if(torch.cuda.is_available()):
				nn_seq = nn_seq.cuda()
			self.residual_list.append(nn_seq) 

		self.up1 = nn.Sequential(
			nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
			nn.InstanceNorm2d(64, affine=True),
			nn.ReLU()
			)
		self.up2 = nn.Sequential(
			nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
			nn.InstanceNorm2d(32, affine=True),
			nn.ReLU()
			)
		self.conv_up = nn.Sequential(
			nn.ReflectionPad2d(3),
			nn.Conv2d(32, channel_size, kernel_size=7, stride=1, padding=0),
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