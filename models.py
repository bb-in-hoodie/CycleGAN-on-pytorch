import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

channel_size = 3

class D_128(nn.Module):
	def __init__(self):
		super(D_128, self).__init__()

		kernel = 4 # kernel width
		pad = 1 # padding size
		stride = 2 # stride		
		relu_slope = 0.2 # negative slope of LeakyReLU

		self.layer1 = nn.Sequential(
			nn.Conv2d(channel_size, 64, stride=stride, kernel_size=kernel, padding=pad),
			nn.LeakyReLU(relu_slope)
			)
		self.layer2 = nn.Sequential(
			nn.Conv2d(64, 128, stride=stride, kernel_size=kernel, padding=pad),
			nn.InstanceNorm2d(128),
			nn.LeakyReLU(relu_slope)
			)
		self.layer3 = nn.Sequential(
			nn.Conv2d(128, 256, stride=stride, kernel_size=kernel, padding=pad),
			nn.InstanceNorm2d(256),
			nn.LeakyReLU(relu_slope)
			)
		self.layer4 = nn.Sequential(
			nn.Conv2d(256, 512, stride=stride, kernel_size=kernel, padding=pad),
			nn.InstanceNorm2d(512),
			nn.LeakyReLU(relu_slope)
			)
		self.fc = nn.Linear(8*8*512, 1)

	def forward(self, x):
		act = self.layer1(x) # act: activation
		act = self.layer2(act)
		act = self.layer3(act)
		act = self.layer4(act)
		scores = F.sigmoid(self.fc(act))
		return scores


class G_128(nn.Module):
	def __init__(self):
		super(G_128, self).__init__()

	def forward(self, x):
		pass
		