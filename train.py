import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as t
from torch.autograd import Variable
import time

import models as m

# Hyperparameters
image_size = 128
image_location = './data/CelebA_Man2Woman/train'

lr_G = 0.0002
lr_D = 0.0001
lambda_cyc = 10
batch_size = 1
total_epoch = 3

# Initial time
init_time = time.time()

# Discriminator and generator
gen_a = m.G_128()
dis_a = m.D_128()
gen_b = m.G_128()
dis_b = m.D_128()

def ZeroGrad():
	gen_a.zero_grad()
	dis_a.zero_grad()
	gen_b.zero_grad()
	dis_b.zero_grad()

# If cuda is available, activate it
print("[CUDA Information]")
if(torch.cuda.is_available()):
	print("CUDA is available : Activate CUDA fuctionality")
	gen_a = gen_a.cuda()
	dis_a = dis_a.cuda()
	gen_b = gen_b.cuda()
	dis_b = dis_b.cuda()
else:
	print("CUDA is not available")

# Load images (label - 0: type a, 1: type b)
transforms = t.Compose([t.Scale(image_size), t.ToTensor()])
train_folder = torchvision.datasets.ImageFolder(root = image_location, transform = transforms)
train_loader = torch.utils.data.DataLoader(train_folder, batch_size = batch_size, shuffle = True)

# Optimizer
gen_a_optim = optim.Adam(gen_a.parameters(), lr=lr_G)
dis_a_optim = optim.Adam(dis_a.parameters(), lr=lr_D)
gen_b_optim = optim.Adam(gen_b.parameters(), lr=lr_G)
dis_b_optim = optim.Adam(dis_b.parameters(), lr=lr_D)

print("[Network Model Information]")
print(gen_a)
print(dis_a)
print()

# Train
for epoch in range(total_epoch):
	for image, label in iter(train_loader):
		if label == 0: # type a
			pass
		elif label == 1: # type b
			pass
