import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as t
from torch.autograd import Variable
import time

from model import Model
import util as u
from switching_sampler import SwitchingBatchSampler
from image_buffer import ImageBuffer

# CUDA and cuDNN
is_cuda = u.check_cuda_available()

# Settings
image_size = 128
image_location = './data/CelebA_Man2Woman/train' #'./data/TestDataset'
checkpoint_log = 500
checkpoint_save_image = 5000
start_spurt_num = 15 # save images on the first n checkpoints

# Initial time
init_time = time.time()

# Create a new model
m = Model(is_cuda)

# Load images [0: Type A, 1: Type B]
transforms = t.Compose([t.Scale(image_size), t.ToTensor(), t.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_folder = torchvision.datasets.ImageFolder(root=image_location, transform=transforms)
if m.batch_size == 1:
	train_loader = torch.utils.data.DataLoader(train_folder, batch_size=m.batch_size, shuffle=True)
else:
	sampler = SwitchingBatchSampler(train_folder, m.batch_size)
	train_loader = torch.utils.data.DataLoader(train_folder, batch_sampler=sampler)

# Train
for epoch in range(m.total_epoch):
	index = 1
	image_buffer = ImageBuffer()
	m.scheduler_step()

	for image, label in iter(train_loader):

		# Make the image as a Variable, create ones and zeros
		image = Variable(image)		
		if (is_cuda):
			image = image.cuda()

		# Set the generators and the discriminators		
		if label[0] == 0: # a -> b -> a
			g_ally = m.g_a
			g_ally_optim = m.g_a_optim
			g_enemy = m.g_b
			g_enemy_optim = m.g_b_optim
			d_ally = m.d_a
			d_ally_optim = m.d_a_optim
			d_enemy = m.d_b
			d_enemy_optim = m.d_b_optim
		else: # b -> a -> b
			g_ally = m.g_b
			g_ally_optim = m.g_b_optim
			g_enemy = m.g_a
			g_enemy_optim = m.g_a_optim
			d_ally = m.d_b
			d_ally_optim = m.d_b_optim
			d_enemy = m.d_a
			d_enemy_optim = m.d_a_optim


		### 1. Train the ally discriminator that the current image is a real one
		d_real_score = d_ally(image)

		ones = Variable(torch.ones(d_real_score.size()))
		if(is_cuda):
			ones = ones.cuda()

		d_real_loss = m.criterion_GAN(d_real_score, ones)

		m.zero_grad_all()
		d_real_loss.backward()
		d_ally_optim.step()

		### 2. Trian the enemy discriminator that the created image is a fake one
		fake_enemy_image = g_enemy(image)		
		chosen_image = image_buffer.pop(fake_enemy_image)
		# When training the enemy discriminator, use a fake image chosen from the pool
		d_fake_score = d_enemy(chosen_image)

		zeros = Variable(torch.zeros(d_fake_score.size()))
		if(is_cuda):
			zeros = zeros.cuda()

		d_fake_loss = m.criterion_GAN(d_fake_score, zeros)

		m.zero_grad_all()
		d_fake_loss.backward()
		d_enemy_optim.step()

		### 3-1. Train the enemy generator that the fake image should looks realistic
		fake_enemy_image = g_enemy(image)
		# When training the enemy discriminator, use a fake image chosen from the pool
		d_fake_score = d_enemy(fake_enemy_image)

		ones = Variable(torch.ones(d_fake_score.size()))
		if(is_cuda):
			ones = ones.cuda()

		g_fake_loss = m.criterion_GAN(d_fake_score, ones)

		### 3-2. Recover the current image and get cycle-consistency loss
		recovered_image = g_ally(fake_enemy_image)
		cc_loss = m.criterion_CC(recovered_image, image)

		### 3-3. Sum those two losses and update the weights
		loss = g_fake_loss + (m.cc_lambda * cc_loss)

		m.zero_grad_all()
		loss.backward()
		g_ally_optim.step()
		g_enemy_optim.step()

		# At each checkpoint, print the log
		if (index % checkpoint_log == 0):
			u.print_log(m, epoch, index, d_real_loss, d_fake_loss, g_fake_loss, cc_loss)

		# At each checkpoint of saving image, save an image
		if ((index % checkpoint_save_image == 0) or
			(epoch == 0 and index <= checkpoint_log * start_spurt_num and index % checkpoint_log == 0)):
			u.save_image(image_size, image, fake_enemy_image, epoch, index)
			u.print_exec_time(time.time()-init_time)

		index += 1

# Print time and save the models
u.print_exec_time(time.time()-init_time, is_final=True)
u.save_model(m)