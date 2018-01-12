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
checkpoint_save = 5000
start_spurt_num = 15 # save images on the first n checkpoints

# Initial time
init_time = time.time()

# Create a new model
m = Model(is_cuda)

# Load pre-trained model
#load_path = "./models/180107-1025"
#u.load_model(m, load_path, 9, 12627)

# Create a new image buffer
use_buffer = m.buffer_size > 0
if use_buffer:
	image_buffer = ImageBuffer(m.buffer_size)

# Load images [0: Type A, 1: Type B]
transforms = t.Compose([t.Scale(image_size), t.ToTensor(), t.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_folder = torchvision.datasets.ImageFolder(root=image_location, transform=transforms)
if m.batch_size == 1:
	train_loader = torch.utils.data.DataLoader(train_folder, batch_size=m.batch_size, shuffle=True)
else:
	sampler = SwitchingBatchSampler(train_folder, m.batch_size, drop_last=True)
	train_loader = torch.utils.data.DataLoader(train_folder, batch_sampler=sampler)

# Train
for epoch in range(m.total_epoch):
	index = 1
	m.scheduler_step()

	for image, label in iter(train_loader):
		# Make an image as a Variable, create Ones vector and Zeros vector
		image = Variable(image)		
		if (is_cuda):
			image = image.cuda()

		# Set the generators and the discriminators	depending on the label of the current image	
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


		################################################################################
		# 1. Train the ally discriminator                                              #
		# : Let the ally discriminator learns that the current image is a real one     #
		################################################################################
		d_real_score = d_ally(image)
		ones = u.get_class_scores(d_real_score.size(), is_ones=True, noisy=True, is_cuda=is_cuda)
		d_real_loss = m.criterion_GAN(d_real_score, ones)

		m.zero_grad_all()
		d_real_loss.backward()
		d_ally_optim.step()


		################################################################################
		# 2. Train the enemy discriminator                                             #
		# : Let the enemy discriminator learns that the generated image is a fake one  #
		################################################################################
		fake_enemy_image = g_enemy(image)		
		if use_buffer:
			chosen_image = image_buffer.pop(fake_enemy_image) # Use a fake image popped from the image buffer
		else:
			chosen_image = fake_enemy_image	# Use the generated image without using an image buffer

		d_fake_score = d_enemy(chosen_image)
		zeros = u.get_class_scores(d_fake_score.size(), is_ones=False, noisy=True, is_cuda=is_cuda)
		d_fake_loss = m.criterion_GAN(d_fake_score, zeros)

		m.zero_grad_all()
		d_fake_loss.backward()
		d_enemy_optim.step()


		################################################################################
		# 3. The enemy generator loss                                                  #
		# : The enemy generator should generate a realistic image                      #
		################################################################################	
		fake_enemy_image = g_enemy(image)
		d_fake_score = d_enemy(fake_enemy_image)
		ones = u.get_class_scores(d_fake_score.size(), is_ones=True, noisy=True, is_cuda=is_cuda)
		g_fake_loss = m.criterion_GAN(d_fake_score, ones)


		################################################################################
		# 4. Cycle-consistency loss                                                    #
		# : The ally generator should recover the generated image                      #
		#   as the original image                                                      #
		################################################################################	
		recovered_image = g_ally(fake_enemy_image)
		cc_loss = m.criterion_CC(recovered_image, image)			

		
		################################################################################
		# 5. Train both generators                                                     #
		# : Final loss is sum of the enemy generator loss and cycle-consistency loss   #
		################################################################################
		loss = g_fake_loss + (m.cc_lambda * cc_loss)
		m.zero_grad_all()
		loss.backward(retain_graph=True)
		g_ally_optim.step()
		g_enemy_optim.step()


		################################################################################
		# 6. Total variation denoising loss                                            #
		# : The generated image should look natural & clear                            #
		################################################################################
		g_enemy_tvd_loss = u.tvd_loss(fake_enemy_image)
		if (is_cuda):
			g_enemy_tvd_loss = g_enemy_tvd_loss.cuda()

		g_ally_tvd_loss = u.tvd_loss(recovered_image)
		if (is_cuda):
			g_ally_tvd_loss = g_ally_tvd_loss.cuda()

		tvd_loss_sum = (g_enemy_tvd_loss + g_ally_tvd_loss)
		tvd_loss = tvd_loss_sum * m.tvd_lambda
			
		m.zero_grad_all()
		tvd_loss.backward()
		g_ally_optim.step()
		g_enemy_optim.step()


		# At each checkpoint, print the log
		if (index % checkpoint_log == 0):
			u.print_log(m, epoch, index, d_real_loss, d_fake_loss, g_fake_loss, cc_loss, tvd_loss_sum)

		# At each image saving checkpoint, save an image
		if ((index % checkpoint_save == 0) or
			(epoch == 0 and index <= checkpoint_log * start_spurt_num and index % checkpoint_log == 0)):
			u.save_image(m, image_size, image, fake_enemy_image, epoch, index)
			u.print_exec_time(time.time()-init_time)
			
		index += 1

	# Save the model and the image every last iteration of an epoch
	u.save_model(m, epoch, index)
	u.save_image(m, image_size, image, fake_enemy_image, epoch, index)

# Print the execution time and save the models
u.print_exec_time(time.time()-init_time, is_final=True)