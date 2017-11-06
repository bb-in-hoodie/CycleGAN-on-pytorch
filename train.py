import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as t
from torch.autograd import Variable
import time

import models as m

# Settings
image_size = 128
image_location = './data/CelebA_Man2Woman/train'
checkpoint_log = 1000 # original : 1000
checkpoint_save_image = 5000
start_spurt_num = 15 # save images on the first n checkpoints

# Hyperparameters
lr_G = 0.0002 # (0.0002 on the paper)
lr_D = 0.0001
cc_lambda = 8 # lambda of cycle-consistency loss (10 on the paper)
batch_size = 1
total_epoch = 100

# Initial time
init_time = time.time()

# Discriminators and generators
gen_a = m.G_128(residual_num = 2) # residual_num : the number of residual blocks (6 was used in the paper)
dis_a = m.D_128()
gen_b = m.G_128(residual_num = 2)
dis_b = m.D_128()

def ZeroGrad():
	gen_a.zero_grad()
	dis_a.zero_grad()
	gen_b.zero_grad()
	dis_b.zero_grad()	

# Loss Function
criterion_GAN = nn.MSELoss()
criterion_CC = nn.L1Loss()

# Optimizer
gen_a_optim = optim.Adam(gen_a.parameters(), lr=lr_G)
dis_a_optim = optim.Adam(dis_a.parameters(), lr=lr_D)
gen_b_optim = optim.Adam(gen_b.parameters(), lr=lr_G)
dis_b_optim = optim.Adam(dis_b.parameters(), lr=lr_D)

print("[Network Model Information]")
print(gen_a)
print(dis_a)
print()

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
print()

# Load images (label - 0: type a, 1: type b)
transforms = t.Compose([t.Scale(image_size), t.ToTensor(), t.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_folder = torchvision.datasets.ImageFolder(root=image_location, transform=transforms)
train_loader = torch.utils.data.DataLoader(train_folder, batch_size=batch_size, shuffle=True)

# Train
for epoch in range(total_epoch):
	index = 1
	for image, label in iter(train_loader):

		# Make the image as a Variable, create ones and zeros
		image = Variable(image)

		# Activate cuda if available
		if (torch.cuda.is_available()):
			image = image.cuda()

		# Set the generators and the discriminators		
		gen_ally = None
		gen_ally_optim = None
		gen_enemy = None
		gen_enemy_optim = None
		dis_ally = None
		dis_ally_optim = None
		dis_enemy = None
		dis_enemy_optim = None

		if label[0] == 0: # a -> b -> a
			gen_ally = gen_a
			gen_ally_optim = gen_a_optim
			gen_enemy = gen_b
			gen_enemy_optim = gen_b_optim
			dis_ally = dis_a
			dis_ally_optim = dis_a_optim
			dis_enemy = dis_b
			dis_enemy_optim = dis_b_optim
		elif label[0] == 1: # b -> a -> b
			gen_ally = gen_b
			gen_ally_optim = gen_b_optim
			gen_enemy = gen_a
			gen_enemy_optim = gen_a_optim
			dis_ally = dis_b
			dis_ally_optim = dis_b_optim
			dis_enemy = dis_a
			dis_enemy_optim = dis_a_optim


		### 1. Train the ally discriminator that the current image is a real one
		dis_real_score = dis_ally(image)

		ones = Variable(torch.ones(dis_real_score.size()))
		if(torch.cuda.is_available()):
			ones = ones.cuda()

		dis_real_loss = criterion_GAN(dis_real_score, ones)

		ZeroGrad()
		dis_real_loss.backward()
		dis_ally_optim.step()

		### 2. Trian the enemy discriminator that the created image is a fake one
		fake_enemy_image = gen_enemy(image)
		# When training the enemy discriminator, use a fake image chosen from the pool
		dis_fake_score = dis_enemy(fake_enemy_image)

		zeros = Variable(torch.zeros(dis_fake_score.size()))
		if(torch.cuda.is_available()):
			zeros = zeros.cuda()

		dis_fake_loss = criterion_GAN(dis_fake_score, zeros)

		ZeroGrad()
		dis_fake_loss.backward()
		dis_enemy_optim.step()

		### 3-1. Train the enemy generator that the fake image should looks realistic
		fake_enemy_image = gen_enemy(image)
		# When training the enemy discriminator, use a fake image chosen from the pool
		dis_fake_score = dis_enemy(fake_enemy_image)

		ones = Variable(torch.ones(dis_fake_score.size()))
		if(torch.cuda.is_available()):
			ones = ones.cuda()

		gen_fake_loss = criterion_GAN(dis_fake_score, ones)

		### 3-2. Recover the current image and get cycle-consistency loss
		recovered_image = gen_ally(fake_enemy_image)
		cc_loss = criterion_CC(recovered_image, image)

		### 3-3. Sum those two losses and update the weights
		loss = gen_fake_loss + (cc_lambda * cc_loss)

		ZeroGrad()
		loss.backward()
		gen_ally_optim.step()
		gen_enemy_optim.step()

		# At each checkpoint, print the progress result and save an image
		if (index % checkpoint_log == 0):
			print("[%d, %d]-------------------------------------------"
				%(epoch, index))
			print("Discriminator real loss : %.4f, Discriminator fake loss : %.4f"
				%(dis_real_loss.data[0], dis_fake_loss.data[0]))
			print("Generator loss : %.4f, Cycle-consistency loss : %.4f * %.1f (cc_lambda)"
				%(gen_fake_loss.data[0], cc_loss.data[0], cc_lambda))

		if ((index % checkpoint_save_image == 0) or
			(epoch == 0 and index <= checkpoint_log * start_spurt_num and index % checkpoint_log == 0)):
			concat_img = torch.cat([image, fake_enemy_image])
			concat_img = concat_img.view(concat_img.size(0), 3, image_size, image_size) / 2 + 0.5 # Undo the normalization
			torchvision.utils.save_image(concat_img.data, "./result/" + str(epoch) + "_" + str(index) + ".png")

			# Printing the execution time
			exec_time = time.time() - init_time
			hours = int(exec_time/3600)
			mins = int((exec_time%3600)/60)
			secs = int((exec_time%60))
			print("====================================================")
			print("Execution time : %dh %dm %ds"%(hours, mins, secs))
			print("====================================================\n")

		index += 1

# Execution time
exec_time = time.time() - init_time
hours = int(exec_time/3600)
mins = int((exec_time%3600)/60)
secs = int((exec_time%60))
print("====================================================")
print("Total execution time : %dh %dm %ds"%(hours, mins, secs))
print("====================================================")

# Save the models
torch.save(gen_a.state_dict(), './models/gen_a.pkl')
torch.save(gen_b.state_dict(), './models/gen_b.pkl')
torch.save(dis_a.state_dict(), './models/dis_a.pkl')
torch.save(dis_b.state_dict(), './models/dis_b.pkl')