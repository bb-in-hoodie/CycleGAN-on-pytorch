import torch
import torchvision
from torch.autograd import Variable

def get_class_scores(size, is_ones, noisy=False, is_cuda=False, noise_width=0.3):
	scores = None

	if noisy:
		scores = torch.rand(size) * noise_width # [0, 0.3)
		if is_ones:
			scores += (1.1-noise_width) # [0.8, 1.1)
	else:
		if is_ones:
			scores = torch.ones(size) # only 1
		else:
			scores = torch.zeros(size) # only 0
			
	scores = Variable(scores)

	if is_cuda:
		scores = scores.cuda()

	return scores

def print_log(m, epoch, index, d_real_loss, d_fake_loss, g_fake_loss, cc_loss):
	g_lr, d_lr = m.g_lr, m.d_lr
	for group in m.g_a_optim.param_groups:
		g_lr = group['lr']
	for group in m.d_a_optim.param_groups:
		d_lr = group['lr']
	print("[%d, %d]-------------------------------------------"
		%(epoch, index))
	print("D lr: %1.1E, D real loss: %.4f, D fake loss: %.4f"
		%(d_lr, d_real_loss.data[0], d_fake_loss.data[0]))
	print("G lr: %1.1E, G loss : %.4f, CC loss : %.4f * %.1f (cc_lambda)"
		%(g_lr, g_fake_loss.data[0], cc_loss.data[0], m.cc_lambda))


def save_image(image_size, image, fake_enemy_image, epoch, index):		
	concat_img = []
	image_num = image.size(0)
	fake_image_num = fake_enemy_image.size(0)

	for i in range(image_num):
		if (i < fake_image_num):
			concat_img.append(image[i])
			concat_img.append(fake_enemy_image[i])

	concat_size = len(concat_img)
	concat_img = torch.cat(concat_img)
	concat_img = concat_img.view(concat_size, 3, image_size, image_size) / 2 + 0.5
	torchvision.utils.save_image(concat_img.data, "./result/" + str(epoch) + "_" + str(index) + ".png")


def print_exec_time(exec_time, is_final=False):		
	hours = int(exec_time/3600)
	mins = int((exec_time%3600)/60)
	secs = int((exec_time%60))
	print("====================================================")
	if is_final:			
		print("Total execution time : %dh %dm %ds"%(hours, mins, secs))
	else:
		print("Execution time : %dh %dm %ds"%(hours, mins, secs))
	print("====================================================\n")


def save_model(m, epoch, index):
	torch.save(m.g_a.state_dict(), './models/' + str(epoch) + '_' + str(index) + '_gen_a.pkl')
	torch.save(m.g_b.state_dict(), './models/' + str(epoch) + '_' + str(index) + '_gen_b.pkl')
	torch.save(m.d_a.state_dict(), './models/' + str(epoch) + '_' + str(index) + '_dis_a.pkl')
	torch.save(m.d_b.state_dict(), './models/' + str(epoch) + '_' + str(index) + '_dis_b.pkl')

def load_model(m, path):
	m.g_a.load_state_dict(torch.load(path+'/gen_a.pkl'))
	m.g_b.load_state_dict(torch.load(path+'/gen_b.pkl'))
	m.d_a.load_state_dict(torch.load(path+'/dis_a.pkl'))
	m.d_b.load_state_dict(torch.load(path+'/dis_b.pkl'))


def check_cuda_available():
	is_cuda = False
	if(torch.cuda.is_available()):
		is_cuda = True
		torch.backends.cudnn.benchmark = True
	return is_cuda
