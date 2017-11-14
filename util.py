import torch
import torchvision

def print_log(m, epoch, index, d_real_loss, d_fake_loss, g_fake_loss, cc_loss):
	g_lr, d_lr = m.g_lr, m.d_lr
	for group in m.g_a_optim.param_groups:
		g_lr = group['lr']
	for group in m.d_a_optim.param_groups:
		d_lr = group['lr']
	print("[%d, %d]-------------------------------------------"
		%(epoch, index))
	print("D lr: %.E, D real loss: %.4f, D fake loss: %.4f"
		%(d_lr, d_real_loss.data[0], d_fake_loss.data[0]))
	print("G lr: %.E, G loss : %.4f, CC loss : %.4f * %.1f (cc_lambda)"
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


def save_model(m):
	torch.save(m.g_a.state_dict(), './models/gen_a.pkl')
	torch.save(m.g_b.state_dict(), './models/gen_b.pkl')
	torch.save(m.d_a.state_dict(), './models/dis_a.pkl')
	torch.save(m.d_b.state_dict(), './models/dis_b.pkl')


def check_cuda_available():
	is_cuda = False
	if(torch.cuda.is_available()):
		is_cuda = True
		torch.backends.cudnn.benchmark = True
	return is_cuda
