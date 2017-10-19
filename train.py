import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as t
from torch.autograd import Variable
import time

import models as m

learning_rate = 0.0001
iter_num = 300

# Initial time
init_time = time.time()

# Discriminator and generator
dis_b = m.D_128()
gen_b = m.G_128()
dis_w = m.D_128()
gen_w = m.G_128()