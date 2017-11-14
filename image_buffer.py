from torch.autograd import Variable
import torch
import random

class ImageBuffer:
	def __init__(self, buffer_size=50):
		self._buffer_size = buffer_size
		self._buffer = []

	def pop(self, images):
		return_images = []

		for image in images.data:

			image = torch.unsqueeze(image, 0)

			if len(self._buffer) < self._buffer_size:
				# If the buffer is not full, appends it
				self._buffer.append(image)
				return_images.append(image)
			else:
				# If the buffer is full, randomly exchange an element of the buffer or just return it
				p = random.uniform(0, 1)
				if p > 0.5:
					index = random.randint(0, len(self._buffer)-1)
					temp = self._buffer[index].clone()
					self._buffer[index] = image
					return_images.append(temp)
				else:
					return_images.append(image)

		return Variable(torch.cat(return_images, 0))