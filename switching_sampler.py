from torch.utils.data.sampler import Sampler
import torch
import random

class SwitchingBatchSampler(Sampler):

	def __init__(self, data_source, batch_size, drop_last=False):
		self.data_source = data_source
		self.batch_size = batch_size
		self.drop_last = drop_last

		# Divide the indices into two indices groups
		self.data_len = len(self.data_source)
		count = 0
		for i in range(self.data_len):
			if self.data_source.imgs[i][1] == 1:
				break
			else:
				count += 1

		print("Total Images: %d [Class 0: %d, Class 1: %d]\n"%(self.data_len, count, (self.data_len-count)))

		self.first_size = count

		if random.uniform(0, 1) > 0.5:
			self.turn = 0
		else:
			self.turn = 1


	def __iter__(self):
		# Initialize both iters
		second_size = self.data_len - self.first_size
		self.first_iter = iter(torch.randperm(self.first_size))
		self.second_iter = iter(torch.randperm(second_size) + self.first_size)

		# Counting variables
		i = 0
		count_first = 0 # Counts how many imgs of first iter has been returned
		count_second = 0 # Counts second iter		
		batch = []

		# Until no data left, keep iterating
		while count_first+count_second < self.data_len:
			# Fill the batch
			if self.turn == 0:
				if count_first == self.first_size:
					self.turn = 1
					if len(batch) > 0 and not self.drop_last:
						yield batch
					batch = []    				
				else:
					batch.append(next(self.first_iter))
					count_first += 1
					i += 1
			else:
				if count_second == (self.data_len-self.first_size):
					self.turn = 0
					if len(batch) > 0 and not self.drop_last:
						yield batch
					batch = []    	
				else:
					batch.append(next(self.second_iter))
					count_second += 1
					i += 1
			# Yield the batch and switch the turn randomly
			if i != 0 and i % self.batch_size == 0:
				yield batch
				batch = []
				if count_first != self.first_size and count_second != second_size and random.uniform(0, 1) > 0.5:
					self.turn = (self.turn + 1) % 2

		# If drop_last is False, return the rest
		if len(batch) > 0 and not self.drop_last:
			yield batch


	def __len__(self):
		if self.drop_last:
			return (self.first_size // self.batch_size)
			+ ((self.data_len - self.first_size) // self.batch_size)
		else:
			return ((self.first_size + self.batch_size - 1) // self.batch_size)
			+ ((self.data_len - self.first_size + self.batch_size - 1) // self.batch_size)