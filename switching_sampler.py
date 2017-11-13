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

		print("Total Images: %d [Class 0: %d, Class 1: %d]"%(self.data_len, count, (self.data_len-count)))

		# always first half should smaller than second half
		if count >= (self.data_len/2):
			self.first_iter = iter(torch.randperm(self.data_len - count) + count)
			self.second_iter = iter(torch.randperm(count))
		else:
			self.first_iter = iter(torch.randperm(count))
			self.second_iter = iter(torch.randperm(self.data_len - count) + count)

		self.first_size = count
		self.turn = 0

	def __iter__(self):
		count = 0 # Counts how many imgs of first iter has been returned
		batch = []
		for i in range(self.data_len):
			# Fill the batch
			if self.turn == 0:
				if count == self.first_size:
					self.turn = 1
					if len(batch) > 0 and not self.drop_last:
						yield batch
					batch = []    				
				else:
					batch.append(next(self.first_iter))
					count += 1
			else:
				batch.append(next(self.second_iter))

			# Yield the batch and switch the turn randomly
			if (i+1) % self.batch_size == 0:
				yield batch
				batch = []
				if count != self.first_size and random.random() > 0.5:
					self.turn = (self.turn + 1) % 2

		# If drop_last is False, return the rest
		if len(batch) > 0 and not self.drop_last:
			yield batch
	'''
	def __iter__(self):
	    batch = []
	    for idx in self.sampler:
	        batch.append(idx)
	        if len(batch) == self.batch_size:
	            yield batch
	            batch = []
	    if len(batch) > 0 and not self.drop_last:
	        yield batch
	'''
	def __len__(self):
		if self.drop_last:
			return (len(self.first_size) // self.batch_size)
			+ (len(self.data_len - self.first_size) // self.batch_size)
		else:
			return ((len(self.first_size) + self.batch_size - 1) // self.batch_size)
			+ ((len(self.data_len - self.first_size) + self.batch_size - 1) // self.batch_size)

'''
    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self):
        return self.data_len
'''