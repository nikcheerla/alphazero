
import numpy as np

import torch
from torch.autograd import Variable

import os, sys, random, itertools, numbers
import IPython


"""Generates array batches from generator.
"""

def batch(data_generator, targets=None, batch_size=128, limit=None):

	if targets == None: targets = next(iter(data_generator)).keys()
	data = {target: [] for target in targets}

	for i, item in enumerate(data_generator):
		for target in targets:
			if target in item:
				data[target].append(item[target])
			else:
				data[target].append(None)

		if limit is not None and i == limit:
			return

		if (i+1) % batch_size == 0:
			data = {target: np.array(data[target]) for target in targets}
			yield data
			data = {target: [] for target in targets}

def stack(batch_generator, targets=None):
	
	if targets == None: targets = next(iter(batch_generator)).keys()
	data = {target: [] for target in targets}
	
	for i, item in enumerate(batch_generator):
		for target in targets:
			if target in item:
				data[target].append(item[target])
			else:
				raise Exception("Batch has no examples of given target.")

	data = {target: np.concatenate(data[target], axis=0) for target in targets}
	return data

def cast(array):
	return array.astype(type(array[0]))

def load_in_torch(data):

	if np.issubdtype(data.dtype, int):
		x = Variable(torch.LongTensor(data.astype(int)), requires_grad=False)
	else:
		x = Variable(torch.FloatTensor(data*1.0), requires_grad=False)
	if torch.cuda.is_available():
		x = x.cuda()
	return x


def numpify(data):

	if isinstance(data, Variable):
		return data.cpu().data.numpy()
	elif isinstance(data, dict):
		for key in data:
			data[key] = numpify(data[key])
	elif isinstance(data, list):
		for i in range(0, len(data)):
			data[i] = numpify(data[i])
	elif isinstance(data, tuple):
		data = tuple((numpify(x) for x in data))

	return data


def torchify(data):

	if isinstance(data, np.ndarray):
		return load_in_torch(data)
	elif isinstance(data, dict):
		for key in data:
			data[key] = torchify(data[key])
	elif isinstance(data, list):
		for i in range(0, len(data)):
			data[i] = torchify(data[i])
	elif isinstance(data, tuple):
		data = tuple((torchify(x) for x in data))

	return data



