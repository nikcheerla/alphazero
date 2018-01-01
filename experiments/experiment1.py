
import sys, os, random, time

from itertools import product
import numpy as np
import IPython

from modules import TrainableModel
from games.chessboard import ChessGame

import multiprocessing
from multiprocessing import Pool, Manager

from alphazero import AlphaZeroController

import torch
import torch.nn as nn
import torch.nn.functional as F



""" CNN representing estimated value for each board state.
"""

class Net(TrainableModel):

	def __init__(self):

		super(TrainableModel, self).__init__()
		self.conv1 = nn.Conv2d(11, 32, kernel_size=(3, 3), padding=(1, 1))
		self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))
		self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1), dilation=2)
		self.conv4 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(2, 2), dilation=2)
		self.linear = nn.Linear(64, 1)
	
	def loss(self, data, data_pred):
		Y_pred = data_pred["target"]
		Y_target = data["target"]

		return F.mse_loss(Y_pred, Y_target)

	def forward(self, x):
		x = x['input']
		x = x.permute(0, 3, 1, 2)
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = F.dropout(x, p=0.2, training=self.training)
		x = F.relu(self.conv3(x))
		x = F.relu(self.conv4(x))
		x = x.mean(dim=2).mean(dim=2)
		x = self.linear(x)
		return {'target': x}


if __name__ == "__main__":

	manager = Manager()
	model = Net()
	model.compile(torch.optim.Adadelta, lr=0.3)
	controller = AlphaZeroController(manager, model)

	for i in range(0, 1000):
		game = ChessGame()
		print (game)
		print ()

		while not game.over():
			game.make_move(controller.best_move(game, playouts=100))

			print (game)
			print ()

